import copy
import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.node import map_arg

from .utils import get_arg_or_kwarg, _pair
from ..mapping import (
    get_parameter_or_buffer,
    propagate_shape,
    replace_node_with_graph_module,
    get_node_bytes,
    _nodes_sequential,
)
from ..mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_gemm_op,
    is_linear,
    is_matmul,
    is_prunable_op,
)
from ...pt2e_utils import fetch_attr
from ...quantize_pt2e import create_getattr_from_value, export_model

logger = logging.getLogger(__name__)

__all__ = [
    "run_matrix_op_l2_tiling",
    "run_vector_op_l2_tiling",
    "run_vector_op_node_l2_tiling",
]

DEFAULT_CACHE_SIZE = 8 * 1024 * 1024  # 8 MiB


def create_new_chain(model, node_to_fuse, cat_node, fusable):
    conv_node = node_to_fuse if is_conv2d(node_to_fuse) else node_to_fuse.args[0]
    input_shape = conv_node.meta["tiled_shapes"]["input"]
    output_shape = conv_node.meta["tiled_shapes"]["output"]

    slice_args = []
    input_nodes = conv_node.args[0]
    while input_nodes.target in [
        torch.ops.aten.slice.Tensor, torch.ops.aten.pad.default
    ]:
        if input_nodes.target == torch.ops.aten.slice.Tensor:
            dim, start, end = input_nodes.args[1:]
            if dim in (2, 3):
                tile_idx = round(float(start) / input_shape[dim])
                tile_start = tile_idx * output_shape[dim]
                tile_end = tile_start + output_shape[dim]
                slice_args.insert(0, (dim, tile_start, tile_end))
        input_nodes = input_nodes.args[0]

    value_remap = {cat_node: node_to_fuse}
    anchor = node_to_fuse.next
    for n in fusable:
        for arg in n.all_input_nodes:
            if arg in value_remap:
                continue

            if arg.op == "get_attr":
                param = fetch_attr(model, arg.target)
                with model.graph.inserting_before(anchor):
                    get_attr = create_getattr_from_value(
                        model, model.graph, arg.name, param
                    )
                propagate_shape(get_attr, model)
                get_attr.meta["dtype"] = arg.meta.get("dtype")
                value_remap[arg] = get_attr
                continue

            slice_node = arg
            for dim, start, end in slice_args:
                with model.graph.inserting_before(node_to_fuse):
                    slice_node = model.graph.call_function(
                        torch.ops.aten.slice.Tensor,
                        (slice_node, dim, start, end),
                    )
                propagate_shape(slice_node, model)
                slice_node.meta["dtype"] = arg.meta.get("dtype")
            value_remap[arg] = slice_node

        with model.graph.inserting_before(anchor):
            new_node = model.graph.node_copy(
                n, lambda n: value_remap.get(n, n)
            )
        propagate_shape(new_node, model)
        new_node.meta["dtype"] = n.meta.get("dtype")
        if (source_fn_st := n.meta.get("source_fn_stack")) is not None:
            new_node.meta["source_fn_stack"] = [
                (new_node.name, source_fn_st[0][1])
            ]
        value_remap[n] = new_node

    for user in list(node_to_fuse.users):
        if user != value_remap[fusable[0]]:
            user.replace_input_with(node_to_fuse, value_remap[fusable[-1]])


def move_fusable_ops_after_conv2d(model, node):
    fusable_ops = []
    next_node = next(iter(node.users))
    while is_elementwise_op(next_node):
        fusable_ops.append(next_node)
        if len(next_node.users) != 1:
            break
        next_node = next(iter(next_node.users))

    if not fusable_ops:
        return

    order = {n: i for i, n in enumerate(model.graph.nodes)}
    if not _nodes_sequential([node] + fusable_ops, order):
        return

    # Find all the conv2d nodes to fuse with
    conv2d_nodes = []
    cat_and_slice_nodes = []
    stack = node.all_input_nodes[:]
    while stack:
        curr = stack.pop()
        if curr.target in [
            torch.ops.aten.cat.default, torch.ops.aten.slice.Tensor,
        ]:
            cat_and_slice_nodes.append(curr)
            stack.extend(curr.all_input_nodes)
        else:
            conv2d_nodes.append(curr)

    for conv_node in conv2d_nodes:
        create_new_chain(model, conv_node, node, fusable_ops)

    for n in cat_and_slice_nodes:
        n.meta["dtype"] = fusable_ops[-1].meta.get("dtype")

    fusable_ops[-1].replace_all_uses_with(node)
    for n in reversed(fusable_ops):
        model.graph.erase_node(n)


def split_conv2d_node(model, node, tiling):
    """
    Replace a conv2d node with a tiled conv2d subgraph.

    Args:
        model: GraphModule
        node: node (must be aten.conv2d or quantized conv2d)
        tiling: (Y, X, K, C)
            - Y: number of tiles along kernel height
            - X: number of tiles along kernel width
            - K : number of tiles along output channels
            - C : number of tiles along input channels
    """
    stride = get_arg_or_kwarg(node, 3, "stride", 1)
    padding = get_arg_or_kwarg(node, 4, "padding", 0)
    dilation = get_arg_or_kwarg(node, 5, "dilation", 1)
    groups = get_arg_or_kwarg(node, 6, "groups", 1)
    bs = node.kwargs.get("block_size", 1)
    bs = bs if isinstance(bs, int) else bs[1]

    is_conv1 = (
        node.shape[2] == 112
        and node.shape[3] == 112
        and stride == [2,2]
        and padding == [3, 3]
    )
    is_dwc = is_depthwise_conv(node)

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    N, K, Y, X = node.shape
    _, C, kH, kW = node.args[1].shape
    _, _, IX, IY = node.args[0].shape

    tile_y, tile_x, tile_c, tile_k = tiling

    tiled_shapes = []
    tiling = (1, K // tile_k, 1, 1)
    tile_configs = []
    for y in range(0, Y, tile_y):
        for x in range(0, X, tile_x):
            oh = min(tile_y, Y - y)
            ow = min(tile_x, X - x)

            if tile_x == X and tile_y == Y:
                # no need to tile
                y_in_start_clamped = 0
                y_in_end_clamped = IY
                x_in_start_clamped = 0
                x_in_end_clamped = IX
                pad_top = 0
                pad_left = 0
                pad_bottom = 0
                pad_right = 0
                kernel_padding = padding
                x_out_valid_start = 0
                y_out_valid_start = 0
            else:
                # Compute receptive field in input
                y_in_start = y * stride[0] - padding[0]
                y_in_end = (
                    y_in_start + (oh - 1) * stride[0] + (kH - 1) * dilation[0] + 1
                )
                x_in_start = x * stride[1] - padding[1]
                x_in_end = (
                    x_in_start + (ow - 1) * stride[1] + (kW - 1) * dilation[1] + 1
                )
                # include even more padding so that after applying padding of
                # three, the new convolution still aligns with the original one,
                # just with some garbage on the right and bottom side
                if is_conv1:
                    y_in_start = y_in_start - (padding[0] % stride[0])
                    x_in_start = x_in_start - (padding[1] % stride[1])
                
                if is_dwc:
                    # adjust the receptive field such that is is mutiple of stride
                    # this required by the dwc hardware
                    rem = (y_in_end - y_in_start) % stride[0]
                    if rem:
                        y_in_end += stride[0] - rem
                    rem = (x_in_end - x_in_start) % stride[1]
                    if rem:
                        x_in_end += stride[1] - rem

                y_in_start_clamped = max(y_in_start, 0)
                x_in_start_clamped = max(x_in_start, 0)
                
                if not is_conv1:
                    x_in_end_clamped = min(x_in_end, IX)
                    y_in_end_clamped = min(y_in_end, IY)
                    # Pad input locally if receptive field goes outside
                    pad_top  = y_in_start_clamped - y_in_start
                    pad_left = x_in_start_clamped - x_in_start
                    pad_bottom = y_in_end - y_in_end_clamped
                    pad_right = x_in_end - x_in_end_clamped
                    no_padding = (
                        pad_top == 0
                        and pad_left == 0
                        and pad_bottom == 0
                        and pad_right == 0
                    )
                    # if there's no padding, and the receptive field is smaller
                    # than output x stride (pixels at the right and bottom
                    # boundary are not used by the kernel), we need to expand the
                    # receptive field to make sure the input size is output x stride
                    # this is a requirement by the hardware for the downsample layer
                    if no_padding and kH == 1 and kW == 1:
                        if (y_in_end_clamped - y_in_start_clamped) % stride[0] != 0:
                            y_in_end_clamped = oh * stride[0] + y_in_start_clamped
                        if (x_in_end_clamped - x_in_start_clamped) % stride[1] != 0:
                            x_in_end_clamped = ow * stride[1] + x_in_start_clamped
                    x_out_valid_start = 0
                    y_out_valid_start = 0
                    # we are explicitly padding the input, remove the kernel padding
                    kernel_padding = (0, 0)
                else:
                    # adjust the receptive field to multiple of 8
                    # this is required by the hardware conv1 replication
                    y_in_end = y_in_end + (16 - ((y_in_end - y_in_start_clamped) % 16))
                    x_in_end = x_in_end + (16 - ((x_in_end - x_in_start_clamped) % 16))
                    y_in_end_clamped = min(y_in_end, IY)
                    x_in_end_clamped = min(x_in_end, IX)
                    # we will never pad on the left and top side
                    pad_top = 0
                    pad_left = 0
                    # feel free to pad on the bottom and right side, just throw away the extra output
                    pad_bottom = y_in_end - y_in_end_clamped
                    pad_right = x_in_end - x_in_end_clamped
                    # calculate the number of pixels we need to throw away on the top and left side
                    x_out_valid_start = (x * stride[1] - x_in_start_clamped) // stride[1]
                    y_out_valid_start = (y * stride[0] - y_in_start_clamped) // stride[0]
                    # maintain original padding for conv1
                    kernel_padding = padding 

            for c_start in range(0, C, tile_c):
                c_end = min(c_start + tile_c, C)
                tile_configs.append({
                    "b_tile": (0, N),
                    "c_tile": (c_start, c_end),
                    "y_tile": (y_in_start_clamped, y_in_end_clamped),
                    "x_tile": (x_in_start_clamped, x_in_end_clamped),
                    "input_padding": (pad_top, pad_left, pad_bottom, pad_right),
                    "kernel_padding": kernel_padding,
                    "out_valid_start": (y_out_valid_start, x_out_valid_start),
                    "out_tile": (oh, ow),
                    "keep_dims_and_padding": is_conv1,
                })
                input_height = y_in_end_clamped - y_in_start_clamped + pad_top + pad_bottom
                input_width = x_in_end_clamped - x_in_start_clamped + pad_left + pad_right
                output_height = (input_height + 2 * kernel_padding[0] - kH) // stride[0] + 1
                output_width = (input_width + 2 * kernel_padding[1] - kW) // stride[1] + 1
                assert output_height >= oh and output_width >= ow, (
                    f"Output height {output_height} is less than tile height {oh}"
                )

                tiled_shape = {}
                if is_dwc:
                    # for depthwise conv, the input channel equals to tile_k
                    tiled_shape["input"] = (N, tile_k, input_height, input_width)
                    if node.target == torch.ops.quantized_ops.conv2d_mx.default:
                        tiled_shape["input_scale"] = (
                            N, tile_k // bs, input_height, input_width
                        )
                        tiled_shape["weight_scale"] = (tile_k, 1, kH, kW)
                else:
                    tiled_shape["input"] = (N, c_end - c_start, input_height, input_width)
                    if node.target == torch.ops.quantized_ops.conv2d_mx.default:
                        tiled_shape["input_scale"] = (
                            N, (c_end - c_start) // bs, input_height, input_width
                        )
                        tiled_shape["weight_scale"] = (
                            tile_k, (c_end - c_start) // bs, kH, kW
                        )

                tiled_shape["weight"] = (tile_k, c_end - c_start, kH, kW)
                tiled_shape["bias"] = (tile_k,)
                tiled_shape["output"] = (N, tile_k, output_height, output_width)
                tiled_shape["keep_dims_and_padding"] = is_conv1
                tiled_shapes.append(tiled_shape)

    pad_value = 0
    if (input_code := node.kwargs.get("input_code")) is not None:
        code = model.get_buffer(input_code.target)
        pad_value = (code == 0).nonzero()[0].item()

    class Conv2dTiled(torch.nn.Module):
        def __init__(self, stride=1, padding=0, dilation=1, groups=1, block_size=1):
            super().__init__()
            self.stride = _pair(stride)
            self.padding = _pair(padding)
            self.dilation = _pair(dilation)
            self.groups = groups
            self.block_size = block_size

        def forward(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            input_scale: Optional[torch.Tensor] = None,
            weight_scale: Optional[torch.Tensor] = None,
            input_code: Optional[torch.Tensor] = None,
            weight_code: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # Iterate spatial tiles
            row_tiles = []
            tile_index = 0
            for y in range(0, Y, tile_y):
                col_tiles = []
                for x in range(0, X, tile_x):
                    acc = None
                    for c_start in range(0, C, tile_c):
                        config = tile_configs[tile_index]
                        tile_index += 1
                        # Extract tile parameters
                        c_start, c_end = config["c_tile"]
                        y_in_start_clamped, y_in_end_clamped = config["y_tile"]
                        x_in_start_clamped, x_in_end_clamped = config["x_tile"]
                        oh, ow = config["out_tile"]
                        pad_top, pad_left, pad_bottom, pad_right = config["input_padding"]
                        kernel_padding = config["kernel_padding"]
                        y_out_valid_start, x_out_valid_start = config["out_valid_start"]
                        keep_dims_and_padding = config["keep_dims_and_padding"]

                        if is_dwc:
                            input_tile = input[:, :, 
                                y_in_start_clamped:y_in_end_clamped,
                                x_in_start_clamped:x_in_end_clamped
                            ]
                        else:
                            input_tile = input[:,
                                c_start:c_end,
                                y_in_start_clamped:y_in_end_clamped,
                                x_in_start_clamped:x_in_end_clamped
                            ]

                        if pad_top or pad_left or pad_bottom or pad_right:
                            input_tile = F.pad(
                                input_tile,
                                (pad_left, pad_right, pad_top, pad_bottom),
                                mode='constant',
                                value=pad_value,
                            )

                        weight_tile = weight[:, c_start:c_end, :, :]

                        args = (
                            input_tile,
                            weight_tile,
                            bias if c_end == C else None,
                            self.stride,
                            kernel_padding,
                            self.dilation,
                            self.groups,
                        )

                        if input_scale is not None:
                            bs = self.block_size
                            if is_dwc:
                                tiled_input_scale = input_scale[
                                    :,
                                    :,
                                    y_in_start_clamped:y_in_end_clamped,
                                    x_in_start_clamped:x_in_end_clamped
                                ]
                            else:
                                tiled_input_scale = input_scale[:,
                                    c_start // bs : c_end // bs,
                                    y_in_start_clamped:y_in_end_clamped,
                                    x_in_start_clamped:x_in_end_clamped
                                ]
                            if pad_top or pad_left or pad_bottom or pad_right:
                                tiled_input_scale = F.pad(
                                    tiled_input_scale,
                                    (pad_left, pad_right, pad_top, pad_bottom),
                                    mode='constant',
                                    value=1.0,
                                )
                            if is_dwc:
                                tiled_weight_scale = weight_scale[:, 0:1, :, :]
                            else:
                                tiled_weight_scale = weight_scale[
                                    :, c_start // bs : c_end // bs, :, :
                                ]
                            kwargs = {
                                "input_scale": tiled_input_scale,
                                "weight_scale": tiled_weight_scale,
                                "block_size": bs,
                                "input_code": input_code,
                                "weight_code": weight_code,
                            }
                            out_patch = torch.ops.quantized_ops.conv2d_mx(*args, **kwargs)
                        else:
                            out_patch = torch.ops.aten.conv2d.default(*args)

                        if keep_dims_and_padding:
                            out_patch = out_patch[
                                :,
                                :,
                                y_out_valid_start:oh + y_out_valid_start,
                                x_out_valid_start:ow + x_out_valid_start,
                            ]

                        acc = out_patch if acc is None else acc + out_patch

                    col_tiles.append(acc)

                row_tiles.append(
                    torch.cat(col_tiles, dim=-1) if len(col_tiles) > 1
                    else col_tiles[0]
                )

            return (
                torch.cat(row_tiles, dim=2) if len(row_tiles) > 1
                else row_tiles[0]
            )

    def load_arg(a):
        return map_arg(a, lambda n: n.value if isinstance(n, Node) else n)

    if tile_y != Y or tile_x != X or tile_c != C:
        mod = Conv2dTiled(stride, padding, dilation, groups, bs)
        kwargs = {k: v for k, v in node.kwargs.items() if v is not None}
        kwargs.pop("block_size", None)
        gm = export_model(mod, load_arg(node.args[:3]), load_arg(kwargs))

        for n in list(gm.graph.nodes):
            if is_prunable_op(n):
                n.replace_all_uses_with(n.all_input_nodes[0])
                gm.graph.erase_node(n)
        gm.graph.lint()

        value_remap = {}
        output = replace_node_with_graph_module(model, gm, node, value_remap)

        # Update metadata on new nodes in the graph
        source_fn = node.meta['source_fn_stack'][-1]
        for n in list(value_remap.values()):
            if n.target in [
                torch.ops.aten.slice.Tensor, torch.ops.aten.pad.default,
            ]:
                n.meta["dtype"] = n.args[0].meta.get("dtype")

            if n.target == node.target:
                n.meta.update({
                    "tiled_shapes": tiled_shapes.pop(0),
                    "l2_tiling": tiling,
                    "dtype": node.meta.get("dtype"),
                    "source_fn_stack": [(n.name, source_fn[1])],
                })

        if output[0].target == torch.ops.aten.cat.default:
            move_fusable_ops_after_conv2d(model, output[0])
    else:
        node.meta["tiled_shapes"] = tiled_shapes[0]
        node.meta["l2_tiling"] = tiling


def _prime_factors(n: int):
    f, p = [], 2
    while p * p <= n:
        while n % p == 0:
            f.append(p)
            n //= p
        p += 1 if p == 2 else 2  # 2,3,5,7,...
    if n > 1:
        f.append(n)
    return f


def construct_tiled_shape(full_shape, tiled_dim: int, dims):
    """
    Reconstruct full-rank tiled shape.

    Args:
      full_shape: tuple/list[int] original shape (len N)
      tiled_dim: int, flattened size of the compressed (tiled) dims
      dims: iterable[int], indices of dims that were flattened into tiled_dim

    Returns:
      Tuple[int] of length N
    """
    full_shape = tuple(full_shape)
    N = len(full_shape)
    if N == 0:
        raise ValueError("full_shape must have at least one dimension.")

    # Normalize & validate compressed dims
    comp = sorted(set(int(i) for i in dims))
    if not comp:
        raise ValueError("dims cannot be empty.")
    if any(i < 0 or i >= N for i in comp):
        raise IndexError(f"dims must be in [0, {N-1}]. Got {dims}.")

    # Distribute prime factors of R across compressed dims (greedy balance)
    tiled = {i: 1 for i in comp}
    for p in _prime_factors(tiled_dim):
        for i in reversed(comp):
            if full_shape[i] % p == 0:
                tiled[i] *= p
                break

    # Build final shape
    out = [tiled[i] if i in comp else full_shape[i] for i in range(N)]
    return tuple(out)


def slice_tensor(node, dim, start, end, model):
    """
    Slice a tensor along a specific dimension using the given start and end indices.

    Args:
        node (Node): The node representing the tensor to be sliced.
        dim (int): The dimension along which to slice.
        start (int): The starting index for the slice.
        end (int): The ending index for the slice.
        graph (Graph): The computational graph to insert the slice operation.

    Returns:
        Node: A new node representing the sliced tensor.
    """
    graph = model.graph
    if node.op == "get_attr":
        param = get_parameter_or_buffer(model, node.target)
        sliced_data = param.data.narrow(dim, start, end - start)

        tiled_node = create_getattr_from_value(
            model, graph, node.target + "_tiled", sliced_data
        )
    else:
        tiled_node = graph.call_function(
            torch.ops.aten.slice.Tensor, (node, dim, start, end),
        )
    propagate_shape(tiled_node, model)
    tiled_node.meta["dtype"] = node.meta.get("dtype")
    return tiled_node


def split_gemm_node(model, node, X, C, K, x_tiled, c_tiled, k_tiled):
    """
    Transform a GEMM node (matmul/linear) into a tiled version along the reduction (C) dimension.
    Emits tiled sub-ops and replaces the original node in the FX graph.

    Args:
        model: FX GraphModule
        node: GEMM node to tile
        X, C, K: GEMM dimensions
        x_tiled, c_tiled, k_tiled: tiling sizes for output-X, reduction-C, and output-K
    """
    graph = model.graph

    input_node = node.args[0]
    weight_node = node.args[1]
    bias = node.args[2] if len(node.args) > 2 else None
    input_scale_node = node.kwargs.get("input_scale")
    weight_scale_node = node.kwargs.get("weight_scale")
    bs = node.kwargs.get("block_size", 1)

    weight_key = "other" if is_matmul(node) else "weight"
    weight_dim = -2 if is_matmul(node) else -1

    # Construct tiled shapes
    input_value = input_node.value
    tiled_input_shape = construct_tiled_shape(
        input_value.shape, x_tiled, list(range(input_value.ndim))[:-1]
    )

    weight_shape = (
        (c_tiled, k_tiled) if is_matmul(node) else (k_tiled, c_tiled)
    )
    weight_scale_shape = (
        (c_tiled // bs, k_tiled) if is_matmul(node)
        else (k_tiled, c_tiled // bs)
    )

    tiled_shapes = {
        "input": tiled_input_shape[:-1] + (c_tiled,),
        weight_key: weight_shape,
        "bias": (k_tiled,),
        "output": tiled_input_shape[:-1] + (k_tiled,),
        "input_scale": tiled_input_shape[:-1] + (c_tiled // bs,),
        "weight_scale": weight_scale_shape,
    }

    num_x_tiles = X // x_tiled
    num_k_tiles = K // k_tiled
    num_c_tiles = C // c_tiled

    if num_c_tiles == 1:
        node.meta["tiled_shapes"] = tiled_shapes
        node.meta["l2_tiling"] = (num_x_tiles, num_k_tiles)
        return

    if (source_fn_st := node.meta.get("source_fn_stack")) is not None:
        source_fn = source_fn_st[-1][1]
    else:
        source_fn = node.target

    psums = None
    for c in range(0, C, c_tiled):
        c_end = min(c + c_tiled, C)
        scale_c, scale_c_end = int(c / bs), int(c_end / bs)

        with graph.inserting_before(node):
            tiled_input = slice_tensor(input_node, -1, c, c_end, model)
            tiled_weight = slice_tensor(weight_node, weight_dim, c, c_end, model)
            kwargs = dict(node.kwargs)
            if input_scale_node is not None:
                kwargs["input_scale"] = slice_tensor(
                    input_scale_node, -1, scale_c, scale_c_end, model
                )
            if weight_scale_node is not None:
                kwargs["weight_scale"] = slice_tensor(
                    weight_scale_node, weight_dim, scale_c, scale_c_end, model
                )
            gemm_inputs = [tiled_input, tiled_weight]
            if not is_matmul(node) and c_end == C:
                gemm_inputs.append(bias)
            tiled_gemm = graph.call_function(
                node.target, tuple(gemm_inputs), kwargs,
            )

        propagate_shape(tiled_gemm)
        tiled_gemm.meta.update({
            "tiled_shapes": copy.deepcopy(tiled_shapes),
            "l2_tiling": (num_x_tiles, num_k_tiles),
            "dtype": node.meta.get("dtype"),
            "source_fn_stack": [(tiled_gemm.name, source_fn)],
        })

        if psums is not None:
            with graph.inserting_before(node):
                psums = graph.call_function(
                    torch.ops.aten.add.Tensor, (psums, tiled_gemm),
                )
            psums.meta["dtype"] = node.meta.get("dtype")
            psums.meta["source_fn_stack"] = [(psums.name, psums.target)]
            propagate_shape(psums)
        else:
            psums = tiled_gemm

    node.replace_all_uses_with(psums)
    graph.erase_node(node)


def get_valid_tiling(
    input_shape,
    fixed_dims=None,
    last_dim=None,
    reverse=False,
    min_sizes=None,
    order=None,
    round_robin=False,
):
    """
    Yields tile shapes by progressively reducing dimensions in a specified order.
    Once a dimension is reduced to its minimum size, it stays fixed. Certain dims
    can be explicitly fixed, or you can specify a single `last_dim` as a shortcut.

    Supports two modes:
        - Sequential mode (default): fully reduce each dimension in order before moving on.
        - Round-robin mode: cycle through dimensions, reducing one step at a time.

    Args:
        input_shape (tuple): The original shape.
        fixed_dims (list or tuple, optional): Indices of dims that should remain fixed.
        last_dim (int, optional): Convenience arg: fix a single dim (e.g., -1 for last).
        reverse (bool): If True, reverse the traversal order (ignored if order is given).
        min_sizes (tuple or list, optional): Minimum size / multiple for each dimension
                                             (default is 1).
        order (list or tuple, optional): Explicit order of dimension indices to reduce.
                                         Example: (2,0,1) means dim2 → dim0 → dim1.
        round_robin (bool): If True, reduce dims in a cyclic round-robin fashion.
    """
    def get_factors(n, min_size):
        return [i for i in range(n, min_size - 1, -1) if n % i == 0]

    def get_tiling(full_shape, tiled_shape):
        return tuple(f // t for f, t in zip(full_shape, tiled_shape))

    dims = len(input_shape)

    # Normalize fixed dims
    fixed = set()
    if fixed_dims is not None:
        fixed.update(dims + d if d < 0 else d for d in fixed_dims)
    if last_dim is not None:
        ld = dims + last_dim if last_dim < 0 else last_dim
        fixed.update(d for d in range(ld, dims))

    # Order of dimensions to traverse
    if order is not None:
        dim_order = list(order)
    else:
        dim_order = list(range(dims))
        if reverse:
            dim_order = dim_order[::-1]

    # Apply default min sizes
    if min_sizes is None:
        min_sizes = [1] * dims
    else:
        min_sizes = [1] * (dims - len(min_sizes)) + list(min_sizes)

    current = list(input_shape)
    yield tuple(current), get_tiling(input_shape, tuple(current))

    if not round_robin:
        # --- Sequential mode ---
        for dim in dim_order:
            if dim in fixed:
                continue
            factors = get_factors(input_shape[dim], min_sizes[dim])
            for f in factors[1:]:  # skip full-size factor
                if f % min_sizes[dim] != 0:
                    continue  # enforce unroll multiple
                current[dim] = f
                yield tuple(current), get_tiling(input_shape, tuple(current))
            current[dim] = max(min_sizes[dim], 1)
    else:
        # --- Round robin mode ---
        factor_lists = {
            d: get_factors(input_shape[d], min_sizes[d])
            for d in dim_order if d not in fixed
        }
        indices = {d: 0 for d in factor_lists}

        active = list(factor_lists.keys())
        while active:
            next_active = []
            for d in active:
                idx = indices[d] + 1
                if idx < len(factor_lists[d]):
                    current[d] = factor_lists[d][idx]
                    yield tuple(current), get_tiling(input_shape, current)
                    indices[d] = idx
                    if idx + 1 < len(factor_lists[d]):
                        next_active.append(d)
            active = next_active


def node_mem(n, tiles, bank_size=None):
    size = get_node_bytes(n) * n.value.numel() / tiles
    if bank_size is not None:
        size = int(math.ceil(size / bank_size) * bank_size)
    return size


def calculate_gemm_tile_size(
    node, x_factor, c_factor, k_factor, bank_size=None
):
    total_bytes = 0
    input_node, weight_node = node.args[0], node.args[1]

    input_tiles = x_factor * c_factor
    weight_tiles = c_factor * k_factor
    output_tiles = x_factor * k_factor

    # Input, weight, and output memory
    total_bytes += node_mem(input_node, input_tiles, bank_size)
    total_bytes += node_mem(weight_node, weight_tiles, bank_size)
    total_bytes += node_mem(node, output_tiles, bank_size)

    # Bias if present
    if not is_matmul(node) and len(node.args) > 2:
        total_bytes += node_mem(node.args[2], k_factor, bank_size)

    # Optional scale factors
    input_scale_node = node.kwargs.get("input_scale")
    if input_scale_node is not None:
        total_bytes += node_mem(input_scale_node, input_tiles, bank_size)

    weight_scale_node = node.kwargs.get("weight_scale")
    if weight_scale_node is not None:
        total_bytes += node_mem(weight_scale_node, weight_tiles, bank_size)

    return total_bytes


def select_gemm_tiling(node, X, C, K, cache_size, unroll_dims, bank_size=None):
    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    # Stage 1: pick a reduction dim that fits in a bank
    c_outer = 1
    if bank_size is not None:
        for (ct,), _ in get_valid_tiling((C,), min_sizes=(unroll_dims[0],)):
            if min(128, X) * ct * get_node_bytes(node.args[0]) <= bank_size:
                c_outer = C // ct
                break

    # Stage 2: search tilings for (X, C, K) in given order
    for (xt, ct, kt), (x_factor, c_factor, k_factor) in get_valid_tiling(
        (X, C // c_outer, K),
        min_sizes=(1, unroll_dims[0], unroll_dims[1]),
        order=(2, 0, 1),
    ):
        total_size = calculate_gemm_tile_size(
            node, x_factor, c_outer * c_factor, k_factor, bank_size=bank_size
        )
        if total_size <= cache_size:
            return xt, ct, kt

    # Stage 3: search tilings without bank constraint
    for (xt, ct, kt), (x_factor, c_factor, k_factor) in get_valid_tiling(
        (X, C // c_outer, K),
        min_sizes=(1, unroll_dims[0], unroll_dims[1]),
        order=(2, 0, 1),
    ):
        total_size = calculate_gemm_tile_size(
            node, x_factor, c_outer * c_factor, k_factor, bank_size=None
        )
        if total_size <= cache_size:
            return xt, ct, kt

    # If no valid tiling found
    raise ValueError(
        f"Cannot tile X={X}, C={C}, K={K} to fit cache size {cache_size}."
    )


def calculate_conv2d_tile_size(
    node, y_factor, x_factor, c_factor, k_factor, bank_size=None
):
    """
    Calculate memory footprint of a conv2d under tiling (batch=1).

    Args:
        node: conv2d node
        c_factor: tiling factor for input channels
        k_factor: tiling factor for output channels
        y_factor: tiling factor for output height
        x_factor: tiling factor for output width
        bank_size: optional, round each memory block up to multiple of bank_size
    """
    stride = get_arg_or_kwarg(node, 3, "stride", (1, 1))
    padding = get_arg_or_kwarg(node, 4, "padding", (0, 0))
    dilation = get_arg_or_kwarg(node, 5, "dilation", (1, 1))
    bs = node.kwargs.get("batch_size", 1)

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    input_node, weight_node = node.args[0], node.args[1]
    bias_node = node.args[2] if len(node.args) > 2 else None

    # Input/output dimensions
    N, K, Y, X = node.shape
    _, C, kH, kW = weight_node.shape

    total_bytes = 0

    # Input memory: depends on receptive field for output tiles
    input_y = (Y // y_factor - 1) * stride[0] + (kH - 1) * dilation[0] + 1
    input_x = (X // x_factor - 1) * stride[1] + (kW - 1) * dilation[1] + 1
    input_size = 1 * (C // c_factor) * input_y * input_x
    total_bytes += get_node_bytes(input_node) * input_size

    # Weight memory
    weight_tiles = c_factor * k_factor
    total_bytes += node_mem(weight_node, weight_tiles, bank_size)

    # Output memory
    output_tiles = k_factor * y_factor * x_factor
    total_bytes += node_mem(node, output_tiles, bank_size)

    # Bias
    if bias_node is not None:
        total_bytes += node_mem(bias_node, k_factor)

    # Optional scale factors
    input_scale_node = node.kwargs.get("input_scale")
    if input_scale_node is not None:
        total_bytes += (
            get_node_bytes(input_scale_node) * input_size / bs
        )

    weight_scale_node = node.kwargs.get("weight_scale")
    if weight_scale_node is not None:
        total_bytes += node_mem(weight_scale_node, weight_tiles, bank_size)

    return total_bytes


def select_conv2d_tiling(node, Y, X, C, K, cache_size, unroll_dims, bank_size=None):
    """
    Pick tiling for conv2d layers to fit in cache.

    Args:
        node: conv2d node
        Y, X: output height, width
        C, K: input/output channels
        cache_size: max allowed memory
        unroll_dims: (c_unroll, k_unroll)
        bank_size: optional bank constraint
    """
    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    # Heuristic: channel-dominant or spatial-dominant
    if C * K > 4 * (Y * X):   # channels dominate
        order = (2, 3, 0, 1)  # C, K, Y, X
    else:
        order = (3, 0, 1, 2)  # K, Y, X, C

    # Stage 1: bank-size constraint on reduction dim
    c_outer = 1
    if bank_size is not None:
        for (ct,), _ in get_valid_tiling((C,), min_sizes=(unroll_dims[0],)):
            # TODO determine the minimum X and Y tile sizes
            if min(14, X) * min(14, Y) * ct * get_node_bytes(node.args[0]) <= bank_size:
                c_outer = C // ct
                break

    # Stage 2: exhaustive K first
    k_outer = 1
    for (yt, xt, ct, kt), (y_factor, x_factor, c_factor, k_factor) in get_valid_tiling(
        (Y, X, C // c_outer, K),
        min_sizes=(1, 1, unroll_dims[0], unroll_dims[1]),
        order=(3,),
    ):
        total_size = calculate_conv2d_tile_size(
            node, 1, c_outer, k_factor, 1, bank_size
        )
        if total_size <= cache_size:
            return yt, xt, ct, kt
        k_outer = K // kt

    # Stage 3: greedy search with bank constraint
    for (yt, xt, ct, kt), (y_factor, x_factor, c_factor, k_factor) in get_valid_tiling(
        (Y, X, C // c_outer, K // k_outer),
        min_sizes=(1, 1, unroll_dims[0], unroll_dims[1]),
        order=order,
        round_robin=True,
    ):
        total_size = calculate_conv2d_tile_size(
            node, x_factor, c_outer * c_factor, k_outer * k_factor, y_factor, bank_size
        )
        if total_size <= cache_size:
            return yt, xt, ct, kt

    # Stage 4: fallback without bank constraint
    for (yt, xt, ct, kt), (y_factor, x_factor, c_factor, k_factor) in get_valid_tiling(
        (Y, X, C // c_outer, K // k_outer),
        min_sizes=(1, 1, unroll_dims[0], unroll_dims[1]),
        order=order,
        round_robin=True,
    ):
        total_size = calculate_conv2d_tile_size(
            node, x_factor, c_outer * c_factor, k_outer * k_factor, y_factor
        )
        if total_size <= cache_size:
            return yt, xt, ct, kt

    # If nothing found
    raise ValueError(
        f"Cannot tile Conv2D Y={Y}, X={X}, C={C}, K={K} to fit cache {cache_size}."
    )


def run_matrix_op_l2_tiling(
    model, unroll, cache_size=DEFAULT_CACHE_SIZE, num_banks=None
):
    """
    Perform tiling on GEMM operations in a model to fit intermediate data into cache.

    Tiling is applied across the output (K), input (X), and channel (C) dimensions with
    the following strategy:
    - Maximize tile size along X (batch * spatial)
    - Minimize splits along C to avoid overhead from summing partial results
    - Ensure K and C tile sizes are multiples of specified minimums
    - Cache is divided across multiple banks

    Args:
        model: A model object with a FX Graph containing GEMM nodes.
        cache_size (int): Total cache size in bytes.
        unroll (int): Systolic array input and output channel unrolling dimension. 
    """
    graph = model.graph

    for node in list(graph.nodes):
        if not is_gemm_op(node):
            continue

        input_node = node.args[0]
        weight_node = node.args[1]
        bank_size = None if num_banks is None else cache_size // num_banks

        if is_conv2d(node):
            _, K, Y, X = node.shape
            C = weight_node.shape[1]

            total_size = calculate_conv2d_tile_size(
                node, 1, 1, 1, 1, bank_size=bank_size
            )

            if total_size <= cache_size:
                logger.info(
                    f"{node} ({Y}, {X}, {C}, {K}), total_size={total_size} fits "
                    "in cache."
                )
                continue

            logger.info(
                f"{node} ({Y}, {X}, {C}, {K}), total_size={total_size} does not "
                "fit in cache."
            )

            y_tiled, x_tiled, c_tiled, k_tiled = select_conv2d_tiling(
                node, Y, X, C, K, cache_size, unroll, bank_size
            )

            weight_shape = tuple(weight_node.shape)
            output_shape = tuple(node.shape)

            weight_tiled = (k_tiled, c_tiled, weight_shape[2], weight_shape[3])
            output_tiled = (1, k_tiled, y_tiled, x_tiled)

            logger.info(
                f"{node}: weight {weight_shape} -> {weight_tiled}, "
                f"output {output_shape} -> {output_tiled}"
            )

            split_conv2d_node(model, node, (y_tiled, x_tiled, c_tiled, k_tiled))
        else:
            if is_linear(node):
                K, C = weight_node.shape
            elif is_matmul(node):
                C, K = weight_node.shape[-2:]
            X = int(input_node.value.numel() / C)

            total_size = calculate_gemm_tile_size(node, 1, 1, 1, bank_size=bank_size)

            if total_size <= cache_size:
                logger.info(
                    f"{node} ({X} x {C} x {K}), total_size={total_size} fits in cache."
                )
                continue

            logger.info(
                f"{node} ({X} x {C} x {K}), total_size={total_size} does not fit "
                "in cache."
            )

            x_tiled, c_tiled, k_tiled = select_gemm_tiling(
                node, X, C, K, cache_size, unroll, bank_size
            )

            logger.info(f"{node} ({X} x {C} x {K}) -> ({x_tiled} x {c_tiled} x {k_tiled})")

            split_gemm_node(model, node, X, C, K, x_tiled, c_tiled, k_tiled)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def get_tiled_shape(shape, tiling):
    if not shape or tiling is None:
        return shape
    ndim = len(shape)
    m = len(tiling)
    if ndim > m:
        tiling = (1,) * (ndim - m) + tiling
    elif ndim < m:
        shape = (1,) * (m - ndim) + shape
    tiled_shape = []
    for i in range(len(shape)):
        tiled_shape.append(shape[i] // tiling[i] if shape[i] > 1 else shape[i])
    return tuple(tiled_shape[-ndim:])


def get_node_to_key(node):
    from torch.fx.operator_schemas import normalize_function

    args_and_kwargs = normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )
    node_to_key = {
        n.meta.get('source_node', n): k
        for k, n in args_and_kwargs.kwargs.items() if isinstance(n, Node)
    }
    return node_to_key


def run_vector_op_node_l2_tiling(
    node,
    unroll,
    cache_size=DEFAULT_CACHE_SIZE,
    num_banks=None,
):
    if not is_elementwise_op(node) and node.target not in [
        torch.ops.aten.softmax.int,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.quantized_ops.layer_norm.default,
        torch.ops.quantized_ops.calculate_mx_qparam.default,
        torch.ops.quantized_ops.quantize_mx.default,
    ]:
        return

    # Certain dimensions cannot be tiled, e.g., transpose or reduction dims
    last_dim = -1
    if node.target == torch.ops.quantized_ops.calculate_mx_qparam.default:
        last_dim = min(node.args[1])
    elif node.target == torch.ops.quantized_ops.quantize_mx.default:
        last_dim = min(node.args[2])
    elif node.target == torch.ops.aten.transpose.int:
        last_dim = min(*node.args[1:])
    elif node.target == torch.ops.aten.permute.default:
        last_dim = next((i for i, d in enumerate(node.args[1]) if i != d), None)

    node_to_key = get_node_to_key(node)
    input_nodes = [
        n for n in node.all_input_nodes
        if (
            "qmap" not in n.name
            and "code" not in n.name
            and isinstance(n.value, torch.Tensor)
            and (n.value.numel() > 1 or n.op != "get_attr")
        )
    ]
    output_shape = (
        node.value.shape if isinstance(node.value, torch.Tensor)
        else node.value[1].shape
    )

    def get_node_size(n, shape, node, bank_size):
        size = get_node_bytes(n) * math.prod(shape)
        if bank_size is not None:
            size = int(math.ceil(size / bank_size) * bank_size)
        # Double input size for softmax/layernorm (scratch space)
        if n == node.args[0] and node.target in [
            torch.ops.aten.softmax.int,
            torch.ops.aten.layer_norm.default,
            torch.ops.quantized_ops.layer_norm.default,
        ]:
            size *= 2
        return size

    def try_tiling(align_bank):
        bank_size = (
            cache_size // num_banks if align_bank and num_banks is not None
            else None
        )
        for tiled_output_shape, tiling in get_valid_tiling(
            output_shape, last_dim=last_dim, min_sizes=(unroll,)
        ):
            tiled_shapes = {
                node_to_key.get(n): get_tiled_shape(tuple(n.shape), tiling)
                for n in input_nodes
            }
            total_size = sum(
                get_node_size(n, tiled_shapes[node_to_key[n]], node, bank_size)
                for n in input_nodes
            )

            if isinstance(node.value, (tuple, list)):
                tiled_shapes["output"] = [
                    get_tiled_shape(t.shape, tiling) for t in node.value
                ]
                total_size += sum(
                    b * math.prod(s)
                    for b, s in zip(get_node_bytes(node), tiled_shapes["output"])
                )
            else:
                tiled_shapes["output"] = tiled_output_shape
                total_size += get_node_size(
                    node, tiled_output_shape, node, bank_size
                )

            logger.debug(
                f"Trying tiling {tiling} for {node} "
                f"(bank_size={bank_size}), total_size={total_size}"
            )

            if total_size <= cache_size:
                if math.prod(tiling) > 1:
                    logger.info(f"Selected tiling {tiling} for {node}")
                    node.meta["tiled_shapes"] = tiled_shapes
                    node.meta["l2_tiling"] = tiling
                return True
        return False

    if try_tiling(True) or try_tiling(False):
        return True

    logger.warning(f"No L2 tiling found for {node.name}")
    return False


def run_vector_op_l2_tiling(
    model, unroll, cache_size=DEFAULT_CACHE_SIZE, num_banks=None
):
    """
    Perform tiling on vector operations in a model to fit intermediate data into cache.

    Tiling is applied across all non-reduction dimensions with the following strategy:
    - Maximize tile size along the last dimension
    - Ensure tile sizes are multiples of specified minimums
    - Cache is divided across multiple banks

    Args:
        model: A model object with a FX Graph containing vector operation nodes.
        cache_size (int): Total cache size in bytes.
        unroll (int): Minimum unrolling dimension for vector operations.
    """
    graph = model.graph

    for node in list(graph.nodes):
        run_vector_op_node_l2_tiling(
            node, unroll, cache_size, num_banks
        )

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model
