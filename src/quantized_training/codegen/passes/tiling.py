import copy
import logging
import math
import re
from typing import List, Tuple, Generator, Optional, Union

import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.node import map_arg
from torch.fx.operator_schemas import normalize_function

from .utils import get_arg_or_kwarg, _pair
from ..mapping import (
    get_node_bytes,
    get_parameter_or_buffer,
    propagate_shape,
    replace_node_with_graph_module,
    _nodes_sequential,
)
from ..mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_linear,
    is_matmul,
    is_prunable_op,
)
from ..banking import (
    get_banking_strategies_for_op,
    require_allocation,
    _get_scope,
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


def _get_tiling_context(node_to_fuse: torch.fx.Node) -> List[Tuple[int, int, int]]:
    """
    Backtracks from node_to_fuse to find slice/pad ops and calculates
    the equivalent output slice parameters.
    """
    conv_node = node_to_fuse
    if not is_conv2d(conv_node) and len(conv_node.args) > 1:
         # Fallback for patterns like relu(conv) or add(conv, bias)
         # Assuming conv is the second arg based on original logic, but safer to check
         potential_conv = conv_node.args[1]
         if isinstance(potential_conv, torch.fx.Node) and is_conv2d(potential_conv):
             conv_node = potential_conv

    if "tiled_shapes" not in conv_node.meta:
        return []

    input_shape = conv_node.meta["tiled_shapes"]["input"]
    output_shape = conv_node.meta["tiled_shapes"]["output"]

    slice_args = []

    # Traverse backwards to find slicing details
    current_node = conv_node.args[0]
    while isinstance(current_node, torch.fx.Node) and current_node.target in [
        torch.ops.aten.slice.Tensor, torch.ops.aten.pad.default
    ]:
        if current_node.target == torch.ops.aten.slice.Tensor:
            dim, start, end = current_node.args[1:]
            # Ensure we only calculate slices for spatial dims (H=2, W=3)
            if dim in (2, 3):
                # Calculate tile index relative to the *tile shape* or *full shape*
                tile_idx = round(float(start) / input_shape[dim])
                tile_start = tile_idx * output_shape[dim]
                tile_end = tile_start + output_shape[dim]
                slice_args.insert(0, (dim, tile_start, tile_end))

        current_node = current_node.args[0]

    return slice_args


def _clone_parameter(
    model: torch.fx.GraphModule, arg: torch.fx.Node, anchor: torch.fx.Node
) -> torch.fx.Node:
    """Creates a new get_attr node for a parameter."""
    param = fetch_attr(model, arg.target)
    prefix = arg.name

    if match := re.fullmatch(r'(code|qmap)(_\d+)?', arg.name):
        prefix = match.group(1)

    with model.graph.inserting_before(anchor):
        new_attr = create_getattr_from_value(model, model.graph, prefix, param)

    propagate_shape(new_attr, model)
    new_attr.meta["dtype"] = arg.meta.get("dtype")
    return new_attr


def _slice_side_input(
    model: torch.fx.GraphModule,
    arg: torch.fx.Node,
    slice_args: List[Tuple[int, int, int]],
    anchor: torch.fx.Node,
) -> torch.fx.Node:
    """Slices a side input (e.g. residual connection) to match the current tile."""
    current_slice_node = arg

    # Optimization: If the side input is not spatial (e.g. 1D bias), do not slice.
    if len(arg.shape) < 4:
        return arg

    for dim, start, end in slice_args:
        with model.graph.inserting_before(anchor):
            current_slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor,
                (current_slice_node, dim, start, end),
            )
        propagate_shape(current_slice_node, model)
        current_slice_node.meta["dtype"] = arg.meta.get("dtype")

    return current_slice_node


def create_new_chain(model, node_to_fuse, cat_node, fusable):
    """
    Replicates the 'fusable' chain of nodes (originally after 'cat_node')
    to be placed after 'node_to_fuse', applying appropriate spatial tiling.
    """
    slice_params = _get_tiling_context(node_to_fuse)

    anchor = node_to_fuse.next
    value_remap = {cat_node: node_to_fuse}

    for n in fusable:
        for arg in n.all_input_nodes:
            if arg in value_remap:
                continue

            if arg.op == "get_attr":
                value_remap[arg] = _clone_parameter(model, arg, anchor)
            else:
                value_remap[arg] = _slice_side_input(
                    model, arg, slice_params, node_to_fuse
                )

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

    # Reconnect users of the original node to the end of the new chain
    last_node = value_remap[fusable[-1]]
    first_node = value_remap[fusable[0]]

    for user in list(node_to_fuse.users):
        if user != first_node:
            user.replace_input_with(node_to_fuse, last_node)


def move_fusable_ops_after_conv2d(model, node):
    order = {n: i for i, n in enumerate(model.graph.nodes)}
    fusable_ops = []
    next_node = next(iter(node.users))
    while is_elementwise_op(next_node):
        chain = [node] + fusable_ops + [next_node]
        if _nodes_sequential(chain, order):
            fusable_ops.append(next_node)
        else:
            break
        # Stop fusing if last node is a quantize op
        if (
            len(next_node.users) != 1
            or next_node.target == torch.ops.quantized_ops.quantize.default
        ):
            break
        next_node = next(iter(next_node.users))

    if not fusable_ops:
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


def _make_conv2d_tiled_module(
    X, C, tile_x, tile_c, pad_value, is_dwc, is_mx_conv, configs
):
    """
    Factory function to create a Conv2dTiled module class with captured parameters.
    """
    class Conv2dTiled(torch.nn.Module):
        def __init__(self, stride=1, padding=0, dilation=1, groups=1, block_size=16):
            super().__init__()
            self.stride = _pair(stride)
            self.padding = _pair(padding)
            self.dilation = _pair(dilation)
            self.groups = groups
            self.block_size = block_size

        def get_input_tile(self, input, cfg):
            c0, c1 = cfg["c_tile"]
            y0, y1 = cfg["y_tile"]
            x0, x1 = cfg["x_tile"]

            if is_dwc:
                tile = input[:, :, y0:y1, x0:x1]
            else:
                tile = input[:, c0:c1, y0:y1, x0:x1]

            padding = cfg["input_padding"]
            if any(p > 0 for p in padding):
                tile = F.pad(tile, padding, "constant", pad_value)

            return tile

        def get_weight_tile(self, weight, cfg):
            c0, c1 = cfg["c_tile"]
            return weight[:, c0:c1, :, :]

        def get_scales(self, input_scale, weight_scale, cfg):
            if not is_mx_conv:
                return None, None

            c0, c1 = cfg["c_tile"]
            y0, y1 = cfg["y_tile"]
            x0, x1 = cfg["x_tile"]

            bs = self.block_size

            if is_dwc:
                in_s = input_scale[:, :, y0:y1, x0:x1]
                wt_s = weight_scale[:, 0:1, :, :]
            else:
                s0, s1 = c0 // bs, c1 // bs
                in_s = input_scale[:, s0:s1, y0:y1, x0:x1]
                wt_s = weight_scale[:, s0:s1, :, :]

            padding = cfg["input_padding"]
            if any(p > 0 for p in padding):
                tile = F.pad(tile, padding, "constant", 1.0)

            return in_s, wt_s

        def run_conv(self, input_tile, weight_tile, bias, cfg, scales, codes):
            args = (
                input_tile,
                weight_tile,
                bias,
                self.stride,
                cfg["conv_padding"],
                self.dilation,
                self.groups,
            )

            if not is_mx_conv:
                return torch.ops.aten.conv2d.default(*args)

            return torch.ops.quantized_ops.conv2d_mx(
                *args,
                input_scale=scales[0],
                weight_scale=scales[1],
                block_size=self.block_size,
                input_code=codes[0],
                weight_code=codes[1],
            )

        def trim_output(self, out, cfg):
            if not cfg["keep_dims_and_padding"]:
                return out

            oy, ox = cfg["output_offset"]
            oh, ow = cfg["output_sizes"]
            return out[:, :, oy:oy+oh, ox:ox+ow]

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
            row_tiles = []
            col_tiles = []

            # Track when to break rows
            tile_c_count = C // tile_c
            tile_x_count = X // tile_x

            for tile_idx, cfg in enumerate(configs):
                input_tile = self.get_input_tile(input, cfg)
                weight_tile = self.get_weight_tile(weight, cfg)
                scale_tiles = self.get_scales(input_scale, weight_scale, cfg)
                out = self.run_conv(
                    input_tile,
                    weight_tile,
                    bias if cfg["c_tile"][1] == C else None,
                    cfg,
                    scale_tiles,
                    (input_code, weight_code),
                )
                out = self.trim_output(out, cfg)

                # accumulate partial channels
                acc = out if (tile_idx % tile_c_count == 0) else (acc + out)

                # when finishing tile_c accumulations
                if (tile_idx + 1) % tile_c_count == 0:
                    col_tiles.append(acc)

                # when finishing tile_x tiles (a full row)
                if len(col_tiles) == tile_x_count:
                    row_tiles.append(torch.cat(col_tiles, dim=-1))
                    col_tiles = []

            return torch.cat(row_tiles, dim=2)

    return Conv2dTiled


def _decompose_conv2d_node(model, node, tile_sizes, tiled_shapes, configs):
    stride = get_arg_or_kwarg(node, 3, "stride", 1)
    padding = get_arg_or_kwarg(node, 4, "padding", 0)
    dilation = get_arg_or_kwarg(node, 5, "dilation", 1)
    groups = get_arg_or_kwarg(node, 6, "groups", 1)
    bs = node.kwargs.get("block_size", 1)

    N, K, Y, X = node.shape
    _, C, kH, kW = node.args[1].shape

    tile_y, tile_x, tile_c, tile_k = tile_sizes

    # Compute pad_value once
    pad_value = 0
    input_code = node.kwargs.get("input_code")
    if input_code is not None:
        code = model.get_buffer(input_code.target)
        pad_value = (code == 0).nonzero()[0].item()

    is_dwc = is_depthwise_conv(node)
    is_mx_conv = node.target == torch.ops.quantized_ops.conv2d_mx.default

    # Create the tiled module using factory function
    Conv2dTiled = _make_conv2d_tiled_module(
        X, C, tile_x, tile_c, pad_value, is_dwc, is_mx_conv, configs
    )

    def load_arg(a):
        return map_arg(a, lambda n: n.value if isinstance(n, Node) else n)

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

    if (source_fn_st := node.meta.get("source_fn_stack")) is not None:
        source_fn = source_fn_st[-1][1]
    else:
        source_fn = node.target

    # Update metadata on new nodes in the graph
    for n in list(value_remap.values()):
        if n.target == torch.ops.aten.slice.Tensor and n.args[0].op == "get_attr":
            c_start, c_end = n.args[2], n.args[3]
            with model.graph.inserting_before(n):
                sliced_param = _slice_tensor(n.args[0], 1, c_start, c_end, model)
            n.replace_all_uses_with(sliced_param)
            model.graph.erase_node(n)
            continue

        if n.target in [
            torch.ops.aten.slice.Tensor, torch.ops.aten.pad.default,
        ]:
            n.meta["dtype"] = n.args[0].meta.get("dtype")

        if n.target == node.target:
            n.meta.update({
                "tiled_shapes": tiled_shapes.pop(0),
                "l2_tiling": (1, K // tile_k, 1, 1),
                "dtype": node.meta.get("dtype"),
                "source_fn_stack": [(n.name, source_fn)],
            })

    if output[0].target == torch.ops.aten.cat.default:
        move_fusable_ops_after_conv2d(model, output[0])


def _pad_input(model, node, arg, padding, pad_value):
    with model.graph.inserting_before(node):
        new_arg = model.graph.call_function(
            torch.ops.aten.pad.default,
            (arg, padding, "constant", pad_value),
        )
    propagate_shape(new_arg, model)
    new_arg.meta["dtype"] = arg.meta.get("dtype")
    node.replace_input_with(arg, new_arg)
    return new_arg


def split_conv2d_node(model, node, tile_sizes):
    """
    Replace a conv2d node with a tiled conv2d subgraph.

    Args:
        model: GraphModule
        node: node (must be aten.conv2d or quantized conv2d)
        tile_sizes: (Y, X, C, K)
    """
    stride  = _pair(get_arg_or_kwarg(node, 3, "stride", 1))
    padding = _pair(get_arg_or_kwarg(node, 4, "padding", 0))
    dilation = _pair(get_arg_or_kwarg(node, 5, "dilation", 1))
    bs = node.kwargs.get("block_size", 1)

    is_conv1 = node.args[0].shape[1] == 3
    is_dwc = is_depthwise_conv(node)

    N, K, Y, X = node.shape
    _, C, kH, kW = node.args[1].shape
    _, _, IX, IY = node.args[0].shape
    tile_y, tile_x, tile_c, tile_k = tile_sizes

    pad_value = 0
    if (input_code := node.kwargs.get("input_code")) is not None:
        code = model.get_buffer(input_code.target)
        pad_value = (code == 0).nonzero()[0].item()

    if (tile_x != X or tile_y != Y) and any(p for p in padding) and not is_conv1:
        pad_hw = (padding[1], padding[1], padding[0], padding[0])
        _pad_input(model, node, node.args[0], pad_hw, pad_value)

        if input_scale := node.kwargs.get("input_scale"):
            _pad_input(model, node, input_scale, pad_hw, 1.0)

        padding = _pair(0)
        node.args = node.args[:4] + (padding,) + node.args[5:]
        propagate_shape(node, model)
        _, _, IY, IX = node.args[0].shape

    def _compute_spatial_tile_region(y, x, oh, ow):
        if tile_x == X and tile_y == Y:
            return {
                "y_tile": (0, IY),
                "x_tile": (0, IX),
                "input_padding": (0, 0, 0, 0),
                "output_offset": (0, 0),
            }

        y_in_start = y * stride[0] - padding[0]
        y_in_end = y_in_start + (oh - 1) * stride[0] + (kH - 1) * dilation[0] + 1
        x_in_start = x * stride[1] - padding[1]
        x_in_end = x_in_start + (ow - 1) * stride[1] + (kW - 1) * dilation[1] + 1

        # Grab more input pixels so that after applying padding of three, the
        # new convolution still aligns with the original one, just with some
        # garbage outputs on the right and bottom side
        if is_conv1:
            y_in_start = y_in_start - (padding[0] % stride[0])
            x_in_start = x_in_start - (padding[1] % stride[1])

        # Adjust receptive field to multiple of stride for depthwise conv
        if is_dwc:
            if rem := (y_in_end - y_in_start) % stride[0]:
                y_in_end += stride[0] - rem
            if rem := (x_in_end - x_in_start) % stride[1]:
                x_in_end += stride[1] - rem

        y_in_start_clamped = max(y_in_start, 0)
        x_in_start_clamped = max(x_in_start, 0)

        if not is_conv1:
            x_in_end_clamped = min(x_in_end, IX)
            y_in_end_clamped = min(y_in_end, IY)

            assert (
                y_in_start_clamped == y_in_start
                or x_in_start_clamped == x_in_start
                or y_in_end_clamped == y_in_end
                or x_in_end_clamped == x_in_end
            ), f"{node}: Unexpected input tile size"

            # downsample layers input size must be multiples of stride
            if kH == 1 and kW == 1:
                y_in_end_clamped = y_in_start_clamped + oh * stride[0]
                x_in_end_clamped = x_in_start_clamped + ow * stride[1]

            pad_top, pad_left, pad_bottom, pad_right = 0, 0, 0, 0
            x_offset, y_offset = 0, 0
        else:
            # conv1 hardware replication constraints
            y_in_end += (16 - ((y_in_end - y_in_start_clamped) % 16))
            x_in_end += (16 - ((x_in_end - x_in_start_clamped) % 16))
            y_in_end_clamped = min(y_in_end, IY)
            x_in_end_clamped = min(x_in_end, IX)

            pad_top = 0
            pad_left = 0
            pad_bottom = y_in_end - y_in_end_clamped
            pad_right = x_in_end - x_in_end_clamped

            y_offset = (y * stride[0] - y_in_start_clamped) // stride[0]
            x_offset = (x * stride[1] - x_in_start_clamped) // stride[1]

        return {
            "y_tile": (y_in_start_clamped, y_in_end_clamped),
            "x_tile": (x_in_start_clamped, x_in_end_clamped),
            "input_padding": (pad_left, pad_right, pad_top, pad_bottom),
            "output_offset": (y_offset, x_offset),
        }

    def _compute_per_tile_shapes(cfg):
        (y0, y1) = cfg["y_tile"]
        (x0, x1) = cfg["x_tile"]
        (pad_left, pad_right, pad_top, pad_bottom) = cfg["input_padding"]
        conv_padding = cfg["conv_padding"]
        (oh, ow) = cfg["output_sizes"]
        (c_start, c_end) = cfg["c_tile"]

        h_in = (y1 - y0) + pad_top + pad_bottom
        w_in = (x1 - x0) + pad_left + pad_right
        h_out = (h_in + 2 * conv_padding[0] - kH) // stride[0] + 1
        w_out = (w_in + 2 * conv_padding[1] - kW) // stride[1] + 1

        assert h_out >= oh and w_out >= ow, (
            f"Output sizes mismatch: ({h_out}, {w_out}) vs ({oh}, {ow})"
        )

        c_tile = c_end - c_start
        tiled_shape = {}

        if is_dwc:
            tiled_shape["input"] = (N, tile_k, h_in, w_in)
            tiled_shape["input_scale"] = (N, tile_k // bs, h_in, w_in)
            tiled_shape["weight_scale"] = (tile_k, 1, kH, kW)
        else:
            tiled_shape["input"] = (N, c_tile, h_in, w_in)
            tiled_shape["input_scale"] = (N, c_tile // bs, h_in, w_in)
            tiled_shape["weight_scale"] = (tile_k, c_tile // bs, kH, kW)

        tiled_shape["weight"] = (tile_k, c_tile, kH, kW)
        tiled_shape["bias"] = (tile_k,)
        tiled_shape["output"] = (N, tile_k, h_out, w_out)
        return tiled_shape

    tiled_shapes = []
    tile_configs = []
    for y in range(0, Y, tile_y):
        for x in range(0, X, tile_x):
            oh = min(tile_y, Y - y)
            ow = min(tile_x, X - x)

            spatial = _compute_spatial_tile_region(y, x, oh, ow)

            for c_start in range(0, C, tile_c):
                c_end = min(c_start + tile_c, C)
                cfg = {
                    **spatial,
                    "c_tile": (c_start, c_end),
                    "conv_padding": padding,
                    "output_sizes": (oh, ow),
                    "keep_dims_and_padding": is_conv1,
                }

                tile_configs.append(cfg)
                tiled_shapes.append(_compute_per_tile_shapes(cfg))

    if tile_c != C or is_conv1:
        _decompose_conv2d_node(
            model, node, tile_sizes, tiled_shapes, tile_configs
        )
    else:
        h_in = tile_y * stride[0]
        w_in = tile_x * stride[1]
        node.meta["tiled_shapes"] = tiled_shapes[0]
        node.meta["l2_tiling"] = (1, K // tile_k, Y // tile_y, X // tile_x)
        node.meta["tile_strides"] = {
            "input": (1, tile_c, h_in, w_in),
            "input_scale": (1, tile_c // bs, h_in, w_in),
        }


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


def _slice_tensor(node, dim, start, end, model):
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


def _make_linear_tiled_module(C, tile_c):
    """
    Factory function to create a LinearTiled module class with captured parameters.
    """
    class LinearTiled(torch.nn.Module):
        def __init__(self, block_size=16):
            super().__init__()
            self.block_size = block_size

        def get_scale_tile(self, input_scale, weight_scale, tidx):
            if input_scale is None or weight_scale is None:
                return None, None

            c0, c1 = tidx * tile_c, min((tidx + 1) * tile_c, C)
            bs = self.block_size

            in_s = input_scale[..., c0 // bs:c1 // bs]
            wt_s = weight_scale[..., c0 // bs:c1 // bs]

            return in_s, wt_s

        def get_csr_tile(self, A_data, A_indices, A_indptr, c0, c1):
            if A_data is None:
                return None, None, None

            return torch.ops.quantized_ops.slice_csr_matrix(
                A_data, A_indices, A_indptr, c0, c1, dim=1
            )

        def run_linear(
            self,
            input_tile,
            weight_tile,
            bias,
            scales,
            codes,
            A_csr,
        ):
            args = (input_tile, weight_tile, bias)

            if scales[0] is None or scales[1] is None:
                return torch.ops.aten.linear.default(*args)

            return torch.ops.quantized_ops.linear_mx(
                *args,
                input_scale=scales[0],
                weight_scale=scales[1],
                block_size=self.block_size,
                input_code=codes[0],
                weight_code=codes[1],
                A_data=A_csr[0],
                A_indices=A_csr[1],
                A_indptr=A_csr[2],
            )

        def forward(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            input_scale: Optional[torch.Tensor] = None,
            weight_scale: Optional[torch.Tensor] = None,
            input_code: Optional[torch.Tensor] = None,
            weight_code: Optional[torch.Tensor] = None,
            A_data: Optional[torch.Tensor] = None,
            A_indices: Optional[torch.Tensor] = None,
            A_indptr: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            psums = None

            for c in range(0, C, tile_c):
                c0, c1 = c, min(c + tile_c, C)

                tiled_input = input[..., c0:c1]
                tiled_weight = weight[..., c0:c1]

                scale_tiles = self.get_scale_tile(
                    input_scale, weight_scale, c // tile_c
                )

                csr_tiles = self.get_csr_tile(
                    A_data, A_indices, A_indptr, c0, c1
                )

                tiled_gemm = self.run_linear(
                    tiled_input,
                    tiled_weight,
                    bias if c1 == C else None,
                    scale_tiles,
                    (input_code, weight_code),
                    csr_tiles,
                )

                if psums is not None:
                    psums = psums + tiled_gemm
                else:
                    psums = tiled_gemm

            return psums

    return LinearTiled


def _make_matmul_tiled_module(C, tile_c):
    """
    Factory function to create a MatmulTiled module class with captured parameters.
    """
    class MatmulTiled(torch.nn.Module):
        def __init__(self, block_size=16):
            super().__init__()
            self.block_size = block_size

        def get_scale_tile(self, input_scale, weight_scale, tidx):
            if input_scale is None or weight_scale is None:
                return None, None

            c0, c1 = tidx * tile_c, min((tidx + 1) * tile_c, C)
            bs = self.block_size

            in_s = input_scale[..., c0 // bs:c1 // bs]
            wt_s = weight_scale[c0 // bs:c1 // bs]

            return in_s, wt_s

        def run_matmul(
            self,
            input_tile,
            weight_tile,
            scales,
            codes,
        ):
            args = (input_tile, weight_tile)

            if scales[0] is None or scales[1] is None:
                return torch.ops.aten.matmul.default(*args)

            return torch.ops.quantized_ops.matmul_mx(
                *args,
                input_scale=scales[0],
                weight_scale=scales[1],
                block_size=self.block_size,
                input_code=codes[0],
                weight_code=codes[1],
            )

        def forward(
            self,
            input: torch.Tensor,
            other: torch.Tensor,
            input_scale: Optional[torch.Tensor] = None,
            weight_scale: Optional[torch.Tensor] = None,
            input_code: Optional[torch.Tensor] = None,
            weight_code: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            psums = None

            for c in range(0, C, tile_c):
                c0, c1 = c, min(c + tile_c, C)

                tiled_input = input[..., c0:c1]
                tiled_weight = other[c0:c1]

                scale_tiles = self.get_scale_tile(
                    input_scale, weight_scale, c // tile_c
                )

                tiled_gemm = self.run_matmul(
                    tiled_input,
                    tiled_weight,
                    scale_tiles,
                    (input_code, weight_code),
                )

                if psums is not None:
                    psums = psums + tiled_gemm
                else:
                    psums = tiled_gemm

            return psums

    return MatmulTiled


def split_gemm_node(model, node, tile_sizes, tiled_shapes):
    """
    Transform a GEMM node (matmul/linear) into a tiled version along the
    reduction (C) dimension. Emits tiled sub-ops and replaces the original
    node in the FX graph.

    Args:
        model: FX GraphModule
        node: GEMM node to tile
        tile_sizes: tiling sizes on X, C, and K dimensions
        tiled_shapes: shape of inputs and outputs after tiling
    """
    x_tiled, c_tiled, k_tiled = tile_sizes

    input_shape = node.args[0].shape
    X = math.prod(input_shape[:-1])
    C = input_shape[-1]

    is_mat = is_matmul(node)
    weight_shape = node.args[1].shape
    K = weight_shape[-1] if is_mat else weight_shape[0]

    tiling = (X // x_tiled, K // k_tiled)

    if C == c_tiled:
        node.meta["tiled_shapes"] = tiled_shapes
        node.meta["l2_tiling"] = tiling
        return

    # Create the tiled module using factory function
    if is_matmul(node):
        GemmTiled = _make_matmul_tiled_module(C, c_tiled)
    else:
        GemmTiled = _make_linear_tiled_module(C, c_tiled)

    def load_arg(a):
        return map_arg(a, lambda n: n.value if isinstance(n, Node) else n)

    mod = GemmTiled(block_size=node.kwargs.get("block_size", 1))
    kwargs = {k: v for k, v in node.kwargs.items() if v is not None}
    kwargs.pop("block_size", None)
    gm = export_model(mod, load_arg(node.args[:3]), load_arg(kwargs))

    for n in list(gm.graph.nodes):
        if is_prunable_op(n):
            n.replace_all_uses_with(n.all_input_nodes[0])
            gm.graph.erase_node(n)
    gm.graph.lint()

    value_remap = {}
    replace_node_with_graph_module(model, gm, node, value_remap)

    # Update metadata on new nodes in the graph
    if (source_fn_st := node.meta.get("source_fn_stack")) is not None:
        source_fn = source_fn_st[-1][1]
    else:
        source_fn = node.target

    for n in list(value_remap.values()):
        if n.target == torch.ops.aten.slice.Tensor and n.args[0].op == "get_attr":
            c_start, c_end = n.args[2], n.args[3]
            with model.graph.inserting_before(n):
                sliced_param = _slice_tensor(n.args[0], 1, c_start, c_end, model)
            n.replace_all_uses_with(sliced_param)
            model.graph.erase_node(n)
            continue

        if n.target in [
            torch.ops.aten.slice.Tensor, torch.ops.aten.pad.default,
        ]:
            n.meta["dtype"] = n.args[0].meta.get("dtype")

        if n.target == node.target:
            n.meta.update({
                "tiled_shapes": copy.deepcopy(tiled_shapes),
                "l2_tiling": tiling,
                "dtype": node.meta.get("dtype"),
                "source_fn_stack": [(n.name, source_fn)],
            })


def get_valid_tiling(
    input_shape: Tuple[int, ...],
    min_sizes: Optional[Union[List[int], Tuple[int, ...]]] = None,
    order: Optional[Union[List[int], Tuple[int, ...]]] = None,
    fixed_dims: Optional[Union[List[int], Tuple[int, ...]]] = None,
    last_dim: Optional[int] = None,
    reverse: bool = False,
    round_robin: bool = False,
) -> Generator[Tuple[Tuple[int, ...], Tuple[int, ...]], None, None]:
    """
    Yields tile shapes by progressively reducing dimensions in a specified order.

    Args:
        input_shape: The original shape (e.g., (1024, 1024)).
        min_sizes: Minimum size for each dimension. If the list is shorter than
                   input_shape, it is padded with 1s on the left.
        order: Explicit order of dimension indices to reduce.
        fixed_dims: Indices of dims that should remain at full size.
        last_dim: Convenience arg to fix dimensions starting from this index.
        reverse: If True, reverses the traversal order (ignored if `order` is provided).
        round_robin: If True, cycles through dimensions reducing them one step at a time.
                     If False, fully reduces one dimension before moving to the next.

    Yields:
        (current_shape, tiling_factors)
    """
    ndim = len(input_shape)

    # --- 1. Normalize Inputs ---

    # helper: resolving negative indices to positive
    def resolve_idx(i): return i + ndim if i < 0 else i

    # Set up fixed dimensions set
    fixed_indices = set()
    if fixed_dims:
        fixed_indices.update(resolve_idx(d) for d in fixed_dims)
    if last_dim is not None:
        start = resolve_idx(last_dim)
        fixed_indices.update(range(start, ndim))

    # Set up traversal order
    if order:
        traversal_order = [resolve_idx(i) for i in order]
    else:
        traversal_order = list(range(ndim))
        if reverse:
            traversal_order.reverse()

    # Align min_sizes to input_shape length (pad left with 1s)
    targets = list(min_sizes) if min_sizes else []
    if len(targets) < ndim:
        targets = [1] * (ndim - len(targets)) + targets

    # --- 2. Pre-calculate Valid Factors ---

    # We calculate all valid tiling sizes for every dimension upfront.
    # A factor is valid if it divides the dimension and >= min_size.
    # Example: input 128 -> [128, 64, 32, ..., min_size]
    dim_factors = {}
    for i in range(ndim):
        limit = max(1, targets[i]) # Ensure min_size is at least 1
        size = input_shape[i]
        limit = min(limit, size)  # Cap limit to size

        if i in fixed_indices:
            factors = [size]
        else:
            # Generate factors in descending order
            factors = [
                f for f in range(size, limit - 1, -1)
                if size % f == 0 and (f % limit == 0) # Enforce limit as a multiple
            ]
            # Ensure the limit itself is included if not strictly divisible but valid
            if not factors or factors[-1] != limit:
                 factors.append(limit)

        dim_factors[i] = factors

    # --- 3. Traversal Logic ---

    # Current state: indices pointing to the current factor used for each dimension
    # initialized to 0 (which corresponds to the full input size)
    current_factor_indices = {i: 0 for i in range(ndim)}

    def get_current_state():
        """Constructs the shape and tiling tuple based on current indices."""
        shape = tuple(dim_factors[i][current_factor_indices[i]] for i in range(ndim))
        tiling = tuple(input_shape[i] // shape[i] for i in range(ndim))
        return shape, tiling

    # Yield the initial full shape
    yield get_current_state()

    if not round_robin:
        # --- Sequential Mode ---
        # Reduce Dim A fully, then move to Dim B, etc.
        for dim_idx in traversal_order:
            if dim_idx in fixed_indices:
                continue

            factors = dim_factors[dim_idx]
            # Iterate through the remaining factors for this dimension
            for i in range(1, len(factors)):
                current_factor_indices[dim_idx] = i
                yield get_current_state()

    else:
        # --- Round Robin Mode ---
        # Reduce Dim A (step 1), yield. Reduce Dim B (step 1), yield. Repeat.
        active_dims = [d for d in traversal_order if d not in fixed_indices]

        while True:
            progress_made = False

            for dim_idx in active_dims:
                current_idx = current_factor_indices[dim_idx]
                max_idx = len(dim_factors[dim_idx]) - 1

                # If this dimension can be reduced further
                if current_idx < max_idx:
                    # Move one step down in size
                    current_factor_indices[dim_idx] += 1
                    progress_made = True
                    yield get_current_state()

            # If we went through all dims and none could change, we are done
            if not progress_made:
                break


def get_node_to_key_map(node):
    args_and_kwargs = normalize_function(
        node.target,
        node.args,
        node.kwargs,
        normalize_to_only_use_kwargs=True
    )
    node_to_key = {
        n.meta.get('source_node', n): k
        for k, n in args_and_kwargs.kwargs.items() if isinstance(n, Node)
    }
    node_to_key[node] = "output"
    return node_to_key


def normalize_shape(node, shape):
    node_to_key = get_node_to_key_map(node)
    shape = {
        n: shape[k] for n, k in node_to_key.items() if k in shape
    }
    return shape


def _conv2d_layout(shape, is_weight=False, do_transpose=False):
    assert len(shape) == 4, "Conv2d shape must be 4D"
    if not do_transpose:
        return shape
    if is_weight:
        return (shape[2], shape[3], shape[1], shape[0])
    else:
        return (shape[0], shape[2], shape[3], shape[1])


def _calculate_conv2d_shapes(node, tile_sizes, divisor=None):
    bs = node.kwargs.get("block_size", 1)
    transposed = node.meta.get("transposed", False)

    y_tile, x_tile, c_tile, k_tile = tile_sizes
    c_scaled = c_tile // bs

    stride = _pair(get_arg_or_kwarg(node, 3, "stride", 1))
    padding = _pair(get_arg_or_kwarg(node, 4, "padding", 0))
    dilation = _pair(get_arg_or_kwarg(node, 5, "dilation", 1))

    weight_shape = node.args[1].shape
    kH, kW, _, _ = _conv2d_layout(weight_shape, True, not transposed)

    iy_tile = (
        (y_tile - 1) * stride[0] - 2 * padding[0]
        + dilation[0] * (kH - 1) + 1
    )
    ix_tile = (
        (x_tile - 1) * stride[1] - 2 * padding[1]
        + dilation[1] * (kW - 1) + 1
    )

    def apply_layout(shape, is_weight):
        return _conv2d_layout(shape, is_weight, transposed)

    return {
        "input": apply_layout((1, c_tile, iy_tile, ix_tile), False),
        "weight": apply_layout((k_tile, c_tile, kH, kW), True),
        "bias": (k_tile,),
        "input_scale": apply_layout((1, c_scaled, iy_tile, ix_tile), False),
        "weight_scale": apply_layout((k_tile, c_scaled, kH, kW), True),
        "output": apply_layout((1, k_tile, y_tile, x_tile), False),
    }


def _calculate_gemm_shapes(node, tile_sizes, divisor=None):
    bs = node.kwargs.get("block_size", 1)

    x_tiled, c_tiled, k_tiled = tile_sizes
    c_scaled = c_tiled // bs

    input_shape = node.args[0].shape
    tiled_input_shape = construct_tiled_shape(
        input_shape, x_tiled, list(range(len(input_shape) - 1))
    )

    batch_dims = tiled_input_shape[:-1]

    is_mat = is_matmul(node)
    weight_transposed = is_mat ^ node.meta.get("transposed", False)

    if weight_transposed:
        weight_shape = (c_tiled, k_tiled)
        weight_scale_shape = (c_scaled, k_tiled)
    else:
        weight_shape = (k_tiled, c_tiled)
        weight_scale_shape = (k_tiled, c_scaled)

    A_data = node.kwargs.get("A_data")
    A_indices = node.kwargs.get("A_indices")

    return {
        "input": batch_dims + (c_tiled,),
        "other" if is_mat else "weight": weight_shape,
        "bias": (k_tiled,),
        "input_scale": batch_dims + (c_scaled,),
        "weight_scale": weight_scale_shape,
        "A_data": tuple(A_data.shape) if A_data else None,
        "A_indices": tuple(A_indices.shape) if A_indices else None,
        "A_indptr": (x_tiled + 1,),
        "output": batch_dims + (k_tiled,),
    }


def _log_tiling_details(node, tiled_shapes, strategy):
    def fmt(s):
        if s is None: return "?"
        return str(tuple(s)).replace(" ", "")

    logger.info(f"Selected tiling for {node} (Strategy: {strategy}):")

    for n in node.all_input_nodes:
        if n in tiled_shapes and require_allocation(n):
            orig_shape = fmt(n.shape)
            tile_shape = fmt(tiled_shapes[n])
            logger.info(f"  In[{n}]: {orig_shape} -> {tile_shape}")

    orig_shape = fmt(node.shape)
    tile_shape = fmt(tiled_shapes[node])
    logger.info(f"  Out[{node}]: {orig_shape} -> {tile_shape}")


def _search_tiling(
    node,
    full_shape,
    min_sizes,
    shape_func,
    cache_size,
    bank_width,
    bank_size,
    order=None,
    last_dim=None,
):
    """
    Generic driver that iterates over banking strategies and valid tilings.
    """
    op_scope = _get_scope(node.target)
    node_to_key = get_node_to_key_map(node)
    key_to_node = {f"{op_scope}::{v}": k for k, v in node_to_key.items()}

    strategies = get_banking_strategies_for_op(node.target)

    for strategy in strategies:
        for tile_sizes, tiling in get_valid_tiling(
            full_shape, min_sizes=min_sizes, order=order, last_dim=last_dim
        ):
            logger.info(f"Trying tiling {tiling} with tile sizes {tile_sizes}")

            tiled_shapes = shape_func(node, tile_sizes, tiling)
            node_to_shape = normalize_shape(node, tiled_shapes)

            total_size, _ = strategy.evaluate(
                key_to_node, node, node_to_shape, bank_width, bank_size
            )

            if total_size <= cache_size:
                _log_tiling_details(node, node_to_shape, strategy)
                return tile_sizes, tiled_shapes

    logger.warning(f"Failed to tile {node} with cache size {cache_size}.")
    return None, None


def select_conv2d_tiling(node, unroll_dims, cache_size, bank_width, bank_size):
    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    N, K, Y, X = node.shape
    C = node.args[0].shape[1]

    full_shape = (Y, X, C, K)

    min_xy = int(math.sqrt(unroll_dims[0]))
    min_sizes = (min_xy, min_xy, unroll_dims[0], unroll_dims[1])

    order = (3, 0, 1, 2)

    logger.info(f"Running L2 tiling for matrix op: {node}")

    return _search_tiling(
        node=node,
        full_shape=full_shape,
        min_sizes=min_sizes,
        order=order,
        shape_func=_calculate_conv2d_shapes,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=bank_size
    )


def select_gemm_tiling(node, unroll_dims, cache_size, bank_width, bank_size):
    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    input_shape = node.args[0].shape
    X = math.prod(input_shape[:-1])
    C = input_shape[-1]

    is_mat = is_matmul(node)
    weight_shape = node.args[1].shape
    K = weight_shape[-1] if is_mat else weight_shape[0]

    # Pick a reduction dim that fits in a bank
    min_x_tile = min(sum(unroll_dims), X)
    input_bytes = get_node_bytes(node.args[0])
    num_c_tile = 1
    if bank_size is not None:
        for (c,), (num_c_tile,) in get_valid_tiling(
            (C,), min_sizes=(unroll_dims[0],)
        ):
            if min_x_tile * c * input_bytes <= bank_size:
                break

    full_shape = (X, C // num_c_tile, K)
    min_sizes = (min_x_tile, unroll_dims[0], unroll_dims[1])
    order = (2, 0, 1)

    logger.info(f"Running L2 tiling for matrix op: {node}")

    return _search_tiling(
        node=node,
        full_shape=full_shape,
        min_sizes=min_sizes,
        order=order,
        shape_func=_calculate_gemm_shapes,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=bank_size
    )


def run_matrix_op_l2_tiling(
    model, unroll, cache_size=None, num_banks=None, bank_width=None,
):
    """
    Perform tiling on GEMM operations to fit intermediate data into cache.

    Args:
        model: A model object with a FX Graph containing GEMM nodes.
        unroll (int): Systolic array input and output channel unrolling dimension. 
        cache_size (int): Total cache size in bytes.
        num_banks (int, optional): Number of cache banks for bank-aligned tiling.
        bank_width (int, optional): Width of memory for bank-aligned tiling.
    """
    graph = model.graph

    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE

    bank_size = None if num_banks is None else cache_size // num_banks

    for node in list(graph.nodes):
        if is_conv2d(node):
            tile_sizes, tiled_shape = select_conv2d_tiling(
                node, unroll, cache_size, None, bank_size
            )

            split_conv2d_node(model, node, tile_sizes)

        elif is_linear(node) or is_matmul(node):
            tile_sizes, tiled_shape = select_gemm_tiling(
                node, unroll, cache_size, None, bank_size
            )

            split_gemm_node(model, node, tile_sizes, tiled_shape)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def get_tiled_shape(shape, divisor):
    ndim = len(shape)
    m = len(divisor)
    if ndim > m:
        divisor = (1,) * (ndim - m) + divisor
    elif ndim < m:
        shape = (1,) * (m - ndim) + shape
    tiled_shape = []
    for i in range(len(shape)):
        tiled_shape.append(shape[i] // divisor[i] if shape[i] > 1 else shape[i])
    return tuple(tiled_shape[-ndim:])


def _calculate_vector_op_shapes(node, tile_sizes, divisor):
    node_to_key = get_node_to_key_map(node)
    shape = {}
    for n, k in node_to_key.items():
        if k == "output":
            if isinstance(node.value, torch.Tensor):
                shape[k] = tile_sizes
            elif isinstance(node.value, (tuple, list)):
                shape[k] = tuple(
                    get_tiled_shape(t.shape, divisor)
                    for t in node.value
                )
        else:
            shape[k] = get_tiled_shape(tuple(n.shape), divisor)
    return shape


def run_vector_op_node_l2_tiling(
    node,
    unroll,
    cache_size=None,
    num_banks=None,
    bank_width=None,
):
    if not is_elementwise_op(node) and node.target not in [
        torch.ops.aten.softmax.int,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.quantized_ops.layer_norm.default,
        torch.ops.quantized_ops.calculate_mx_qparam.default,
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    ]:
        return

    # Certain dimensions cannot be tiled, e.g., transpose and reduction dims
    last_dim = -1
    if node.target == torch.ops.quantized_ops.calculate_mx_qparam.default:
        last_dim = min(node.args[1])
    elif node.target in [
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    ]:
        last_dim = min(node.args[2])
    elif node.target == torch.ops.aten.transpose.int:
        last_dim = min(*node.args[1:])
    elif node.target == torch.ops.aten.permute.default:
        last_dim = next((i for i, d in enumerate(node.args[1]) if i != d), None)

    output_shape = (
        node.value.shape if isinstance(node.value, torch.Tensor)
        else node.value[1].shape
    )

    logger.info(f"Running L2 tiling for vector op: {node}")

    tile_sizes, tiled_shapes = _search_tiling(
        node=node,
        full_shape=output_shape,
        min_sizes=(unroll,),
        last_dim=last_dim,
        shape_func=_calculate_vector_op_shapes,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=None if num_banks is None else cache_size // num_banks,
    )

    if tile_sizes is not None:
        node.meta["tiled_shapes"] = tiled_shapes
        node.meta["l2_tiling"] = tuple(
            s // ts for s, ts in zip(output_shape, tile_sizes)
        )


def run_vector_op_l2_tiling(
    model, unroll, cache_size=None, num_banks=None, bank_width=None,
):
    """
    Perform tiling on vector operations in a model to fit intermediate data into cache.

    Args:
        model: A model object with a FX Graph containing vector operation nodes.
        unroll (int): Minimum unrolling dimension for vector operations.
        cache_size (int): Total cache size in bytes.
        bank_width (int, optional): Width of memory for bank-aligned tiling.
        num_banks (int, optional): Number of memory banks for bank-aligned tiling.
    """
    graph = model.graph

    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE

    for node in list(graph.nodes):
        run_vector_op_node_l2_tiling(node, unroll, cache_size, num_banks)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model
