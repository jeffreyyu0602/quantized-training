import logging
import math
import operator
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch.fx import GraphModule, Node

from .utils import get_arg_or_kwarg, _pair
from ..mapping import (
    duplicate_shared_nodes,
    propagate_shape,
)
from ..mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_gemm_op,
    is_indexing_or_concatenation_op,
    is_linear,
    is_matmul,
    is_nop,
    is_reshape_op,
)
from ...pt2e_utils import deduplicate_nodes, fetch_attr
from ...quantize_pt2e import create_getattr_from_value

logger = logging.getLogger(__name__)

__all__ = [
    "eliminate_reshape_with_no_effect",
    "transpose_conv2d_inputs_and_weights",
    "transpose_linear_weights",
]

TRANSPOSED_OPERATORS = {
    torch.ops.aten.conv2d.default: torch.ops.quantized_ops.conv2d.default,
    torch.ops.aten.max_pool2d.default: torch.ops.quantized_ops.max_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d.default: torch.ops.quantized_ops.adaptive_avg_pool2d.default,
    torch.ops.quantized_ops.conv2d_mx.default: torch.ops.quantized_ops.conv2d_mx.default,
}

AXES_ARG_INDEX_MAP = {
    torch.ops.quantized_ops.calculate_mx_qparam.default: 1,
    torch.ops.quantized_ops.dequantize.default: 3,
    torch.ops.quantized_ops.quantize.default: 3,
    torch.ops.quantized_ops.quantize_mx.default: 2,
}

NCHW_TO_NHWC = (0, 2, 3, 1)
NHWC_TO_NCHW = (0, 3, 1, 2)
WEIGHT_NCHW_TO_HWIO = (2, 3, 1, 0)


def conv2d_transposed(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    output = torch.ops.aten.conv2d.default(
        input.permute(0, 3, 1, 2),
        weight.permute(3, 2, 0, 1) if groups == 1 else weight,
        bias,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        groups,
    )
    return output.permute(0, 2, 3, 1)


def extract_conv2d_graph(model: GraphModule, start: Node, visited: Set[Node]) -> Set[Node]:
    """DFS downstream traversal to find conv2d nodes connected through elementwise ops."""
    stack = [start]
    nodes_in_graph = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        visited.add(node)
        nodes_in_graph.add(node)

        for user in list(node.users.keys()) + node.all_input_nodes:
            # Stack op cannot be fused because it creates a new dimension
            if user.target == torch.ops.aten.stack.default:
                continue

            # Only include reshape if it is a 4D tensor
            if is_reshape_op(user) and len(user.shape) == 4:
                stack.append(user)

            if (
                user.target == operator.getitem
                and user.args[0].target == torch.ops.quantized_ops.quantize_mx.default
            ):
                stack.append(user)

            if (
                user.target in TRANSPOSED_OPERATORS
                or is_elementwise_op(user)
                or is_indexing_or_concatenation_op(user)
                or user.target in [
                    torch.ops.aten.pad.default,
                    torch.ops.quantized_ops.calculate_mx_qparam.default,
                    torch.ops.quantized_ops.quantize_mx.default,
                ]
            ):
                stack.append(user)

    node_position = {n: i for i, n in enumerate(model.graph.nodes)}
    return sorted(nodes_in_graph, key=lambda n: node_position[n])


def remap_pad_after_permute(
    pad: Tuple[int, ...], dims: Tuple[int, ...], ndim: int
) -> Tuple[int, ...]:
    """
    Remap padding after permuting a tensor.

    Args:
        pad: Original pad tuple as in torch.nn.functional.pad (starts from last dim).
        dims: Permutation dimensions.
        ndim: Number of dimensions in the original tensor.

    Returns:
        Tuple[int, ...]: New pad tuple corresponding to permuted tensor.
    """
    # number of padded dimensions
    k = len(pad) // 2
    assert k <= ndim, "Pad dimensions exceed tensor dimensions"

    # original padded dims (from last to first)
    original_padded_dims = list(range(ndim - k, ndim))

    dim_to_new_index = {d: dims.index(d) for d in range(ndim)}

    new_pad_pairs = {i: (0, 0) for i in range(ndim)}

    # Assign padding for dimensions that were originally padded
    for i, orig_dim in enumerate(reversed(original_padded_dims)):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        new_pad_pairs[dim_to_new_index[orig_dim]] = (left, right)

    # Collect pads in reverse order (last-first)
    new_pad = []
    for i in sorted(new_pad_pairs.keys(), reverse=True):
        new_pad.extend(new_pad_pairs[i])

    return tuple(new_pad)


def _get_path_to_conv2d(node: torch.fx.Node):
    for user in node.users:
        if is_conv2d(user):
            return [node, user]

        if (
            is_nop(user) or is_indexing_or_concatenation_op(user)
            or user.target in [
                torch.ops.quantized_ops.quantize.default,
                torch.ops.aten.pad.default,
            ]
        ):
            path = _get_path_to_conv2d(user)
            if path is not None:
                return [node] + path
    return None


def _process_conv2d_input_nodes(
    node: Node, model: GraphModule, island_set: Set[Node]
):
    graph = model.graph
    path = _get_path_to_conv2d(node)

    # Case A: Input is a weight (Parameter) or weight scale
    if node.op == "get_attr" and path is not None:
        conv2d_node = path[-1]
        if is_depthwise_conv(conv2d_node) or path[-2] not in (
            conv2d_node.args[1], conv2d_node.kwargs.get("weight_scale")
        ):
            return

        logger.debug(f"Permuting parameter {node}")
        param = fetch_attr(model, node.target)
        param.data = param.data.permute(2, 3, 1, 0)

        node.meta["dims"] = WEIGHT_NCHW_TO_HWIO

    # Case B: Input is a node flow from outside the island
    if node.op != "get_attr" and len(node.shape) == 4:
        is_weight_node = (
            path is not None and id(path[-2]) == id(path[-1].args[1])
        )
        dims = WEIGHT_NCHW_TO_HWIO if is_weight_node else NCHW_TO_NHWC

        logger.debug(f"Insert permute after {node} with dims {dims}")
        with graph.inserting_after(node):
            permute_node = graph.call_function(
                torch.ops.aten.permute.default, (node, dims),
            )

        permute_node.meta["dims"] = dims
        permute_node.meta["dtype"] = node.meta.get("dtype")

        for user in list(node.users.keys()):
            if user in island_set:
                user.replace_input_with(node, permute_node)


def _rewrite_node_args_for_layout(node: Node) -> None:
    input_dims = node.all_input_nodes[0].meta.get("dims")
    node.meta["dims"] = input_dims

    args = tuple(node.args)

    if node.target == torch.ops.aten.pad.default:
        pad = remap_pad_after_permute(args[1], input_dims, node.value.ndim)
        node.args = args[:1] + (pad,) + args[2:]

    if is_indexing_or_concatenation_op(node):
        dim = get_arg_or_kwarg(node, 1, "dim", 0)
        if dim < 0:
            dim = dim + len(input_dims)
        node.args = args[:1] + (input_dims.index(dim),) + args[2:]

    if is_reshape_op(node):
        if node.target == torch.ops.aten.transpose.int:
            dims = (args[1], args[2])
        else:
            dims = args[1]
        dims = [d + max(input_dims) + 1 if d < 0 else d for d in dims]
        dims = tuple(input_dims.index(d) for d in dims)
        node.args = args[:1] + (dims,) + args[2:]

    idx = AXES_ARG_INDEX_MAP.get(node.target)
    if idx is not None and idx < len(args) and args[idx] is not None:
        axes = [a + len(input_dims) if a < 0 else a for a in args[idx]]
        axes = tuple(input_dims.index(a) for a in axes)
        node.args = args[:idx] + (axes,) + args[idx + 1:]

    if node.target in TRANSPOSED_OPERATORS:
        node.target = TRANSPOSED_OPERATORS[node.target]
        node.meta["transposed"] = True


def transpose_conv2d_inputs_and_weights(model: GraphModule):
    graph = model.graph
    visited_nodes: Set[Node] = set()

    torch.nn.functional.conv2d = conv2d_transposed

    for node in list(graph.nodes):
        if node in visited_nodes or node.target not in TRANSPOSED_OPERATORS:
            continue

        # Extract the cluster of nodes that can share the NHWC layout
        island_nodes = extract_conv2d_graph(model, node, visited_nodes)
        island_set = set(island_nodes)

        for node_to_treat in island_nodes:
            # Inspect inputs to see if they come from outside the island (NCHW)
            for input_node in list(node_to_treat.all_input_nodes):
                if input_node in island_set or "dims" in input_node.meta:
                    continue

                _process_conv2d_input_nodes(input_node, model, island_set)

            for user in list(node_to_treat.users.keys()):
                if user in island_set or "dims" in user.meta:
                    continue

                logger.debug(f"Insert permute before {user} with dims (0, 3, 1, 2)")
                with graph.inserting_before(user):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default,
                        (node_to_treat, NHWC_TO_NCHW),
                    )
                permute_node.meta["dtype"] = node_to_treat.meta.get("dtype")
                user.replace_input_with(node_to_treat, permute_node)

            _rewrite_node_args_for_layout(node_to_treat)

            def permute(t, dims):
                return tuple(t[i] for i in dims)

            tiled_shapes = node_to_treat.meta.get("tiled_shapes")
            if is_conv2d(node_to_treat) and tiled_shapes is not None:
                for key, arg in [
                    ("input", node_to_treat.args[0]),
                    ("weight", node_to_treat.args[1])
                ]:
                    input_dims = arg.meta["dims"]
                    tiled_shapes[key] = permute(tiled_shapes[key], input_dims)

                    scale_key = f"{key}_scale"
                    if scale_key in tiled_shapes:
                        tiled_shapes[scale_key] = permute(
                            tiled_shapes[scale_key], input_dims
                        )

                tiled_shapes["output"] = permute(tiled_shapes["output"], NCHW_TO_NHWC)

                tiling = node_to_treat.meta["l2_tiling"]
                node_to_treat.meta["l2_tiling"] = permute(tiling, NCHW_TO_NHWC)

                if stride := node_to_treat.meta.get("tile_strides"):
                    stride["input"] = permute(stride["input"], NCHW_TO_NHWC)
                    stride["input_scale"] = permute(stride["input_scale"], NCHW_TO_NHWC)
                    node_to_treat.meta["tile_strides"] = stride

    graph.lint()
    model.recompile()
    return model


def eliminate_reshape_with_no_effect(model: GraphModule):
    deleted_nodes = set()
    for node in list(model.graph.nodes):
        if not is_reshape_op(node) or node in deleted_nodes:
            continue

        curr_node = node
        input_node = node.all_input_nodes[0]

        group = []
        while len(curr_node.users) == 1 and (is_reshape_op(curr_node) or is_nop(curr_node)):
            group.append(curr_node)
            curr_node = next(iter(curr_node.users))

        input_tensor = input_node.value.flatten()
        while group and torch.any(group[-1].value.flatten() != input_tensor):
            group.pop()

        if len(group) <= 1:
            continue

        logger.debug(f"Eliminating reshape group: {[n.name for n in group]}")

        output_shape = group[-1].value.shape

        with model.graph.inserting_before(node):
            reshape_node = model.graph.call_function(
                torch.ops.aten.reshape.default,
                (input_node, output_shape),
            )

        propagate_shape(reshape_node)

        group[-1].replace_all_uses_with(reshape_node)

        for n in reversed(group):
            model.graph.erase_node(n)
            deleted_nodes.add(n)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def make_linear_wrapper(transpose=False, skip_fc=False):
    """
    Returns a function that wraps torch.nn.functional.linear with optional
    weight transposition.
    """
    def wrapped_linear(input, weight, bias=None):
        is_fc = math.prod(input.shape[:-1]) == 1
        do_transpose = transpose and not (skip_fc and is_fc)
        return torch.ops.aten.linear.default(
            input, weight.T if do_transpose else weight, bias
        )
    return wrapped_linear


def make_matmul_wrapper(transpose=False, skip_fc=False):
    """
    Returns a function that wraps torch.matmul with optional transposition of
    the second argument.
    """
    def wrapped_matmul(input, other):
        is_fc = math.prod(input.shape[:-1]) == 1
        do_transpose = transpose and not (skip_fc and is_fc)
        return torch.ops.aten.matmul.default(
            input, other if do_transpose else other.T
        )
    return wrapped_matmul


def find_upstream_matching_transpose(
    tnode: Node,
    *,
    max_hops: int = 64,
) -> Tuple[Optional[Node], List[Node]]:
    """
    Given a transpose node `tnode` that must be aten.transpose.int(-2, -1),
    walk upstream through a small, explicit set of allowed ops to find an
    earlier/matching transpose(-2, -1). Returns (found_transpose, path).

    The search explores all input edges recursively (up to `max_hops`).
    `path` is the sequence of nodes from `tnode` to the match (inclusive).
    """
    if tnode.target != torch.ops.aten.transpose.int:
        return None

    allowed_targets = {
        torch.ops.aten.select.int,
        torch.ops.quantized_ops.calculate_mx_qparam.default,
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
        torch.ops.quantized_ops.quantize_mx.default,
    }

    def dfs(cur: Node, hops: int) -> Optional[List[Node]]:
        if hops >= max_hops:
            return None
        path = [cur]

        # Found a matching transpose(-2, -1) or a graph input
        if cur.target == torch.ops.aten.transpose.int:
            return path
        if cur.op == "get_attr":
            return path

        if is_nop(cur) or cur.target in allowed_targets:
            for inp in cur.all_input_nodes:
                path.extend(dfs(inp, hops + 1) or [])
        return path

    found_path = dfs(tnode.args[0], 0)
    if not found_path:
        return None
    return list(set([tnode] + found_path))


def _rank(n: Node) -> int:
    return len(n.shape)


def _norm_axes(args, r: int) -> set:
    """Convert negative dims to positive indices."""
    axes = []
    for a in args[1:]:
        if isinstance(a, int):
            axes.append(a if a >= 0 else r + a)
    return set(axes)


def _insert_transposed_input(arg: Node, model: GraphModule):
    with model.graph.inserting_after(arg):
        if arg.op == "get_attr":
            transposed = create_getattr_from_value(
                model, model.graph, arg.name + "_T", arg.value.mT
            )
        else:
            transposed = model.graph.call_function(
                torch.ops.aten.transpose.int, (arg, -2, -1)
            )
    transposed.meta["dtype"] = arg.meta.get("dtype")
    return transposed


def _fix_axes_after_transpose(node: Node) -> List[int]:
    if (index := AXES_ARG_INDEX_MAP.get(node.target)) is None:
        return

    axes = get_arg_or_kwarg(node, index, "axes")
    rank = _rank(node)

    # Build forward and inverse permutation for transpose(-2, -1)
    perm = list(range(rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    inv_perm = [perm.index(i) for i in range(rank)]

    # Normalize negative axes first
    norm_axes = [(a + rank) % rank for a in axes]

    # Apply inverse permutation
    new_axes = tuple(inv_perm[a] for a in norm_axes)
    node.args = node.args[:index] + (new_axes,) + node.args[index + 1 :]


def _fuse_quantize_mx_last_axis(model: GraphModule):
    """
    Replace calculate_mx_qparam + quantize with quantize_mx when the
    quantization is performed along the last axis.
    """
    graph = model.graph
    for node in list(graph.nodes):
        if node.target != torch.ops.quantized_ops.calculate_mx_qparam.default:
            continue

        axes = get_arg_or_kwarg(node, 1, "axes")
        rank = _rank(node)
        if axes != (rank - 1,) and axes != (-1,):
            continue

        args = node.args[1:] + (None,) * (5 - len(node.args[1:]))

        quantize_node = next(iter(
            n for n in node.users
            if n.target == torch.ops.quantized_ops.quantize.default
        ))

        assert quantize_node.args[0] == node.args[0], "Unexpected quantize input"

        qmap = quantize_node.args[5]
        output_code = get_arg_or_kwarg(quantize_node, 6, "output_code")
        new_code = None

        with graph.inserting_before(node):
            new_qmap = graph.node_copy(qmap)
            if output_code is not None:
                new_code = graph.node_copy(output_code)
            quantize_mx_node = graph.call_function(
                torch.ops.quantized_ops.quantize_mx.default,
                (node.args[0], new_qmap) + args + (new_code,)
            )
            scale_node = graph.call_function(
                operator.getitem, (quantize_mx_node, 0)
            )
            output_node = graph.call_function(
                operator.getitem, (quantize_mx_node, 1)
            )

        propagate_shape(new_qmap, model)
        if new_code is not None:
            propagate_shape(new_code, model)
        propagate_shape(quantize_mx_node, model)
        propagate_shape(scale_node, model)
        propagate_shape(output_node, model)

        scale_node.meta["dtype"] = node.meta.get("dtype")
        output_node.meta["dtype"] = quantize_node.meta.get("dtype")
        quantize_mx_node.meta["dtype"] = (
            scale_node.meta.get("dtype"), output_node.meta.get("dtype")
        )

        node.replace_all_uses_with(scale_node)
        quantize_node.replace_all_uses_with(output_node)

        logger.info(f"Replaced {node} and {quantize_node} with {quantize_mx_node}")

    graph.lint()
    model.recompile()
    return model


def process_double_transpose_chain(
    model: GraphModule, chain: List[Node], transposed_nodes: Dict[Node, Node] = None
) -> bool:
    """
    Optimizes a chain like [select_3, select_2, quantize_default_1, transpose_3]
    when there's a matching matmul-side transpose (user of chain[0]).

    Steps:
      1. Check if the two transposes cancel (considering selects).
      2. If yes, detach intermediate nodes and remove the redundant transpose.

    Returns:
        bool: True if optimization was applied, else False.
    """
    graph = model.graph

    if transposed_nodes is None:
        transposed_nodes = {}

    chain = [n for n in chain if n.op == "call_function"]
    if not chain or len(chain) < 2:
        return False

    up_t = chain[0]
    down_t = chain[-1]

    if up_t.target != torch.ops.aten.transpose.int:
        return False

    # Validate transpose axes
    down_rank = _rank(down_t)
    if _norm_axes(down_t.args, down_rank) != {down_rank - 2, down_rank - 1}:
        return False

    up_rank = _rank(up_t)
    if _norm_axes(up_t.args, up_rank) != {up_rank - 2, up_rank - 1}:
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if up_rank < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the upstream transpose node
    chain = duplicate_shared_nodes(graph, chain[1:])

    # Rewrite graph to remove cancelling transposes
    for n in chain:
        for arg in n.all_input_nodes:
            if arg == up_t:
                n.replace_input_with(up_t, up_t.args[0])
                continue
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])
        _fix_axes_after_transpose(n)

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not up_t.users:
        graph.erase_node(up_t)

    return True


def move_transpose_before_dq(
    model: GraphModule, chain: List[Node], transposed_nodes: Dict[Node, Node] = None
) -> bool:
    """
    Optimizes a chain like [dequantize_default, select_3, select_2, transpose_3].

    Steps:
      1. Check if there's a dequantize operation in the chain.
      2. If yes, move the transpose before the dequantize.

    Returns:
        bool: True if optimization was applied, else False.
    """
    graph = model.graph

    if transposed_nodes is None:
        transposed_nodes = {}

    chain = [n for n in chain if n.op == "call_function"]
    for i, n in enumerate(chain):
        if n.target == torch.ops.quantized_ops.dequantize.default:
            break

    chain = chain[i:]  # Keep only from dequantize to end

    if not chain or len(chain) < 2:
        return False

    down_t = chain[-1]

    # Validate transpose axes
    down_rank = len(down_t.shape)
    if _norm_axes(down_t.args, down_rank) != {down_rank - 2, down_rank - 1}:
        return False

    # Ensure selects are on first dimension only
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if any(n.args[1] != 0 for n in selects):
        return False

    chain = duplicate_shared_nodes(graph, chain)
    dequantize_node = chain[0]

    # Insert transpose after dequantize input
    dq_input = dequantize_node.args[0]
    up_t = next((
        n for n in dq_input.users if n.target == torch.ops.aten.transpose.int
    ), None)
    if up_t is not None and up_t.meta.get("dtype") == dq_input.meta.get("dtype"):
        dequantize_node.replace_input_with(dq_input, up_t)
    else:
        with graph.inserting_after(dq_input):
            up_t = graph.call_function(
                torch.ops.aten.transpose.int, (dq_input, -2, -1)
            )
        up_t.meta["dtype"] = dq_input.meta.get("dtype")
        dequantize_node.replace_input_with(dq_input, up_t)
        propagate_shape(up_t)

    for n in chain:
        for arg in n.all_input_nodes:
            if arg in chain or arg.value.ndim < 2 or arg == up_t:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])
        _fix_axes_after_transpose(n)

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    return True


def fold_transpose_into_constant(
    model: GraphModule, chain: List[Node], transposed_nodes: Dict[Node, Node] = None
) -> bool:
    graph = model.graph
    if not chain or len(chain) < 2:
        return False

    if transposed_nodes is None:
        transposed_nodes = {}

    attr_node = chain[0]
    down_t = chain[-1]

    if attr_node.op != "get_attr":
        return False

    # Ensure selects are on first dimension only
    up_rank = _rank(attr_node)
    selects = [n for n in chain if n.target == torch.ops.aten.select.int]
    if up_rank < len(selects) + 2 or any(n.args[1] != 0 for n in selects):
        return False

    # We don't need to duplicate the transpose node
    chain = duplicate_shared_nodes(model.graph, chain[1:])

    for n in chain:
        for arg in n.all_input_nodes:
            if arg in chain or arg.value.ndim < 2:
                continue
            if arg not in transposed_nodes:
                transposed_nodes[arg] = _insert_transposed_input(arg, model)
            n.replace_input_with(arg, transposed_nodes[arg])

    down_t.replace_all_uses_with(down_t.args[0])
    graph.erase_node(down_t)

    if not attr_node.users:
        graph.erase_node(attr_node)

    return True


def transpose_linear_weights(
    model: GraphModule, transpose_weight, transpose_fc: bool = False
):
    """
    Transpose the weights of linear layers in the given FX graph module.

    Args:
        model (GraphModule): The FX graph module to transform.
        transpose_weight (bool): Whether to transpose weights of linear layers.
        transpose_fc (bool): Whether to transpose weights of fully connected layers.

    Returns:
        GraphModule: The transformed FX graph module with transposed weights.
    """
    skip_fc = not transpose_fc
    torch.nn.functional.linear = make_linear_wrapper(transpose_weight, skip_fc)
    torch.matmul = make_matmul_wrapper(transpose_weight, skip_fc)

    transposed_nodes = {}

    def _update_tiled_shapes(node: Node):
        if (tiled_shapes := node.meta.get("tiled_shapes")) is None:
            return

        if "weight" in tiled_shapes:
            w0, w1 = tiled_shapes["weight"]
            tiled_shapes["weight"] = (w1, w0)

        if "weight_scale" in tiled_shapes:
            s0, s1 = tiled_shapes["weight_scale"]
            tiled_shapes["weight_scale"] = (s1, s0)

    def _handle_linear(node: Node, is_fc) -> None:
        if (is_fc and skip_fc) or (not is_fc and not transpose_weight):
            return

        weight_node = node.args[1]
        weight = fetch_attr(model, weight_node.target)
        weight.data = weight.data.T

        if (scale_node := node.kwargs.get("weight_scale")) is not None:
            scale = fetch_attr(model, scale_node.target)
            scale.data = scale.data.T

        # Mark spmm_csr users as having a transposed weight.
        for user in list(weight_node.users):
            if user.target == torch.ops.quantized_ops.spmm_csr.default:
                user.kwargs = {**user.kwargs, "weight_transposed": True}

        if node.target == torch.ops.aten.linear.default:
            node.target = torch.ops.quantized_ops.linear.default

        _update_tiled_shapes(node)
        node.meta["transposed"] = True

    def _transpose_node(node: Node, user: Node):
        with model.graph.inserting_before(user):
            new_node = model.graph.call_function(
                torch.ops.aten.transpose.int, (node, -2, -1)
            )

        new_node.meta["dtype"] = node.meta.get("dtype")
        propagate_shape(new_node, model)
        user.replace_input_with(node, new_node)

        path = find_upstream_matching_transpose(new_node)
        if not path:
            return None

        node_order = {n: i for i, n in enumerate(model.graph.nodes)}
        sorted_path = sorted(path, key=lambda n: node_order[n])

        success = process_double_transpose_chain(
            model, sorted_path, transposed_nodes
        )
        return None if success else sorted_path

    def _handle_matmul(node: Node, is_fc) -> None:
        if (is_fc and transpose_fc) or (not is_fc and transpose_weight):
            return

        weight_node = node.args[1]
        path = _transpose_node(weight_node, node)
        if path is not None:
            move_transpose_before_dq(model, path, transposed_nodes)

        if (scale_node := node.kwargs.get("weight_scale")) is not None:
            path = _transpose_node(scale_node, node)
            if path is not None:
                fold_transpose_into_constant(model, path, transposed_nodes)

        if node.target == torch.ops.aten.matmul.default:
            node.target = torch.ops.quantized_ops.matmul.default

        _update_tiled_shapes(node)
        node.meta["transposed"] = True

    for node in list(model.graph.nodes):
        if is_gemm_op(node):
            input_node = node.args[0]
            is_fc = math.prod(input_node.shape[:-1]) == 1

            if is_linear(node):
                _handle_linear(node, is_fc)

            if is_matmul(node):
                _handle_matmul(node, is_fc)

    deduplicate_nodes(model.graph)
    _fuse_quantize_mx_last_axis(model)

    model.graph.lint()
    model.recompile()
    return model
