import collections
import itertools
import logging
import math
import operator
from itertools import repeat
from typing import Callable, List, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .mapping import (
    get_parameter_or_buffer,
    propagate_shape,
    replace_node_with_graph_module,
    get_node_bytes,
)
from .mapping_utils import (
    _is_gemm_op,
    _is_matmul,
    _is_elementwise_op,
    _is_nop,
    _is_reshape_op,
    _is_indexing_or_concatenation_op,
)
from ..pt2e_utils import get_aten_graph_module
from ..quantize_pt2e import create_getattr_from_value, export_model
from ..quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "convert_cat_and_stack_as_stack_on_dim0",
    "convert_cat_with_mismatched_shapes_to_stack",
    "convert_expand_to_memory_copy",
    "eliminate_reshape_with_no_effect",
    "extract_input_preprocessor",
    "get_conv_bn_layers",
    "transpose_conv2d_inputs_and_weights",
    "pad_gemm_inputs_to_hardware_unroll_size",
    "pad_vector_ops_to_hardware_unroll_size",
    "pad_vit_embeddings_output",
    "replace_conv2d_with_im2col",
    "replace_target_with_vmap",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "replace_target",
    "rewrite_fx_graph",
    "run_matrix_op_l2_tiling",
    "run_vector_op_l2_tiling",
    "transpose_linear_weights",
    "strip_softmax_dtype",
]


DEFAULT_CACHE_SIZE = 8 * 1024 * 1024  # 8 MiB


def get_conv_bn_layers(model):
    layers = []
    module_names = list(model._modules)
    for k, name in enumerate(module_names):
        if len(list(model._modules[name]._modules)) > 0:
            conv_bn_pairs = get_conv_bn_layers(model._modules[name])
            layers.extend([[f'{name}.{conv}', f'{name}.{bn}'] for conv, bn in conv_bn_pairs])
        else:
            if isinstance(model._modules[name], torch.nn.BatchNorm2d):
                if isinstance(model._modules[module_names[k-1]], torch.nn.Conv2d):
                    layers.append([module_names[k-1], name])
    return layers


def fuse_conv_bn(model: torch.fx.GraphModule) -> torch.nn.Module:
    """
    Fuses convolution/BN and linear/BN layers for inference purposes.
    """
    from torch.nn.utils import fuse_conv_bn_weights

    for node in list(model.graph.nodes):
        if (
            node.target == torch.ops.aten._native_batch_norm_legit.default and
            node.args[0].target == torch.ops.aten.conv2d.default and
            len(node.args[0].users) == 1 and
            node.args[5] == False  # inference mode
        ):
            n = node.args[0]

            conv_w = model.get_parameter(n.args[1])
            conv_b = model.get_parameter(n.args[2])

            bn_w = model.get_parameter(node.args[1])
            bn_b = model.get_parameter(node.args[2])
            bn_rm = model.get_buffer(node.args[3])
            bn_rv = model.get_buffer(node.args[4])
            bn_eps = node.args[7]

            fused_conv_w, fused_conv_b = fuse_conv_bn_weights(
                conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b
            )

            model.register_parameter(n.args[1].target, fused_conv_w)
            model.register_parameter(n.args[2].target, fused_conv_b)

            node.replace_all_uses_with(n)
            model.graph.erase_node(node)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()

    return model


def convert_cat_and_stack_as_stack_on_dim0(model: GraphModule):
    """
    Transforms occurrences of `torch.cat` and `torch.stack` operations on non-zero dimensions
    into a `torch.stack` on the 0th dimension, followed by a `permute` operation to restore
    the original order.

    Args:
        model (GraphModule): The PyTorch FX GraphModule to be modified.

    Returns:
        GraphModule: The transformed GraphModule with `torch.cat` and `torch.stack` operations adjusted.
    """
    graph = model.graph
    for node in list(graph.nodes):
        if node.target not in [
            torch.ops.aten.cat.default, torch.ops.aten.stack.default
        ]:
            continue
        cat_node = node

        if not all(hasattr(n, "shape") for n in cat_node.args[0]):
            logger.warning(f"Node {cat_node} does not have shape attributes for all inputs.")
            continue

        shapes = [n.shape for n in cat_node.args[0]]
        input_shape = list(shapes[0])

        if not all(list(s) == input_shape for s in shapes):
            logger.warning(
                "Concatenated tensors have different shapes in node %s. Shapes: %s",
                cat_node, shapes
            )
            continue

        concat_dim = cat_node.args[1] if len(cat_node.args) > 1 else 0
        if concat_dim < 0:
            concat_dim += len(input_shape)

        if len(cat_node.args) == 1 or concat_dim == 0:
            continue

        # Always stack along the first dimension
        if cat_node.target == torch.ops.aten.stack.default:
            cat_node.args = (cat_node.args[0], 0)
            stack_node = cat_node
        else:
            with graph.inserting_after(cat_node):
                stack_node = graph.call_function(
                    torch.ops.aten.stack.default, (cat_node.args[0], 0)
                )
        propagate_shape(stack_node)

        # Permute the concatenated tensor to match the original order
        dims = list(range(len(input_shape) + 1))[1:]
        dims = dims[:concat_dim] + [0] + dims[concat_dim:]

        logger.info(f"Converting {cat_node} to stack on dim 0 with permute {dims}")

        with graph.inserting_after(stack_node):
            permute_node = graph.call_function(
                torch.ops.aten.permute.default, (stack_node, dims),
            )
        propagate_shape(permute_node)
        # get_source_partitions expects 'permute' as the source function. This is
        # hacky but there is no other way to set this meta field properly.
        permute_node.meta['source_fn_stack'] = [(permute_node.name, 'permute')]
        output_node = permute_node

        # Flatten the permuted tensor if it is a cat operation
        if cat_node.target == torch.ops.aten.cat.default:
            with graph.inserting_after(permute_node):
                output_node = graph.call_function(
                    torch.ops.aten.flatten.using_ints,
                    (permute_node, concat_dim, concat_dim + 1),
                )
            propagate_shape(output_node)

        # Replace all use of the cat node with the new node
        for node in list(cat_node.users):
            if id(node) == id(output_node):
                continue
            node.replace_input_with(cat_node, output_node)

        if cat_node.target == torch.ops.aten.cat.default:
            graph.erase_node(cat_node)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def convert_cat_with_mismatched_shapes_to_stack(model: GraphModule):
    """
    Convert `torch.cat` operations where input tensors have different shapes by replacing them
    with a `torch.stack` operation along the concatenated dimensions.

    Args:
        model (GraphModule): The PyTorch FX GraphModule to be modified.

    Returns:
        GraphModule: The transformed GraphModule with `torch.cat` operations adjusted to `torch.stack`.
    """
    partitions = get_source_partitions(model.graph, [torch.cat])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        node = partition.output_nodes[0]
        if node.target != torch.ops.aten.cat.default:
            continue

        input_shape = list(node.args[0][0].shape)
        if all(list(n.shape) == input_shape for n in node.args[0][1:]):
            continue

        logger.info(f"Node {node} has different input shapes")
        dim = node.args[1]

        args = map_arg(node.args, lambda n: n.value)
        shape = list(args[0][0].shape[:dim])

        class Concat(torch.nn.Module):
            def forward(self, *inputs):
                result = []
                for idx in itertools.product(*[range(dim) for dim in shape]):
                    tensor = torch.cat([x[idx] for x in inputs], dim=0)
                    result.append(tensor)
                output = torch.stack(result, dim=0)
                return output.reshape(*shape, *output.shape[1:])

        gm = export_model(Concat(), (*args[0],))
        replace_node_with_graph_module(model, gm, node)

    model.graph.lint()

    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def convert_expand_to_memory_copy(model: torch.fx.GraphModule):
    """
    Convert `torch.expand` operations into explicit memory copying by replicating input elements.
    This replaces implicit broadcasting with actual memory duplication, ensuring that expanded
    dimensions are materialized as stacked tensors.

    Args:
        model (torch.fx.GraphModule): The PyTorch FX GraphModule to be modified.

    Returns:
        torch.fx.GraphModule: The transformed GraphModule where `torch.expand` operations are
        replaced with explicit memory copies.
    """
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.expand.default:
            continue

        # Skip if the expand operation is a no-op
        if all(x == 1 or x == -1 for x in node.args[1]):
            continue

        input_node = node.args[0]
        sizes = node.args[1]
        original_shape = input_node.meta["val"].shape
        assert len(sizes) >= len(original_shape), (
            "Sizes must have at least as many dimensions as the original tensor."
        )

        # Add singleton dimensions to match the size length
        while len(original_shape) < len(sizes):
            input = input.unsqueeze(0)
            original_shape = input.shape

        class Expand(torch.nn.Module):
            def forward(self, input):
                # Stack along the first dimension repeatedly to create the expanded shape
                for dim, size in enumerate(sizes):
                    if input.shape[dim] == 1 and size > 1:
                        stacked_tensors = []
                        for _ in range(size):
                            stacked_tensors.append(input.squeeze(dim) + 0)
                        input = torch.stack(stacked_tensors, dim=dim)
                    elif input.shape[dim] != size:
                        raise ValueError(
                            f"Cannot expand dimension {dim} from {input.shape[dim]} to {size}."
                        )

                return input

        gm = export_model(Expand(), (input_node.meta["val"],))
        replace_node_with_graph_module(model, gm, node)
        model.graph.erase_node(node)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def replace_interpolate():
    from torch.library import Library, impl

    template = (
        "interpolate(Tensor input, SymInt[] size, float[]? scale_factor = None,"
        "str mode = 'nearest', bool? align_corners = None, "
        "bool? recompute_scale_factor = None, bool antialias = False) -> Tensor"
    )

    global m
    m = Library("custom", "DEF")
    m.define(template)

    orig_interpolate = torch.nn.functional.interpolate

    @impl(m, "interpolate", "CompositeExplicitAutograd")
    def interpolate(*args, **kwargs):
        return orig_interpolate(*args, **kwargs)

    torch.nn.functional.interpolate = torch.ops.custom.interpolate


def replace_rmsnorm_with_layer_norm(model, layer_norm, example_input):
    """Replace LLaMA RMSNorm with ATen layer_norm
    """
    original_graph = model.graph

    pattern = get_aten_graph_module(layer_norm, example_input)
    _convert_scalars_to_attrs(pattern)
    pattern_graph = pattern.graph

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)
    print(f"Found {len(_matches)} matches")

    weight_node = next(iter(n for n in pattern_graph.nodes if n.target == "weight"))

    for match in _matches:
        input_node = match.placeholder_nodes[0]
        output_node = match.returning_nodes[0]
        input_shape = input_node.meta["val"].shape
        new_weight_node = match.nodes_map[weight_node]
        layer_norm_inputs = [input_node, [input_shape[-1]], new_weight_node]

        with original_graph.inserting_before(output_node):
            new_node = original_graph.call_function(
                torch.ops.aten.layer_norm.default,
                tuple(layer_norm_inputs),
                {}
            )

        output_node.replace_all_uses_with(new_node)
        original_graph.erase_node(output_node)

        new_node.meta["source_fn_stack"] = [(new_node.name, "layer_norm")]

    original_graph.lint()
    original_graph.eliminate_dead_code()
    model.recompile()


def replace_target_with_vmap(
    model: GraphModule,
    target: Callable
) -> GraphModule:
    nodes_map = {}
    for node in list(model.graph.nodes):
        if node.target != target:
            continue

        if (get_attr_node := nodes_map.get(node.target, None)) is None:
            values = (torch.arange(2 ** 16, dtype=torch.int16).view(torch.bfloat16))
            code = node.target(values, *node.args[1:])

            with model.graph.inserting_before(node):
                get_attr_node = create_getattr_from_value(
                    model, model.graph, f'_tensor_constant_', code
                )

            nodes_map[node.target] = get_attr_node

        with model.graph.inserting_before(node):
            new_node = model.graph.call_function(
                torch.ops.quantized_ops.vmap.default,
                (node.args[0], get_attr_node)
            )

        new_node.meta = node.meta
        new_node.meta["source_fn_stack"] = [(new_node.name, "vmap")]

        node.replace_all_uses_with(new_node)
        model.graph.erase_node(node)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def strip_softmax_dtype(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.softmax.int:
            node.args = node.args[:2]

    graph.lint()
    model.recompile()
    return model


def slicing_and_padding_cancel_out(shape, slice_dim, start, end, pad):
    ndim = len(shape)
    k = len(pad) // 2

    pad_pairs = list(reversed(list(zip(pad[0::2], pad[1::2]))))
    full_pad_pairs = [(0, 0)] * (ndim - k) + pad_pairs

    for dim, (left, right) in enumerate(full_pad_pairs):
        if dim != slice_dim:
            if left != 0 or right != 0:
                return False
        else:
            if left != start or right != shape[slice_dim] - end:
                return False

    return True


def pad_gemm_inputs_to_hardware_unroll_size(
    model: torch.fx.GraphModule,
    C_unroll,
    K_unroll,
) -> torch.fx.GraphModule:
    """
    Pad inputs and weights to conv2d nodes in a torch.fx.GraphModule so that
    the input channels (C) and output channels (K) are multiples of the provided
    unroll factors.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        C_unroll (int): Unroll factor for the input channels (C_in).
        K_unroll (int): Unroll factor for the output channels (C_out).

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """

    def pad_input_node(input, pad, scale, scale_pad):
        pad_quantize_mx_input = (
            scale is not None and
            input.target == operator.getitem and
            scale.target == operator.getitem and
            input.args[0] == scale.args[0] and
            input.args[0].target == torch.ops.quantized_ops.quantize_mx.default
        )

        node_to_pad = input.args[0].args[0] if pad_quantize_mx_input else input

        skip_padding = False
        if node_to_pad.target == torch.ops.aten.slice.Tensor:
            arg = node_to_pad.args[0]
            skip_padding = slicing_and_padding_cancel_out(
                arg.value.shape, *node_to_pad.args[1:], pad
            )

        if skip_padding:
            new_input = node_to_pad.args[0]
        else:
            with model.graph.inserting_after(node_to_pad):
                new_input = model.graph.call_function(
                    torch.ops.aten.pad.default, (node_to_pad, pad),
                )

            propagate_shape(new_input)
            new_input.meta["dtype"] = node_to_pad.meta.get("dtype")

        if pad_quantize_mx_input:
            input.args[0].replace_input_with(node_to_pad, new_input)
            propagate_shape(input.args[0])
            propagate_shape(input)
            propagate_shape(scale)
        else:
            node.replace_input_with(node_to_pad, new_input)

            if scale is not None and any(x for x in scale_pad):
                with model.graph.inserting_before(node):
                    padded_scale = model.graph.call_function(
                        torch.ops.aten.pad.default, (scale, scale_pad),
                    )

                node.replace_input_with(scale, padded_scale)

                propagate_shape(padded_scale)
                padded_scale.meta["dtype"] = scale.meta.get("dtype")

    for node in list(model.graph.nodes):
        if not _is_gemm_op(node) or is_depthwise_conv(node):
            continue

        input = node.args[0]
        C_in = input.shape[1] if is_conv2d(node) else input.shape[-1]

        # Skip CNN first layer and fully-connected layer
        if is_conv2d(node) and C_in == 3 or math.prod(input.shape[:-1]) == 1:
            continue

        pad_C = (C_unroll - (C_in % C_unroll)) % C_unroll

        # Pad input along C dimension
        if pad_C:
            input_pad = [0, 0, 0, 0, 0, pad_C] if is_conv2d(node) else [0, pad_C]
            bs = node.kwargs.get("block_size", 1)
            input_scale = node.kwargs.get("input_scale")
            input_scale_pad = (
                [0, 0, 0, 0, 0, int(pad_C / bs)] if is_conv2d(node) else [0, int(pad_C / bs)]
            )
            pad_input_node(input, input_pad, input_scale, input_scale_pad)

        weight = node.args[1]
        C_in = weight.shape[-2] if _is_matmul(node) else weight.shape[1]
        C_out = weight.shape[-1] if _is_matmul(node) else weight.shape[0]

        pad_C = (C_unroll - (C_in % C_unroll)) % C_unroll
        pad_K = (K_unroll - (C_out % K_unroll)) % K_unroll

        # Pad weight along K and C
        if pad_C or pad_K:
            bs = node.kwargs.get("block_size", 1)
            weight_scale = node.kwargs.get("weight_scale")

            if is_conv2d(node):
                weight_pad = [0, 0, 0, 0, 0, pad_C, 0, pad_K]
                weight_scale_pad = [0, 0, 0, 0, 0, int(pad_C / bs), 0, pad_K]
            elif _is_matmul(node):
                weight_pad = [0, pad_K, 0, pad_C]
                weight_scale_pad = [0, pad_K, 0, int(pad_C / bs)]
            else:
                weight_pad = [0, pad_C, 0, pad_K]
                weight_scale_pad = [0, int(pad_C / bs), 0, pad_K]

            if weight.op == "get_attr":
                logger.debug(f"Pad {weight} with {weight_pad}")
                param = get_parameter_or_buffer(model, weight.target)
                param.data = F.pad(param.data, weight_pad)
                weight.value = param.data

                if weight_scale is not None:
                    scale_param = get_parameter_or_buffer(model, weight_scale.target)
                    scale_param.data = F.pad(scale_param.data, weight_scale_pad)
                    weight_scale.value = scale_param.data
            else:
                pad_input_node(weight, weight_pad, weight_scale, weight_scale_pad)

            if len(node.args) > 2 and node.args[2] is not None and pad_K:
                bias = node.args[2]
                bias_param = get_parameter_or_buffer(model, bias.target)
                bias_param.data = F.pad(bias_param.data, [0, pad_K])
                bias.value = bias_param.data

        propagate_shape(node)

        if pad_K:
            slice_dim = 1 if is_conv2d(node) else -1
            with model.graph.inserting_after(node):
                slice_node = model.graph.call_function(
                    torch.ops.aten.slice.Tensor, (node, slice_dim, 0, C_out),
                )

            node.replace_all_uses_with(slice_node)
            slice_node.replace_input_with(slice_node, node)

            propagate_shape(slice_node)
            slice_node.meta["dtype"] = node.meta.get("dtype")

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def pad_vector_ops_to_hardware_unroll_size(
    model: torch.fx.GraphModule,
    K_unroll,
) -> torch.fx.GraphModule:
    """
    Pad inputs to vector operations to multiples of the hardware unroll size.
    Only support softmax operation for now.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        K_unroll (int): Unroll factor for the output channels.

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.softmax.int:
            continue

        input = node.args[0]
        reduction_dim = input.shape[-1]

        pad_K = (K_unroll - (reduction_dim % K_unroll)) % K_unroll

        if not pad_K:
            continue

        with model.graph.inserting_after(input):
            new_input = model.graph.call_function(
                torch.ops.aten.pad.default,
                (input, [0, pad_K], "constant", float("-inf")),
            )

        propagate_shape(new_input)
        new_input.meta["dtype"] = input.meta.get("dtype")
        node.replace_input_with(input, new_input)

        propagate_shape(node)

        with model.graph.inserting_after(node):
            slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor, (node, -1, 0, reduction_dim),
            )

        node.replace_all_uses_with(slice_node)
        slice_node.replace_input_with(slice_node, node)

        propagate_shape(slice_node)
        slice_node.meta["dtype"] = node.meta.get("dtype")

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def pad_vit_embeddings_output(
    model,
    embeddings,
    example_inputs,
    dynamic_shapes=None,
    array_size=32
):
    original_graph = model.graph

    pattern = get_aten_graph_module(
        embeddings, example_inputs, dynamic_shapes=dynamic_shapes
    )
    pattern_graph = pattern.graph

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)
    print(f"Found {len(_matches)} matches")

    if not _matches:
        return model

    vit_embed_out = _matches[0].returning_nodes[0]\

    orig_dim = vit_embed_out.meta["val"].shape[-2]
    pad = (array_size - (orig_dim % array_size)) % array_size
    print(f"Padding {vit_embed_out} with {pad}")

    with model.graph.inserting_after(vit_embed_out):
        pad_node = model.graph.call_function(
            torch.ops.aten.pad.default,
            (vit_embed_out, [0, 0, 0, pad]),
        )

    propagate_shape(pad_node)
    pad_node.meta["val"] = pad_node.value

    for user in list(vit_embed_out.users):
        if id(user) != id(pad_node):
            user.replace_input_with(vit_embed_out, pad_node)

    for node in model.graph.nodes:
        if node.target in [torch.ops.aten.view.default, torch.ops.aten.reshape.default]:
            new_size = [x if x != orig_dim else x + pad for x in node.args[1]]
            node.args = (node.args[0], new_size)

    model.graph.lint()
    model.recompile()
    return model


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")


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


def is_conv2d(node: Node) -> bool:
    return node.op == "call_function" and node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d_mx.default
    ]


def is_depthwise_conv(node: Node) -> bool:
    return (
        node.target in [
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_ops.conv2d_mx.default
        ] and
        len(node.args) == 7 and
        node.args[6] != 1
    )


def dfs_collect_connected_conv2d_chain(model: GraphModule, start: Node, visited: Set[Node]) -> Set[Node]:
    """DFS downstream traversal to find conv2d nodes connected through elementwise ops."""
    stack = [start]
    chain = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        chain.add(node)

        for user in list(node.users.keys()) + node.all_input_nodes:
            if is_conv2d(user) or _is_elementwise_op(user) or user.target in [
                torch.ops.aten.adaptive_avg_pool2d.default,
                torch.ops.aten.max_pool2d.default,
                torch.ops.aten.slice.Tensor,
                torch.ops.aten.pad.default,
                torch.ops.quantized_ops.quantize_mx.default,
                operator.getitem,
            ]:
                stack.append(user)

    order = {n: i for i, n in enumerate(model.graph.nodes)}
    chain = sorted(chain, key=lambda n: order[n])
    return chain


def remap_pad_after_permute(pad: Tuple[int, ...], order: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    """
    Remap padding after permuting a tensor.

    Args:
        pad: Original pad tuple as in torch.nn.functional.pad (starts from last dim).
        order: Permutation order of dimensions.
        ndim: Number of dimensions in the original tensor.

    Returns:
        Tuple[int, ...]: New pad tuple corresponding to permuted tensor.
    """
    # number of padded dimensions
    k = len(pad) // 2
    assert k <= ndim, "Pad dimensions exceed tensor dimensions"

    # original padded dims (from last to first)
    original_padded_dims = list(range(ndim - k, ndim))

    dim_to_new_index = {d: order.index(d) for d in range(ndim)}

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


def transpose_conv2d_inputs_and_weights(model: GraphModule):
    graph = model.graph
    visited: Set[Node] = set()

    torch.nn.functional.conv2d = conv2d_transposed

    def get_path_to_target(node: torch.fx.Node, targets):
        if not isinstance(targets, (list, tuple)):
            targets = [targets]

        for user in node.users:
            if user.target in targets:
                return [node, user]

            if (
                _is_nop(user) or _is_indexing_or_concatenation_op(user)
                or user.target in [
                    torch.ops.quantized_ops.quantize.default,
                    torch.ops.aten.pad.default,
                ]
            ):
                path = get_path_to_target(user, targets)
                if path is not None:
                    return [node] + path
        return None

    for node in graph.nodes:
        is_pool = node.target in [
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.aten.max_pool2d.default,
        ]

        if node in visited or (not is_conv2d(node) and not is_pool):
            continue

        swapped_nodes = []
        node_dim_order = {}
        conv_chain = dfs_collect_connected_conv2d_chain(model, node, visited)

        for node_in_chain in conv_chain:
            for arg in node_in_chain.all_input_nodes:
                if arg in conv_chain or arg in swapped_nodes:
                    if arg.all_input_nodes:
                        node_dim_order[arg] = node_dim_order.get(arg.all_input_nodes[0])
                    continue

                path = get_path_to_target(arg, [
                    torch.ops.aten.conv2d.default,
                    torch.ops.quantized_ops.conv2d_mx.default,
                    torch.ops.quantized_ops.conv2d.default,
                ])

                # Permute weight and weight scale
                if arg.op == "get_attr" and path is not None:
                    input_node = path[-2]
                    conv2d_node = path[-1]

                    if is_depthwise_conv(conv2d_node):
                        continue

                    if input_node == conv2d_node.args[1]:
                        logger.debug(f"Permuting weight node {arg}")
                        param = get_parameter_or_buffer(model, arg.target)
                        param.data = param.data.permute(2, 3, 1, 0)
                        node_dim_order[arg] = (2, 3, 1, 0)

                    if input_node == conv2d_node.kwargs.get("weight_scale"):
                        logger.debug(f"Permuting weight scale node {arg}")
                        scale = get_parameter_or_buffer(model, arg.target)
                        scale.data = scale.data.permute(2, 3, 1, 0)
                        node_dim_order[arg] = (2, 3, 1, 0)

                if arg.op == "get_attr":
                    continue

                is_weight_node = path is not None and id(path[-2]) == id(path[-1].args[1])
                dims = (2, 3, 1, 0) if is_weight_node else (0, 2, 3, 1)

                logger.debug(f"Insert permute before {arg} with dims {dims}")
                with graph.inserting_after(arg):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default, (arg, dims),
                    )
                permute_node.meta["dtype"] = arg.meta.get("dtype")
                node_in_chain.replace_input_with(arg, permute_node)

                node_dim_order[permute_node] = dims

            for user in list(node_in_chain.users.keys()):
                if user in conv_chain or user in swapped_nodes:
                    continue
                logger.debug(f"Insert permute after {user} with dims (0, 3, 1, 2)")
                with graph.inserting_before(user):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default, (node_in_chain, (0, 3, 1, 2)),
                    )
                permute_node.meta["dtype"] = node_in_chain.meta.get("dtype")
                user.replace_input_with(node_in_chain, permute_node)

            if node_in_chain.target == torch.ops.aten.slice.Tensor:
                order = node_dim_order[node_in_chain.args[0]]
                args = tuple(node_in_chain.args)
                node_in_chain.args = args[:1] + (order.index(args[1]),) + args[2:]

            if node_in_chain.target == torch.ops.aten.pad.default:
                pad = remap_pad_after_permute(
                    node_in_chain.args[1],
                    node_dim_order[node_in_chain.args[0]],
                    node_in_chain.value.ndim,
                )
                node_in_chain.args = (node_in_chain.args[0], pad) + node_in_chain.args[2:]

            if node_in_chain.target == torch.ops.quantized_ops.quantize_mx.default:
                node_in_chain.args = node_in_chain.args[:1] + (-1,) + node_in_chain.args[2:]

            if node_in_chain.target == torch.ops.aten.conv2d.default:
                with graph.inserting_before(node_in_chain):
                    conv_node = graph.call_function(
                        torch.ops.quantized_ops.conv2d.default, args=node_in_chain.args
                    )

                logger.debug(f"Replace conv2d node {node_in_chain} with {conv_node}")

                conv_node.meta = node_in_chain.meta
                node_in_chain.replace_all_uses_with(conv_node)
                graph.erase_node(node_in_chain)

                swapped_nodes.append(conv_node)
                node_in_chain = conv_node

            node_dim_order[node_in_chain] = node_dim_order[node_in_chain.all_input_nodes[0]]

    graph.lint()
    model.recompile()
    return model


def eliminate_reshape_with_no_effect(model: GraphModule):
    deleted_nodes = set()
    for node in list(model.graph.nodes):
        if not _is_reshape_op(node) or node in deleted_nodes:
            continue

        curr_node = node
        input_node = node.all_input_nodes[0]

        group = []
        while len(curr_node.users) == 1 and (_is_reshape_op(curr_node) or _is_nop(curr_node)):
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


def linear_transposed(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None
) -> torch.Tensor:
    return torch.ops.aten.linear.default(input, weight.T, bias)


def linear_transposed_without_fc(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None
) -> torch.Tensor:
    if math.prod(input.shape[:-1]) == 1:
        return torch.ops.aten.linear.default(input, weight, bias)
    return torch.ops.aten.linear.default(input, weight.T, bias)


def transpose_linear_weights(model: GraphModule, transpose_fc: bool = False):
    """
    Transpose the weights of linear layers in the given FX graph module.

    Args:
        model (GraphModule): The FX graph module to transform.
        transpose_fc (bool): If True, transpose the weights of fully connected layers.

    Returns:
        GraphModule: The transformed FX graph module with transposed weights.
    """
    torch.nn.functional.linear = (
        linear_transposed if transpose_fc else linear_transposed_without_fc
    )

    for node in model.graph.nodes: 
        if node.target not in [
            torch.ops.aten.linear.default,
            torch.ops.quantized_ops.linear_mx.default,
        ]:
            continue

        input_node = node.args[0]
        input_shape = input_node.value.shape

        # TODO: handle torch.matmul second operand when not transposing FC
        if not transpose_fc and math.prod(input_shape[:-1]) == 1:
            continue

        node.meta["transposed"] = True

        weight_node = node.args[1]
        weight = get_parameter_or_buffer(model, weight_node.target)
        weight.data = weight.data.T

        for user in list(weight_node.users):
            if user.target == torch.ops.quantized_ops.spmm_csr.default:
                user.kwargs = {
                    **user.kwargs,
                    "weight_transposed": True
                }

        if (tiled_shapes := node.meta.get("tiled_shapes")) is not None:
            shape = tiled_shapes["weight"]
            tiled_shapes["weight"] = (shape[1], shape[0])

            if "weight_scale" in tiled_shapes:
                scale_shape = tiled_shapes["weight_scale"]
                tiled_shapes["weight_scale"] = (scale_shape[1], scale_shape[0])

        if node.target == torch.ops.quantized_ops.linear_mx.default:
            scale_node = node.kwargs.get("weight_scale")
            scale = get_parameter_or_buffer(model, scale_node.target)
            scale.data = scale.data.T
            continue

        with model.graph.inserting_before(node):
            linear_node = model.graph.call_function(
                torch.ops.quantized_ops.linear.default, args=node.args
            )

        linear_node.meta = node.meta
        linear_node.meta["transposed"] = True

        node.replace_all_uses_with(linear_node)
        model.graph.erase_node(node)

    model.graph.lint()
    model.recompile()
    return model


def replace_target(model, decomposition_table):
    graph = model.graph
    for node in graph.nodes:
        if (target := decomposition_table.get(node.target)) is None:
            continue

        with graph.inserting_after(node):
            new_node = graph.call_function(target, node.args)
        propagate_shape(new_node)
        new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()


def replace_conv2d_with_im2col(model: GraphModule, unroll=16):
    """
    Replace Conv2d operations with In2col operations in the given FX graph module.

    Args:
        model (GraphModule): The FX graph module to transform.

    Returns:
        GraphModule: The transformed FX graph module.
    """
    for node in model.graph.nodes:
        if node.target != torch.ops.aten.conv2d.default:
            continue

        input_node = node.args[0]
        weight_node = node.args[1]

        if input_node.value.shape[1] >= unroll:
            continue

        N, C_in, H_in, W_in = input_node.value.shape
        C_out, _, kH, kW = weight_node.value.shape

        args = [None, None, None, 1, 0, 1, 1]
        args[:len(node.args)] = node.args

        stride = _pair(args[3])
        padding = _pair(args[4])
        dilation = _pair(args[5])

        H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

        param = model.get_parameter(weight_node.target)
        param.data = param.data.reshape(C_out, -1)
        weight_node.value = param.data
        weight_node.shape = param.data.shape

        bias_node = args[2]
        if bias_node is not None:
            param = model.get_parameter(bias_node.target)
            param.data = param.data.reshape(C_out, 1)
            bias_node.value = param.data

        with model.graph.inserting_before(node):
            in2col_node = model.graph.call_function(
                torch.ops.aten.im2col.default,
                (input_node, (kH, kW), dilation, padding, stride),
            )
            matmul_node = model.graph.call_function(
                torch.ops.aten.matmul.default,
                (weight_node, in2col_node),
            )
            add_node = model.graph.call_function(
                torch.ops.aten.add.Tensor,
                (matmul_node, bias_node),
            )
            reshape_node = model.graph.call_function(
                torch.ops.aten.reshape.default,
                (add_node, (N, C_out, H_out, W_out)),
            )

        in2col_node.meta = input_node.meta
        matmul_node.meta = node.meta

        propagate_shape(in2col_node)
        propagate_shape(matmul_node)
        propagate_shape(add_node)
        propagate_shape(reshape_node)

        in2col_node.meta["source_fn_stack"] = [(in2col_node.name, in2col_node.target)]
        matmul_node.meta["source_fn_stack"] = [(matmul_node.name, torch.matmul)]
        add_node.meta["source_fn_stack"] = [(add_node.name, "add")]
        reshape_node.meta["source_fn_stack"] = [(reshape_node.name, "reshape")]

        node.replace_all_uses_with(reshape_node)
        model.graph.erase_node(node)

    model.graph.lint()
    model.recompile()
    return model


def extract_input_preprocessor(model: GraphModule):
    """
    Extract the input preprocessor from the model.

    Args:
        model (GraphModule): The FX graph module to transform.

    Returns:
        GraphModule: The transformed FX graph module with the input preprocessor extracted.
    """
    placeholder = next(iter(n for n in model.graph.nodes if n.op == "placeholder"))
    preprocess_nodes = [placeholder]

    user = next(iter(placeholder.users))

    while _is_nop(user) or user.target in [
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.im2col.default,
        torch.ops.quantized_ops.quantize.default,
    ]:
        preprocess_nodes.extend(
            n for n in user.all_input_nodes if n not in preprocess_nodes
        )
        preprocess_nodes.append(user)
        user = next(iter(user.users))

    m = torch.nn.Module()

    new_graph = torch.fx.Graph()
    value_remap = {}
    for node in preprocess_nodes:
        if node.op == 'placeholder':
            value_remap[node] = new_graph.placeholder(node.name)
        else:
            value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

            if node.op == "get_attr":
                param = get_parameter_or_buffer(model, node.target)
                m.register_buffer(node.target, param)
    new_graph.output(value_remap[preprocess_nodes[-1]])
    new_graph.lint()
    new_graph.print_tabular()

    with model.graph.inserting_before(placeholder):
        new_placeholder = model.graph.placeholder(f"{placeholder.name}_preprocess")
    preprocess_nodes[-1].replace_all_uses_with(new_placeholder)

    new_placeholder.meta["dtype"] =  preprocess_nodes[-1].meta.get("dtype")

    model.graph.lint()
    model.graph.eliminate_dead_code()
    # Placeholder node needs to be manually erased
    model.graph.erase_node(placeholder)
    model.recompile()
    return model, GraphModule(m, new_graph)


def rewrite_fx_graph(model: torch.fx.GraphModule, fn: Callable):
    """
    Transforms a given PyTorch FX GraphModule by identifying and replacing
    nodes that match a user-defined match_and_rewrite with alternative implementations.

    Args:
        model (torch.fx.GraphModule): The input FX GraphModule to be transformed.
        fn (Callable): A function that takes three arguments:
            - sources: The underlying function, module, or primitive operation
              responsible for a given FX node (from node.meta["source_fn_stack"]).
            - example_args (Tuple): A tuple of example arguments for the node,
              extracted from node metadata.
            - example_kwargs (Dict): A dictionary of example keyword arguments for the node.

            The `match_and_rewrite` function should return:
                - A `torch.nn.Module` or callable implementing an equivalent
                  or decomposed version of the operation if a match is found.
                - `None` otherwise.

    Returns:
        torch.fx.GraphModule: The transformed GraphModule with selected nodes
        replaced by decomposed modules returned by `match_and_rewrite`.

    Notes:
        - Each matched node is replaced using `export_model` with the returned
          module from `match_and_rewrite`.
        - The original node is erased from the graph after replacement.
        - The transformed graph is cleaned up via linting, dead code elimination,
          and recompilation.

    Example:
        >>> def match_and_rewrite(source_fn, args, kwargs):
        ...     if source_fn not in [torch.nn.Conv2d, torch.nn.functional.conv2d]:
        ...         return None
        ...     # Replace with a no-op or alternative module
        ...     class Identity(nn.Module):
        ...         def forward(self, x): return x
        ...     return Identity
        >>> transformed = rewrite_fx_graph(fx_model, match_and_rewrite)
    """
    for node in list(model.graph.nodes):
        if node.op != "call_function":
            continue

        if (source_fn_st := node.meta.get("source_fn_stack")) is None:
            continue

        source_fn = source_fn_st[-1][1]

        def get_value(n: Node):
            if "val" in n.meta:
                return n.meta["val"]
            return getattr(n, "value", None)

        example_args = map_arg(node.args, get_value)
        example_kwargs = map_arg(node.kwargs, get_value)

        if (cls := fn(source_fn, example_args, example_kwargs)) is None:
            continue

        new_args = map_arg(tuple(node.all_input_nodes), get_value)
        gm = export_model(cls(), new_args, example_kwargs)

        # PyTorch PT2E expect nodes to have no kwargs in the exported graph.
        # Clone has a memory_format kwarg, zeros_like has a pin_memory kwarg, and
        # gelu has a has an approximate kwarg that persist in exported graph.
        # This is just a work around for these.
        for n in list(gm.graph.nodes):
            if n.target == torch.ops.aten.zeros.default:
                n.kwargs = {}

        replace_node_with_graph_module(model, gm, node)

        model.graph.erase_node(node)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def calculate_gemm_node_size(node, x_factor, c_factor, k_factor):
    def node_mem(n, div_factors):
        return get_node_bytes(n) * n.value.numel() / div_factors

    total_bytes = 0
    input_node, weight_node = node.args[0], node.args[1]

    # Input, weight, and output memory
    total_bytes += node_mem(input_node, x_factor * c_factor)
    total_bytes += node_mem(weight_node, c_factor * k_factor)
    total_bytes += node_mem(node, x_factor * k_factor)

    # Bias if present
    if not _is_matmul(node) and len(node.args) > 2:
        total_bytes += node_mem(node.args[2], k_factor)

    # Optional scale factors
    input_scale_node = node.kwargs.get("input_scale")
    if input_scale_node is not None:
        total_bytes += node_mem(input_scale_node, x_factor * c_factor)

    weight_scale_node = node.kwargs.get("weight_scale")
    if weight_scale_node is not None:
        total_bytes += node_mem(weight_scale_node, c_factor * k_factor)

    return total_bytes


def choose_tile(node, X, C, K, max_mem_bytes, unroll_dims):
    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    def snap(x, unroll):
        return int((x // unroll) * unroll)

    X_tile = X
    C_tile = snap(C, unroll_dims[0])
    K_tile = snap(K, unroll_dims[1])

    # Shrink output channel dim first, then reduction dim, then input dim
    while True:
        total_size = calculate_gemm_node_size(
            node,
            X // X_tile,
            C // C_tile,
            K // K_tile
        )

        if total_size <= max_mem_bytes:
            break

        if K_tile > unroll_dims[1]:
            K_tile = snap(K_tile / 2, unroll_dims[1])
        elif X_tile > 128:
            X_tile = int(X_tile / 2)
        elif C_tile > unroll_dims[0]:
            C_tile = snap(C_tile / 2, unroll_dims[0])
        else:
            raise ValueError(f"Cannot tile X={X}, C={C}, K={K} to fit cache size {max_mem_bytes}.")

    return X_tile, C_tile, K_tile


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
            model, graph, node.target + "_", sliced_data
        )
        tiled_node.value = sliced_data
        tiled_node.meta["dtype"] = node.meta.get("dtype")
    else:
        tiled_node = graph.call_function(
            torch.ops.aten.slice.Tensor, (node, dim, start, end),
        )
        propagate_shape(tiled_node)
        tiled_node.meta["dtype"] = node.meta.get("dtype")
    return tiled_node


def run_matrix_op_l2_tiling(model, unroll, cache_size=DEFAULT_CACHE_SIZE):
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
        if node.target not in [
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
            torch.ops.quantized_ops.linear_mx.default,
            torch.ops.quantized_ops.matmul_mx.default,
        ]:
            continue

        input_node = node.args[0]
        weight_node = node.args[1]
        input_scale_node = node.kwargs.get("input_scale")
        weight_scale_node = node.kwargs.get("weight_scale")
        block_size = node.kwargs.get("block_size", 1)

        input_value, weight_value = input_node.value, weight_node.value

        is_matmul = node.target in [
            torch.ops.aten.matmul.default, torch.ops.quantized_ops.matmul_mx.default
        ]
        if is_matmul:
            C = int(weight_value.shape[0])
            K = int(weight_value.shape[1])
            reduction_dim = -2
            weight_key = "other"
        else:
            C = int(weight_value.shape[1])
            K = int(weight_value.shape[0])
            reduction_dim = -1
            weight_key = "weight"

        X = int(input_value.numel() / C)

        total_size = calculate_gemm_node_size(node, 1, 1, 1)

        if total_size <= cache_size:
            logger.info(f"{node}: X={X}, C={C}, K={K}, total_size={total_size} fits in cache, no tiling needed.")
            continue

        X_tile, C_tile, K_tile = choose_tile(
            node, X, C, K, cache_size, unroll,
        )

        num_x_tiles = X // X_tile
        num_k_tiles = K // K_tile
        num_c_tiles = C // C_tile

        logger.info(f"{node}: X={X}, C={C}, K={K} -> X_tile={X_tile}, C_tile={C_tile}, K_tile={K_tile}")

        if num_c_tiles == 1:
            node.meta["tiled_shapes"] = {
                "input": (X_tile, C),
                weight_key: (C, K_tile) if is_matmul else (K_tile, C),
                "bias": (K_tile,),
                "input_scale": (X_tile, C // block_size),
                "weight_scale": (C // block_size, K_tile) if is_matmul else (K_tile, C // block_size),
                "output": (X_tile, K_tile),
            }
            node.meta["l2_tiling"] = (num_x_tiles, num_k_tiles)
            continue

        if (source_fn_st := node.meta.get("source_fn_stack")) is not None:
            source_fn = source_fn_st[-1][1]
        else:
            source_fn = node.target

        last_output = None
        for c in range(0, C, C_tile):
            c_end = min(c + C_tile, C)
            scale_c, scale_c_end = int(c / block_size), int(c_end / block_size)

            with graph.inserting_before(node):
                tiled_input = slice_tensor(input_node, -1, c, c_end, model)
                tiled_weight = slice_tensor(weight_node, reduction_dim, c, c_end, model)

                if input_scale_node is not None:
                    tiled_input_scale = slice_tensor(
                        input_scale_node, -1, scale_c, scale_c_end, model
                    )
                else:
                    tiled_input_scale = None

                if weight_scale_node is not None:
                    tiled_weight_scale = slice_tensor(
                        weight_scale_node, reduction_dim, scale_c, scale_c_end, model
                    )
                else:
                    tiled_weight_scale = None

                if input_scale_node is not None or weight_scale_node is not None:
                    tiled_gemm = graph.call_function(
                        node.target,
                        (tiled_input, tiled_weight) + node.args[2:],
                        {
                            **node.kwargs,
                            "input_scale": tiled_input_scale,
                            "weight_scale": tiled_weight_scale,
                        },
                    )
                else:
                    tiled_gemm = graph.call_function(
                        node.target, (tiled_input, tiled_weight) + node.args[2:],
                    )

                propagate_shape(tiled_gemm)
                tiled_gemm.meta["dtype"] = node.meta.get("dtype")
                tiled_gemm.meta["source_fn_stack"] = [(tiled_gemm.name, source_fn)]

                if last_output is not None:
                    last_output = graph.call_function(
                        torch.ops.aten.add.Tensor, (last_output, tiled_gemm),
                    )
                    last_output.meta["source_fn_stack"] = [(last_output.name, last_output.target)]
                    propagate_shape(last_output)
                else:
                    last_output = tiled_gemm

            tiled_gemm.meta["tiled_shapes"] = {
                "input": (X_tile, C_tile),
                weight_key: (C_tile, K_tile) if is_matmul else (K_tile, C_tile),
                "bias": (K_tile,),
                "input_scale": (X_tile, C_tile // block_size),
                "weight_scale": (C_tile // block_size, K_tile) if is_matmul else (K_tile, C_tile // block_size),
                "output": (X_tile, K_tile),
            }
            tiled_gemm.meta["l2_tiling"] = (num_x_tiles, num_k_tiles)

        node.replace_all_uses_with(last_output)
        graph.erase_node(node)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def get_tiled_shapes(input_shape, fix_last_dim=False, last_dim=-1, reverse=False, min_sizes=None):
    """
    Yields tile shapes by progressively reducing from outermost to innermost (or reverse).
    Once a dimension is reduced to 1, it stays fixed. Last dim can be fixed optionally.
    
    Args:
        input_shape (tuple): The original shape.
        fix_last_dim (bool): Whether to keep the last_dim fixed.
        last_dim (int): Index of the last dim to fix (can be negative).
        reverse (bool): If True, traverse dimensions from innermost to outermost.
        min_sizes (tuple or list): Minimum size allowed for each dimension (default is 1).
    """
    def get_factors(n, min_size):
        return [i for i in range(n, min_size - 1, -1) if n % i == 0]

    dims = len(input_shape)
    last_dim = dims + last_dim if last_dim < 0 else last_dim
    stop = last_dim if fix_last_dim else dims

    # Directional order
    dim_order = list(range(stop))
    if reverse:
        dim_order = dim_order[::-1]

    # Apply default min sizes
    if min_sizes is None:
        min_sizes = [1] * dims
    else:
        min_sizes = list(min_sizes) + [1] * (dims - len(min_sizes))

    current = list(input_shape)
    yield tuple(current)

    for dim in dim_order:
        factors = get_factors(input_shape[dim], min_sizes[dim])
        for f in factors[1:]:  # skip full-size factor
            current[dim] = f
            yield tuple(current)
        current[dim] = max(min_sizes[dim], 1)  # lock at min size


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


def run_vector_op_l2_tiling(model, unroll, cache_size=DEFAULT_CACHE_SIZE):
    def get_reduce_factor(full_shape, tile_shape):
        return tuple(f // t for f, t in zip(full_shape, tile_shape))

    def get_reduced_shape(shape, reduce_factor):
        assert len(shape) == len(reduce_factor)
        return tuple(
            s // r if s > 1 else s for s, r in zip(shape, reduce_factor)
        )

    for node in list(model.graph.nodes):
        if not _is_elementwise_op(node) and node.target not in [
            torch.ops.aten.softmax.int,
            torch.ops.aten.layer_norm.default,
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
            torch.ops.quantized_ops.calculate_mx_qparam.default,
            torch.ops.quantized_ops.quantize_mx.default,
        ]:
            continue

        output_shape = (
            node.value.shape if isinstance(node.value, torch.Tensor)
            else node.value[1].shape
        )

        if node.target in [
            torch.ops.quantized_ops.calculate_mx_qparam.default,
            torch.ops.quantized_ops.quantize_mx.default,
        ]:
            last_dim = node.args[1]
        elif node.target == torch.ops.aten.transpose.int:
            last_dim = min(*node.args[1:])
        elif node.target == torch.ops.aten.permute.default:
            last_dim = next((i for i, d in enumerate(node.args[1]) if i != d), None)
        else:
            last_dim = -1

        for tiled_output_shape in get_tiled_shapes(output_shape, True, last_dim):
            reduce_factor = get_reduce_factor(output_shape, tiled_output_shape)
            tile_numel = math.prod(tiled_output_shape)
            total_tile_size = get_node_bytes(node) * tile_numel

            if node.target == torch.ops.aten.softmax.int:
                total_tile_size *= 3

            if isinstance(node.value, (tuple, list)):
                tiled_shapes = {
                    "output": [get_reduced_shape(t.shape, reduce_factor) for t in node.value]
                }
            else:
                tiled_shapes = {"output": tiled_output_shape}

            node_to_key = get_node_to_key(node)

            for n in node.all_input_nodes:
                if n.name.startswith("code"):
                    continue

                input_shape = list(n.value.shape)
                aligned_shape = [1] * (len(output_shape) - len(input_shape)) + input_shape
                tiled_shape = tuple(
                    s // r if s > 1 else s for s, r in zip(aligned_shape, reduce_factor)
                )
                total_tile_size += get_node_bytes(n) * math.prod(tiled_shape)

                tiled_shapes[node_to_key.get(n)] = tiled_shape[-len(input_shape):]

            if total_tile_size <= cache_size:
                if math.prod(reduce_factor) > 1:
                    logger.info(f"Tile {node} with shape {tiled_output_shape} (reduce factor={reduce_factor}")
                    node.meta["tiled_shapes"] = tiled_shapes
                    node.meta["l2_tiling"] = reduce_factor
                break
        else:
            logger.warning(f"Warning: No tile shape found to fit {node} into cache.")
