import collections
import copy
import itertools
import logging
import math
import operator
from itertools import repeat
from typing import Callable, List, Set, Tuple, Union, Optional, Dict

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .mapping import (
    duplicate_shared_nodes,
    get_parameter_or_buffer,
    get_tiled_input_shape,
    propagate_shape,
    replace_node_with_graph_module,
    get_node_bytes,
)
from .mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_gemm_op,
    is_indexing_or_concatenation_op,
    is_linear,
    is_matmul,
    is_nop,
    is_pooling,
    is_prunable_op,
    is_reshape_op,
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
    "remove_autocast_nodes",
    "replace_conv2d_with_im2col",
    "replace_target_with_vmap",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
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
    logger.info(f"Found {len(_matches)} matches")

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

        new_node.meta = output_node.meta
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
            values = torch.arange(2 ** 16, dtype=torch.int16).view(torch.bfloat16)
            code = node.target(values, *node.args[1:])

            with model.graph.inserting_before(node):
                get_attr_node = create_getattr_from_value(
                    model, model.graph, '_tensor_constant_', code
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


def remove_autocast_nodes(model: torch.fx.GraphModule):
    """
    Handle autocast HOP by replacing the autocast node with its wrapped module
    and directly calling the arguments.
    """
    graph = model.graph
    named_modules = dict(model.named_modules())
    for node in list(graph.nodes):
        if isinstance(node.target, torch._higher_order_ops.wrap.WrapWithAutocast):
            wrapped_func = node.args[4]
            mod = named_modules.get(wrapped_func.target, None)
            if mod is not None:
                with graph.inserting_before(node):
                    new_node = graph.call_module(
                        wrapped_func.target, tuple(node.args[5:])
                    )
                    node.replace_all_uses_with(new_node)
                    replace_node_with_graph_module(model, mod, new_node)
                graph.erase_node(node)
    graph.eliminate_dead_code()
    model.graph.lint()
    model.compile()


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
        if not is_gemm_op(node):
            continue

        input = node.args[0]
        C_in = input.shape[1] if is_conv2d(node) else input.shape[-1]

        # Skip CNN first layer with input channels equal to 3
        if is_conv2d(node) and C_in == 3:
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
        C_in = weight.shape[-2] if is_matmul(node) else weight.shape[1]
        C_out = weight.shape[-1] if is_matmul(node) else weight.shape[0]

        if is_depthwise_conv(node):
            C_in *= node.args[6]
            args = list(node.args)
            args[-1] += pad_C
            node.args = tuple(args)

        pad_C = (C_unroll - (C_in % C_unroll)) % C_unroll
        pad_K = (K_unroll - (C_out % K_unroll)) % K_unroll

        if is_depthwise_conv(node):
            pad_K = pad_C

        # Pad weight along K and C
        if pad_C or pad_K:
            bs = node.kwargs.get("block_size", 1)
            weight_scale = node.kwargs.get("weight_scale")

            if is_depthwise_conv(node):
                weight_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
                weight_scale_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
            elif is_conv2d(node):
                weight_pad = [0, 0, 0, 0, 0, pad_C, 0, pad_K]
                weight_scale_pad = [0, 0, 0, 0, 0, int(pad_C / bs), 0, pad_K]
            elif is_matmul(node):
                weight_pad = [0, pad_K, 0, pad_C]
                weight_scale_pad = [0, pad_K, 0, int(pad_C / bs)]
            else:
                weight_pad = [0, pad_C, 0, pad_K]
                weight_scale_pad = [0, int(pad_C / bs), 0, pad_K]

            if weight.op == "get_attr":
                logger.debug(f"Pad {weight} with {weight_pad}")
                param = get_parameter_or_buffer(model, weight.target)
                param.data = F.pad(param.data, weight_pad)
                propagate_shape(weight, model)

                if weight_scale is not None:
                    scale_param = get_parameter_or_buffer(model, weight_scale.target)
                    scale_param.data = F.pad(scale_param.data, weight_scale_pad)
                    propagate_shape(weight_scale, model)
            else:
                pad_input_node(weight, weight_pad, weight_scale, weight_scale_pad)

            if len(node.args) > 2 and node.args[2] is not None and pad_K:
                bias = node.args[2]
                bias_param = get_parameter_or_buffer(model, bias.target)
                bias_param.data = F.pad(bias_param.data, [0, pad_K])
                propagate_shape(bias, model)

        propagate_shape(node)

        def slice_output(output_node, slice_args):
            nodes_require_slice = []
            for user in list(output_node.users.keys()):
                if is_elementwise_op(user):
                    if len(user.all_input_nodes) == 1 or user.target in [
                        torch.ops.quantized_ops.dequantize.default,
                        torch.ops.quantized_ops.quantize.default,
                    ]:
                        propagate_shape(user)
                        slice_output(user, slice_args)
                        continue

                    # if all inputs are a matching slice, then strip the redundant slice.
                    if all(
                        n == output_node or (
                            n.target == torch.ops.aten.slice.Tensor and
                            n.args[1:] == slice_args
                        )
                        for n in user.all_input_nodes
                    ):
                        for n in user.all_input_nodes:
                            if n.target == torch.ops.aten.slice.Tensor:
                                user.replace_input_with(n, n.args[0])

                        propagate_shape(user)
                        slice_output(user, slice_args)
                        continue

                nodes_require_slice.append(user)

            if nodes_require_slice:
                with model.graph.inserting_after(output_node):
                    slice_node = model.graph.call_function(
                        torch.ops.aten.slice.Tensor, (output_node, *slice_args),
                    )

                for n in nodes_require_slice:
                    n.replace_input_with(output_node, slice_node)

                propagate_shape(slice_node)
                slice_node.meta["dtype"] = output_node.meta.get("dtype")

        if pad_K:
            slice_dim = 1 if is_conv2d(node) else -1
            slice_output(node, (slice_dim, 0, C_out))

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
    logger.info(f"Found {len(_matches)} matches")

    if not _matches:
        return model

    vit_embed_out = _matches[0].returning_nodes[0]\

    orig_dim = vit_embed_out.meta["val"].shape[-2]
    pad = (array_size - (orig_dim % array_size)) % array_size
    logger.info(f"Padding {vit_embed_out} with {pad}")

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
                is_conv2d(user) or
                is_pooling(user) or
                is_elementwise_op(user) or
                is_indexing_or_concatenation_op(user) or
                user.target in [
                    torch.ops.aten.pad.default,
                    torch.ops.quantized_ops.calculate_mx_qparam.default,
                    torch.ops.quantized_ops.quantize_mx.default,
                    operator.getitem,
                ]
            ):
                stack.append(user)

    order = {n: i for i, n in enumerate(model.graph.nodes)}
    nodes_in_graph = sorted(nodes_in_graph, key=lambda n: order[n])
    return nodes_in_graph


def remap_pad_after_permute(
    pad: Tuple[int, ...], order: Tuple[int, ...], ndim: int
) -> Tuple[int, ...]:
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


TRANSPOSED_OPERATORS = {
    torch.ops.aten.conv2d.default: torch.ops.quantized_ops.conv2d.default,
    torch.ops.aten.max_pool2d.default: torch.ops.quantized_ops.max_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d.default: torch.ops.quantized_ops.adaptive_avg_pool2d.default,
}


AXES_ARG_INDEX_MAP = {
    torch.ops.quantized_ops.calculate_mx_qparam.default: 1,
    torch.ops.quantized_ops.dequantize.default: 3,
    torch.ops.quantized_ops.quantize.default: 3,
    torch.ops.quantized_ops.quantize_mx.default: 2,
}


def transpose_conv2d_inputs_and_weights(model: GraphModule):
    graph = model.graph
    visited: Set[Node] = set()
    node_dim_order = {}

    torch.nn.functional.conv2d = conv2d_transposed

    def get_path_to_conv2d(node: torch.fx.Node):
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
                path = get_path_to_conv2d(user)
                if path is not None:
                    return [node] + path
        return None

    for node in list(graph.nodes):
        if node in visited or not (is_conv2d(node) or is_pooling(node)):
            continue

        conv2d_graph = extract_conv2d_graph(model, node, visited)
        handled = []

        for node_to_treat in conv2d_graph:
            for arg in node_to_treat.all_input_nodes:
                if arg in conv2d_graph or arg in handled:
                    continue

                path = get_path_to_conv2d(arg)

                # Permute weight and weight scale param
                if arg.op == "get_attr" and path is not None:
                    input_node = path[-2]
                    conv2d_node = path[-1]

                    # Skip depthwise conv weights
                    if is_depthwise_conv(conv2d_node):
                        continue

                    if input_node in (
                        conv2d_node.args[1], conv2d_node.kwargs.get("weight_scale")
                    ):
                        logger.debug(f"Permuting parameter {arg}")
                        param = get_parameter_or_buffer(model, arg.target)
                        param.data = param.data.permute(2, 3, 1, 0)
                        node_dim_order[arg] = (2, 3, 1, 0)

                    handled.append(arg)

                # Permute input tensor
                if arg.op != "get_attr" and len(arg.shape) == 4:
                    is_weight_node = path is not None and id(path[-2]) == id(path[-1].args[1])
                    dims = (2, 3, 1, 0) if is_weight_node else (0, 2, 3, 1)

                    logger.debug(f"Insert permute after {arg} with dims {dims}")
                    with graph.inserting_after(arg):
                        permute_node = graph.call_function(
                            torch.ops.aten.permute.default, (arg, dims),
                        )
                    permute_node.meta["dtype"] = arg.meta.get("dtype")
                    node_to_treat.replace_input_with(arg, permute_node)
                    node_dim_order[permute_node] = dims

            for user in list(node_to_treat.users.keys()):
                if user in conv2d_graph or user in handled:
                    continue
                logger.debug(f"Insert permute before {user} with dims (0, 3, 1, 2)")
                with graph.inserting_before(user):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default, (node_to_treat, (0, 3, 1, 2)),
                    )
                permute_node.meta["dtype"] = node_to_treat.meta.get("dtype")
                user.replace_input_with(node_to_treat, permute_node)

            if node_to_treat.target == torch.ops.aten.pad.default:
                pad = remap_pad_after_permute(
                    node_to_treat.args[1],
                    node_dim_order[node_to_treat.args[0]],
                    node_to_treat.value.ndim,
                )
                node_to_treat.args = (
                    node_to_treat.args[0], pad, *node_to_treat.args[2:]
                )

            if node_to_treat.target in AXES_ARG_INDEX_MAP:
                order = node_dim_order[node_to_treat.all_input_nodes[0]]
                args = tuple(node_to_treat.args)
                idx = AXES_ARG_INDEX_MAP[node_to_treat.target]
                axes = [a + max(order) + 1 if a < 0 else a for a in args[idx]]
                axes = tuple(order.index(a) for a in axes)
                node_to_treat.args = args[:idx] + (axes,) + args[idx + 1 :]

            if is_indexing_or_concatenation_op(node_to_treat):
                order = node_dim_order[node_to_treat.all_input_nodes[0]]
                args = tuple(node_to_treat.args)
                dims = args[1] + max(order) + 1 if args[1] < 0 else args[1]
                node_to_treat.args = args[:1] + (order.index(dims),) + args[2:]

            if is_reshape_op(node_to_treat):
                order = node_dim_order[node_to_treat.all_input_nodes[0]]
                args = tuple(node_to_treat.args)
                if node_to_treat.target == torch.ops.aten.transpose.int:
                    dims = (args[1], args[2])
                else:
                    dims = args[1]
                dims = [d + max(order) + 1 if d < 0 else d for d in dims]
                dims = tuple(order.index(d) for d in dims)
                node_to_treat.args = args[:1] + (order.index(dims),) + args[2:]

            if node_to_treat.target in TRANSPOSED_OPERATORS:
                with graph.inserting_before(node_to_treat):
                    new_node = graph.call_function(
                        TRANSPOSED_OPERATORS[node_to_treat.target],
                        node_to_treat.args,
                        node_to_treat.kwargs,
                    )
                logger.debug(f"Replace node {node_to_treat} with {new_node}")
                new_node.meta = node_to_treat.meta
                node_to_treat.replace_all_uses_with(new_node)
                graph.erase_node(node_to_treat)
                handled.append(new_node)
                node_to_treat = new_node

            if is_conv2d(node_to_treat):
                node_to_treat.meta["transposed"] = True

            tiled_shapes = node_to_treat.meta.get("tiled_shapes")
            if is_conv2d(node_to_treat) and tiled_shapes is not None:
                for key, arg in [
                    ("input", node_to_treat.args[0]),
                    ("weight", node_to_treat.args[1])
                ]:
                    order = node_dim_order[arg]
                    tiled_shapes[key] = tuple(tiled_shapes[key][i] for i in order)

                    scale_key = f"{key}_scale"
                    if scale_key in tiled_shapes:
                        tiled_shapes[scale_key] = tuple(
                            tiled_shapes[scale_key][i] for i in order
                        )

                tiled_shapes["output"] = tuple(
                    tiled_shapes["output"][i] for i in (0, 2, 3, 1)
                )

                tiling = node_to_treat.meta["l2_tiling"]
                node_to_treat.meta["l2_tiling"] = (1, 1, 1, tiling[0])

            node_dim_order[node_to_treat] = node_dim_order[
                node_to_treat.all_input_nodes[0]
            ]

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
        transpose_fc (bool): If True, transpose the weights of fully connected layers.

    Returns:
        GraphModule: The transformed FX graph module with transposed weights.
    """
    skip_fc = not transpose_fc
    torch.nn.functional.linear = make_linear_wrapper(transpose_weight, skip_fc)
    torch.matmul = make_matmul_wrapper(transpose_weight, skip_fc)

    transposed_nodes = {}

    for node in list(model.graph.nodes):
        if not is_gemm_op(node):
            continue

        input_node = node.args[0]
        input_shape = input_node.value.shape
        is_fc = math.prod(input_shape[:-1]) == 1

        weight_node = node.args[1]
        scale_node = node.kwargs.get("weight_scale")

        if is_linear(node):
            if (is_fc and not transpose_fc) or (not is_fc and not transpose_weight):
                continue

            weight = get_parameter_or_buffer(model, weight_node.target)
            weight.data = weight.data.T

            if scale_node is not None:
                scale = get_parameter_or_buffer(model, scale_node.target)
                scale.data = scale.data.T

            for user in list(weight_node.users):
                if user.target == torch.ops.quantized_ops.spmm_csr.default:
                    user.kwargs = {
                        **user.kwargs,
                        "weight_transposed": True
                    }

            node.meta["transposed"] = True

            if (tiled_shapes := node.meta.get("tiled_shapes")) is not None:
                shape = tiled_shapes["weight"]
                tiled_shapes["weight"] = (shape[1], shape[0])

                if "weight_scale" in tiled_shapes:
                    scale_shape = tiled_shapes["weight_scale"]
                    tiled_shapes["weight_scale"] = (scale_shape[1], scale_shape[0])

            if node.target == torch.ops.aten.linear.default:
                with model.graph.inserting_before(node):
                    linear_transposed = model.graph.call_function(
                        torch.ops.quantized_ops.linear.default, node.args
                    )
                node.replace_all_uses_with(linear_transposed)
                model.graph.erase_node(node)
                linear_transposed.meta = node.meta

        # Matmul is already transposed by default
        if is_matmul(node):
            if (is_fc and transpose_fc) or (not is_fc and transpose_weight):
                continue

            with model.graph.inserting_before(node):
                weight_transposed = model.graph.call_function(
                    torch.ops.aten.transpose.int, (weight_node, -2, -1)
                )
            weight_transposed.meta["dtype"] = weight_node.meta.get("dtype")
            propagate_shape(weight_transposed, model)

            if scale_node is not None:
                with model.graph.inserting_before(node):
                    scale_transposed = model.graph.call_function(
                        torch.ops.aten.transpose.int, (scale_node, -2, -1)
                    )
                scale_transposed.meta["dtype"] = scale_node.meta.get("dtype")
                propagate_shape(scale_transposed, model)

            if (tiled_shapes := node.meta.get("tiled_shapes")) is not None:
                shape = tiled_shapes["weight"]
                tiled_shapes["weight"] = (shape[1], shape[0])

                if "weight_scale" in tiled_shapes:
                    scale_shape = tiled_shapes["weight_scale"]
                    tiled_shapes["weight_scale"] = (scale_shape[1], scale_shape[0])

            node.meta["transposed"] = True

            if node.target == torch.ops.aten.matmul.default:
                with model.graph.inserting_before(node):
                    matmul_transposed = model.graph.call_function(
                        torch.ops.quantized_ops.matmul.default,
                        (input_node, weight_transposed),
                    )
                node.replace_all_uses_with(matmul_transposed)
                model.graph.erase_node(node)
                matmul_transposed.meta = node.meta
            else:
                node.args = (input_node, weight_transposed)
                node.kwargs = {**node.kwargs, "weight_scale": scale_transposed}

            node_order = {n: i for i, n in enumerate(model.graph.nodes)}
            path = find_upstream_matching_transpose(weight_transposed)
            sorted_path = sorted(path, key=lambda n: node_order[n])
            success = process_double_transpose_chain(model, sorted_path, transposed_nodes)
            if not success:
                move_transpose_before_dq(model, sorted_path, transposed_nodes)

            if scale_node is not None:
                scale_path = find_upstream_matching_transpose(scale_transposed)
                sorted_path = sorted(scale_path, key=lambda n: node_order[n])
                fold_transpose_into_constant(model, sorted_path, transposed_nodes)

    model.graph.lint()
    model.recompile()
    return model


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
        propagate_shape(weight_node, model)

        bias_node = args[2]
        if bias_node is not None:
            param = model.get_parameter(bias_node.target)
            param.data = param.data.reshape(C_out, 1)
            propagate_shape(bias_node, model)

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

    while is_nop(user) or user.target in [
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


def get_arg_or_kwarg(node, idx, key, default=None):
    if len(node.args) > idx:
        return node.args[idx]
    return node.kwargs.get(key, default)


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

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    N, K, Y, X = node.shape
    _, C, kH, kW = node.args[1].shape
    _, _, IX, IY = node.args[0].shape

    tile_y, tile_x, tile_k, tile_c = tiling

    tile_iy = (tile_y - 1) * stride[0] + (kH - 1) * dilation[0] + 1
    tile_ix = (tile_x - 1) * stride[1] + (kW - 1) * dilation[1] + 1

    tiled_shapes = {
        "input": (N, tile_c, tile_iy, tile_ix),
        "weight": (tile_k, tile_c, kH, kW),
        "bias": (tile_k,),
        "output": (N, tile_k, tile_y, tile_x),
        "input_scale": (N, tile_c // bs, tile_iy, tile_ix),
        "weight_scale": (tile_k, tile_c // bs, kH, kW),
    }
    tiling = (1, K // tile_k, 1, 1)

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
            for y in range(0, Y, tile_y):
                col_tiles = []
                for x in range(0, X, tile_x):
                    oh = min(tile_y, Y - y)
                    ow = min(tile_x, X - x)

                    acc = None

                    # Compute receptive field in input
                    y_in_start = y * self.stride[0] - self.padding[0]
                    y_in_end = (
                        y_in_start + (oh - 1) * self.stride[0]
                        + (kH - 1) * self.dilation[0] + 1
                    )
                    x_in_start = x * self.stride[1] - self.padding[1]
                    x_in_end = (
                        x_in_start + (ow - 1) * self.stride[1]
                        + (kW - 1) * self.dilation[1] + 1
                    )

                    y_in_start_clamped = max(y_in_start, 0)
                    x_in_start_clamped = max(x_in_start, 0)
                    y_in_end_clamped = min(y_in_end, IY)
                    x_in_end_clamped = min(x_in_end, IX)

                    # Pad input locally if receptive field goes outside
                    pad_top  = y_in_start_clamped - y_in_start
                    pad_left = x_in_start_clamped - x_in_start
                    pad_bottom = y_in_end - y_in_end_clamped
                    pad_right = x_in_end - x_in_end_clamped

                    # Iterate input channels
                    for c_start in range(0, C, tile_c):
                        c_end = min(c_start + tile_c, C)

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
                            (0, 0),  # padding already handled
                            self.dilation,
                            self.groups,
                        )

                        if input_scale is not None:
                            bs = self.block_size
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
                    "tiled_shapes": copy.deepcopy(tiled_shapes),
                    "l2_tiling": tiling,
                    "dtype": node.meta.get("dtype"),
                    "source_fn_stack": [(n.name, source_fn[1])],
                })

        if output[0].target == torch.ops.aten.cat.default:
            move_cat_after_fusable_ops(model, output[0], tiled_shapes["output"])

        model.graph.erase_node(node)
    else:
        node.meta["tiled_shapes"] = tiled_shapes
        node.meta["l2_tiling"] = tiling


def get_slice_args(full_shape, tiled_shape, tile_idx):
    # infer how many tiles exist per dimension
    tiling = tuple(f // t for f, t in zip(full_shape, tiled_shape))

    # unravel flat index into multi-d tile coordinate
    tile = []
    tmp = tile_idx
    for t in reversed(tiling):
        tile.append(tmp % t)
        tmp //= t
    tile = tuple(reversed(tile))

    slices = []
    for dim, (tile_size, tidx) in enumerate(zip(tiled_shape, tile)):
        if tiling[dim] == 1:
            continue  # no split in this dimension
        start = tidx * tile_size
        stop = start + tile_size
        slices.append((dim, start, stop))

    return slices


def create_new_chain(model, anchor, cat_node, fusable, tiled_shape, tile_idx=0):
    new_cat_inputs = []
    for idx, node_to_cat in enumerate(cat_node.args[0]):
        value_remap = {fusable[0]: node_to_cat}
        for n in fusable[1:]:
            for arg in n.all_input_nodes:
                if arg not in value_remap and arg.op != "get_attr":
                    shape = get_tiled_input_shape(tiled_shape, arg.shape)
                    # Only tile on Y and X dimensions
                    shape = arg.shape[:2] + shape[2:]
                    slice_args = get_slice_args(arg.shape, shape, tile_idx + idx)
                    sliced = arg
                    insert_point = arg.next
                    for dim, start, end in slice_args:
                        with model.graph.inserting_before(insert_point):
                            sliced = model.graph.call_function(
                                torch.ops.aten.slice.Tensor,
                                (sliced, dim, start, end),
                            )
                        propagate_shape(sliced, model)
                        sliced.meta["dtype"] = arg.meta.get("dtype")
                    value_remap[arg] = sliced
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
        new_cat_inputs.append(new_node)

    with model.graph.inserting_before(anchor):
        new_cat = model.graph.call_function(
            torch.ops.aten.cat.default,
            (new_cat_inputs,) + cat_node.args[1:],
        )
    propagate_shape(new_cat, model)
    new_cat.meta["dtype"] = new_cat_inputs[0].meta.get("dtype")
    return new_cat


def move_cat_after_fusable_ops(model, node, tiled_shape):
    nodes_to_fuse = [node]
    next_node = next(iter(node.users))
    while is_elementwise_op(next_node):
        nodes_to_fuse.append(next_node)
        if len(next_node.users) != 1:
            break
        next_node = next(iter(next_node.users))

    if len(nodes_to_fuse) == 1:
        return

    anchor = nodes_to_fuse[-1].next
    if node.all_input_nodes[0].target == torch.ops.aten.cat.default:
        x_tile_size = len(node.all_input_nodes[0].args[0])
        cat_inputs = []
        for i, n in enumerate(node.args[0]):
            new_cat = create_new_chain(
                model, anchor, n, nodes_to_fuse, tiled_shape, i * x_tile_size
            )
            cat_inputs.append(new_cat)

        with model.graph.inserting_before(anchor):
            new_cat = model.graph.call_function(
                torch.ops.aten.cat.default, (cat_inputs,) + node.args[1:]
            )
        propagate_shape(new_cat, model)
        new_cat.meta["dtype"] = cat_inputs[0].meta.get("dtype")
    else:
        new_cat = create_new_chain(
            model, anchor, node, nodes_to_fuse, tiled_shape
        )
    nodes_to_fuse[-1].replace_all_uses_with(new_cat)

    # Have to remove nodes manually as there are inplace ops
    for n in reversed(nodes_to_fuse):
        model.graph.erase_node(n)


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

            split_conv2d_node(model, node, (y_tiled, x_tiled, k_tiled, c_tiled))
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


def run_vector_op_l2_tiling(
    model, unroll, cache_size=DEFAULT_CACHE_SIZE, num_banks=None
):
    for node in list(model.graph.nodes):
        if not is_elementwise_op(node) and node.target not in [
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
        found_tiling = False

        bank_size = cache_size // num_banks if num_banks is not None else None

        def compute_size(n, shape):
            size = get_node_bytes(n) * math.prod(shape)
            if bank_size is not None:
                size = int(math.ceil(size / bank_size) * bank_size)
            # Double input size of softmax and layernorm for scratch space
            if n == node.args[0] and node.target in [
                torch.ops.aten.softmax.int, torch.ops.aten.layer_norm.default,
            ]:
                size *= 2
            return size

        for tiled_output_shape, tiling in get_valid_tiling(
            output_shape, last_dim=last_dim, min_sizes=(unroll,)
        ):
            tiled_shapes = {
                node_to_key.get(n): get_tiled_shape(tuple(n.shape), tiling)
                for n in input_nodes
            }

            total_size = sum(
                compute_size(n, tiled_shapes[node_to_key[n]])
                for n in input_nodes
            )

            if isinstance(node.value, (tuple, list)):
                output_shapes = [
                    get_tiled_shape(t.shape, tiling) for t in node.value
                ]
                tiled_shapes["output"] = output_shapes
                total_size += sum(
                    b * math.prod(s)
                    for b, s in zip(get_node_bytes(node), output_shapes)
                )
            else:
                tiled_shapes["output"] = tiled_output_shape
                total_size += compute_size(node, tiled_output_shape)

            if total_size <= cache_size:
                if math.prod(tiling) > 1:
                    logger.info(
                        f"Tile {node} with shape {tiled_output_shape} "
                        f"(reduce factor={tiling})"
                    )
                    node.meta["tiled_shapes"] = tiled_shapes
                    node.meta["l2_tiling"] = tiling
                found_tiling = True
                break

        if not found_tiling:
            logger.warning(f"No tiling found to fit {node} into cache.")
