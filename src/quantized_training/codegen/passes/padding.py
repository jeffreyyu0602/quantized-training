import logging
import operator
from typing import List

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

from .utils import get_arg_or_kwarg
from ..mapping import (
    get_parameter_or_buffer,
    propagate_shape,
)
from ..mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_gemm_op,
    is_matmul,
)
from ...pt2e_utils import get_aten_graph_module, fetch_attr
from ...quantize_pt2e import create_getattr_from_value

logger = logging.getLogger(__name__)

__all__ = [
    "pad_gemm_inputs_to_hardware_unroll_size",
    "pad_vector_ops_to_hardware_unroll_size",
    "pad_vit_embeddings_output",
]


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
    model: GraphModule,
    C_unroll,
    K_unroll,
) -> GraphModule:
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
                # quantize dequantize ops have to be per-tensor
                if user.target in [
                    torch.ops.quantized_ops.dequantize.default,
                    torch.ops.quantized_ops.quantize.default,
                ]:
                    bs = get_arg_or_kwarg(user, 4, "block_size")
                    if bs is None:
                        slice_output(user, slice_args)
                        continue

                if is_elementwise_op(user):
                    if len(user.all_input_nodes) == 1:
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


def pad_layer_norm_to_hardware_unroll_size(
    model: GraphModule,
    node: Node,
    unroll: int,
) -> GraphModule:
    input = node.args[0]
    normalize_shape = node.args[1]
    weight = node.args[2]
    bias = node.args[3] if len(node.args) > 3 else None

    orig_k = input.shape[-1]
    pad_k = (-orig_k) % unroll
    if pad_k == 0:
        return model

    logger.info(f"Padding layer_norm {node} last dimension with {pad_k}")

    def pad_param(attr_node: Node):
        if attr_node.op == "get_attr":
            param = fetch_attr(model, attr_node.target)
            new_param = F.pad(param, [0, pad_k])
            with model.graph.inserting_after(attr_node):
                new_attr = create_getattr_from_value(
                    model, model.graph, f"{attr_node.target}_padded", new_param
                )
        else:
            with model.graph.inserting_after(attr_node):
                new_attr = model.graph.call_function(
                    torch.ops.aten.pad.default, (attr_node, [0, pad_k]),
                )
        propagate_shape(new_attr, model)
        return new_attr

    new_weight = pad_param(weight)
    new_bias = pad_param(bias) if bias is not None else None

    with model.graph.inserting_before(node):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default, (input, [0, pad_k]),
        )
        layer_norm = model.graph.call_function(
            torch.ops.quantized_ops.layer_norm.default,
            (new_input, normalize_shape, new_weight, new_bias) + node.args[4:]
        )
        slice_node = model.graph.call_function(
            torch.ops.aten.slice.Tensor, (layer_norm, -1, 0, orig_k),
        )

    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")

    propagate_shape(layer_norm)

    propagate_shape(slice_node)
    node.replace_all_uses_with(slice_node)
    model.graph.erase_node(node)


def pad_vector_ops_to_hardware_unroll_size(
    model: GraphModule,
    K_unroll,
) -> GraphModule:
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
        if node.target == torch.ops.aten.layer_norm.default:
            pad_layer_norm_to_hardware_unroll_size(model, node, K_unroll)
            continue

        if node.target != torch.ops.aten.softmax.int:
            continue

        input = node.args[0]
        reduction_dim = input.shape[-1]

        pad_k = (K_unroll - (reduction_dim % K_unroll)) % K_unroll

        if not pad_k:
            continue

        with model.graph.inserting_after(input):
            new_input = model.graph.call_function(
                torch.ops.aten.pad.default,
                (input, [0, pad_k], "constant", float("-inf")),
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
    model: GraphModule,
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
