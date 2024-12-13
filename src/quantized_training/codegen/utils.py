import logging
from typing import List

import torch
from torch._export import capture_pre_autograd_graph
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

from .mapping import _decompose_node
from ..pt2e_utils import get_aten_graph_module
from ..quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs

logger = logging.getLogger(__name__)


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

    param_node = next(iter(n for n in pattern_graph.nodes if n.name == "_param_constant0"))

    for match in _matches:
        mapped_param_node = match.nodes_map[param_node]
        input_node = match.placeholder_nodes[0]
        output_node = match.returning_nodes[0]
        input_shape = input_node.meta["val"].shape
        layer_norm_inputs = [input_node, [input_shape[-1]], mapped_param_node]
        with original_graph.inserting_before(output_node):
            new_node = original_graph.call_function(
                torch.ops.aten.layer_norm.default,
                tuple(layer_norm_inputs),
                {},
            )
        output_node.replace_all_uses_with(new_node)
        original_graph.erase_node(output_node)

    original_graph.lint()
    original_graph.eliminate_dead_code()
    model.recompile()


def eliminate_dtype_conversion(model: torch.fx.GraphModule):
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.to.dtype:
            node.replace_all_uses_with(node.args[0])
            model.graph.erase_node(node)

        if node.target == torch.ops.aten.softmax.int and len(node.args) > 2:
            node.args = node.args[:-1]

    model.graph.lint()
    model.recompile()


def convert_expand(model: torch.fx.GraphModule):
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.expand.default:
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
                            stacked_tensors.append(input.squeeze(dim) * 1)
                        input = torch.stack(stacked_tensors, dim=dim)
                    elif input.shape[dim] != size:
                        raise ValueError(f"Cannot expand dimension {dim} from {input.shape[dim]} to {size}.")

                return input

        gm: torch.fx.GraphModule = capture_pre_autograd_graph(
            Expand(), (input_node.meta["val"],)
        )

        _decompose_node(model, gm, node)
        model.graph.erase_node(node)

    model.graph.lint()
    model.recompile()


def pad_matmul_to_multiples_of_unroll_dim(
    model: torch.fx.GraphModule,
    ic_unroll = 32,
    oc_unroll = 32
):
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.matmul.default:
            continue

        input1 = node.args[0]
        input2 = node.args[1]
        input1_ndim = sum(1 for d in input1.meta["val"].shape if d > 1)
        input2_ndim = sum(1 for d in input2.meta["val"].shape if d > 1)
        if input1_ndim < 3 and input2_ndim < 3:
            continue

        input_pad = (ic_unroll - (input1.meta["val"].shape[-2] % ic_unroll)) % ic_unroll
        ic_pad = (ic_unroll - (input1.meta["val"].shape[-1] % ic_unroll)) % ic_unroll
        oc_pad = (oc_unroll - (input2.meta["val"].shape[-1] % oc_unroll)) % oc_unroll

        if input_pad or ic_pad:
            with model.graph.inserting_before(node):
                pad_node = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (input1, [0, ic_pad, 0, input_pad]),
                )
                node.replace_input_with(input1, pad_node)
            if (dtype := input1.meta.get("dtype")) is not None:
                pad_node.meta["dtype"] = dtype

        if ic_pad or oc_pad:
            with model.graph.inserting_before(node):
                pad_node = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (input2, [0, oc_pad, 0, ic_pad]),
                )
                node.replace_input_with(input2, pad_node)
            if (dtype := input2.meta.get("dtype")) is not None:
                pad_node.meta["dtype"] = dtype

        user_node = next(iter(node.users))
        output_node = node
        if input_pad:
            with model.graph.inserting_before(user_node):
                output_node = model.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (node, -2, 0, input1.meta["val"].shape[-2]),
                )
            for user in list(node.users):
                if id(user) != id(output_node):
                    user.replace_input_with(node, output_node)
            if (dtype := node.meta.get("dtype")) is not None:
                output_node.meta["dtype"] = dtype

        if oc_pad:
            with model.graph.inserting_before(user_node):
                slice_node = model.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (output_node, -1, 0, input2.meta["val"].shape[-1]),
                )
            for user in list(output_node.users):
                if id(user) != id(slice_node):
                    user.replace_input_with(output_node, slice_node)
            if (dtype := output_node.meta.get("dtype")) is not None:
                slice_node.meta["dtype"] = dtype

    model.graph.lint()
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

    for user in list(vit_embed_out.users):
        if id(user) != id(pad_node):
            user.replace_input_with(vit_embed_out, pad_node)

    for node in model.graph.nodes:
        if node.target == torch.ops.aten.view.default:
            new_size = [x if x != orig_dim else x + pad for x in node.args[1]]
            node.args = (node.args[0], new_size)

    model.graph.lint()
    model.recompile()
    return model


def replace_permute_with_transpose(model: torch.fx.GraphModule):
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.permute.default:
            continue

        # if permuted dims is the last two dims
        permute_dims = node.args[1]
        tranpose_dims = list(range(len(permute_dims)))
        tranpose_dims[-2], tranpose_dims[-1] = tranpose_dims[-1], tranpose_dims[-2]
        if permute_dims != tranpose_dims:
            continue

        with model.graph.inserting_before(node):
            transpose_node = model.graph.call_function(
                torch.ops.aten.transpose.int,
                (node.args[0], -1, -2),
            )

        # Since we are doing a 1 to 1 replacement, we can copy over the meta data
        transpose_node.meta = node.meta

        node.replace_all_uses_with(transpose_node)
        model.graph.erase_node(node)

        logger.info(f"Replaced permute with transpose: {node} -> {transpose_node}")

    model.graph.lint()
    model.recompile()
    return model
