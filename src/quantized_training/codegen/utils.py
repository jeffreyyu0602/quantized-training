import itertools
import logging
from typing import Callable, List

import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import GraphModule
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .mapping import graph_copy
from ..pt2e_utils import get_aten_graph_module
from ..quantize_pt2e import create_getattr_from_value
from ..quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "convert_cat_and_stack_as_stack_on_dim0",
    "convert_cat_with_mismatched_shapes_to_stack",
    "convert_expand_to_memory_copy",
    "eliminate_dtype_conversion",
    "get_conv_bn_layers",
    "pad_matmul_inputs_for_unroll_alignment",
    "pad_vit_embeddings_output",
    "replace_target_with_vmap",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "rewrite_quantize_mx_for_lastdim",
]


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

    partitions = get_source_partitions(graph, [torch.stack, torch.cat])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        cat_node = partitions.pop(0).output_nodes[0]
        if cat_node.target not in [
            torch.ops.aten.cat.default, torch.ops.aten.stack.default
        ]:
            continue

        if not all(hasattr(n, "shape") for n in cat_node.args[0]):
            logger.warning(f"Node {cat_node} do not have a shape attribute")
            continue

        input_shape = list(cat_node.args[0][0].shape)

        if not all(list(n.shape) == input_shape for n in cat_node.args[0][1:]):
            shapes = [n.shape for n in cat_node.args[0]]
            logger.warning(
                "Concatenated tensors have different shapes in node %s. Shapes: %s",
                cat_node,
                shapes
            )
            continue

        if len(cat_node.args) == 1 or cat_node.args[1] == 0:
            continue

        concat_dim = cat_node.args[1]
        if concat_dim < 0:
            concat_dim += len(input_shape)

        # Always stack along the first dimension
        if cat_node.target == torch.ops.aten.stack.default:
            cat_node.args = (cat_node.args[0], 0)
            stack_node = cat_node
        else:
            with graph.inserting_after(cat_node):
                stack_node = graph.call_function(
                    torch.ops.aten.stack.default, (cat_node.args[0], 0)
                )

        # Permute the concatenated tensor to match the original order
        dims = list(range(len(input_shape) + 1))[1:]
        dims.insert(concat_dim, 0)

        with graph.inserting_after(stack_node):
            permute_node = graph.call_function(
                torch.ops.aten.permute.default, (stack_node, dims),
            )
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

        args = torch.fx.node.map_arg(node.args, lambda n: n.value)
        shape = list(args[0][0].shape[:dim])

        class Concat(torch.nn.Module):
            def forward(self, *inputs):
                result = []
                for idx in itertools.product(*[range(dim) for dim in shape]):
                    tensor = torch.cat([x[idx] for x in inputs], dim=0)
                    result.append(tensor)
                output = torch.stack(result, dim=0)
                return output.reshape(*shape, *output.shape[1:])

        gm = capture_pre_autograd_graph(Concat(), (*args[0],))
        graph_copy(model, gm, node)

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
                            stacked_tensors.append(input.squeeze(dim) * 1)
                        input = torch.stack(stacked_tensors, dim=dim)
                    elif input.shape[dim] != size:
                        raise ValueError(
                            f"Cannot expand dimension {dim} from {input.shape[dim]} to {size}."
                        )

                return input

        gm: torch.fx.GraphModule = capture_pre_autograd_graph(
            Expand(), (input_node.meta["val"],)
        )

        graph_copy(model, gm, node)
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

    get_attr_node = next(iter(n for n in pattern_graph.nodes if n.name == "_param_constant0"))

    for match in _matches:
        input_node = match.placeholder_nodes[0]
        output_node = match.returning_nodes[0]
        input_shape = input_node.meta["val"].shape
        new_get_attr_node = match.nodes_map[get_attr_node]
        layer_norm_inputs = [input_node, [input_shape[-1]], new_get_attr_node]

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


def rewrite_quantize_mx_for_lastdim(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target != torch.ops.quantized_ops.quantize_mx.default:
            continue

        axis = node.args[1]
        if axis == -1 or axis == node.value[1].ndim - 1:
            continue

        with graph.inserting_before(node):
            transpose_node = graph.call_function(
                torch.ops.aten.transpose.int,
                (node.args[0], axis, -1),
            )

        node.replace_input_with(node.args[0], transpose_node)

        node.args = node.args[:1] + (-1,) + node.args[2:]

        for user in node.users:
            with graph.inserting_after(user):
                transpose_back = graph.call_function(
                    torch.ops.aten.transpose.int,
                    (user, -1, axis),
                )

            for n in list(user.users):
                if id(n) != id(transpose_back):
                    n.replace_input_with(user, transpose_back)

            transpose_back.meta["dtype"] = node.meta["dtype"]
            transpose_back.meta["source_fn_stack"] = [(transpose_back.name, "transpose")]

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def replace_quantize_mx_with_reduce(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target != torch.ops.quantized_ops.quantize_mx.default:
            continue

        axis = node.args[1]
        if axis == -1 or axis == node.value[1].ndim - 1:
            continue

        axis = axis if axis >= 0 else axis + node.value[1].ndim
        quant_max = node.args[2]
        block_size = node.args[3]
        dtype = node.args[4]

        from ..fake_quantize import get_quantization_map

        @torch.compiler.assume_constant_result
        def get_code():
            code = get_quantization_map(dtype)
            return code[0] if isinstance(code, tuple) else code

        class QuantizeMXDecomposed(torch.nn.Module):
            def forward(self, input):
                new_shape = input.shape[:axis] + (-1, block_size) + input.shape[axis + 1:]
                reshape = input.reshape(new_shape)
                scale = torch.amax(torch.abs(reshape), axis + 1) * (1.0 / quant_max)
                code = get_code()
                quantized = torch.ops.quantized_ops.quantize(
                    input, scale, dtype, code, block_size
                )
                return scale, quantized

        gm = capture_pre_autograd_graph(QuantizeMXDecomposed(), (node.args[0].meta["val"],))
        graph_copy(model, gm, node)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def eliminate_dtype_conversion(model: torch.fx.GraphModule):
    for node in list(model.graph.nodes):
        # Eliminate dtype conversion nodes.
        if node.target == torch.ops.aten.to.dtype:
            node.replace_all_uses_with(node.args[0])
            model.graph.erase_node(node)
            continue

        # Remove the dtype argument from softmax nodes.
        if node.target == torch.ops.aten.softmax.int and len(node.args) > 2:
            node.args = node.args[:-1]

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def pad_matmul_inputs_for_unroll_alignment(
    model: torch.fx.GraphModule,
    c_unroll: int = 32,
    k_unroll: int = 32,
) -> torch.fx.GraphModule:
    """
    Pad inputs to GEMM (matrix multiplication) nodes in a torch.fx.GraphModule so that
    the dimensions C and K are multiples of the provided unroll factors.

    The GEMM operation is assumed to multiply:
        - input1 of shape [..., X, C]
        - input2 of shape [..., C, K]
    resulting in an output of shape [..., X, K].

    Padding is applied as follows:
        - For input1: pad the C dimension (last dimension) by pad_C.
        - For input2: pad the C dimension (second-to-last) by pad_C and the K dimension (last)
          by pad_K.

    After the GEMM, the output is sliced to remove the extra padded columns in the K dimension.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        c_unroll (int): Unroll factor for the C (shared inner) dimension.
        k_unroll (int): Unroll factor for the K dimension.

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.matmul.default:
            continue

        # Get the two input nodes and their shapes.
        input1, input2 = node.args[0], node.args[1]
        shape1 = input1.meta["val"].shape  # Expected: [..., X, C]
        shape2 = input2.meta["val"].shape  # Expected: [..., C, K]

        # Process only if the inputs are at least 3D (batched or higher-dimensional).
        input1_ndim = sum(1 for d in shape1 if d > 1)
        input2_ndim = sum(1 for d in shape2 if d > 1)
        if input1_ndim < 3 and input2_ndim < 3:
            continue

        # Interpret dimensions as: input1 is [..., X, C] and input2 is [..., C, K]
        C = shape1[-1]
        K = shape2[-1]

        # Compute required padding.
        pad_C = (c_unroll - (C % c_unroll)) % c_unroll
        pad_K = (k_unroll - (K % k_unroll)) % k_unroll

        # Pad input1 (shape: [..., X, C]) along C dimension if needed.
        if pad_C > 0:
            with model.graph.inserting_before(node):
                padded_input1 = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (input1, [0, pad_C]),
                )
            node.replace_input_with(input1, padded_input1)
            if (dtype := input1.meta.get("dtype")) is not None:
                padded_input1.meta["dtype"] = dtype

        # Pad input2 (shape: [..., C, K]) along C and K dimensions if needed.
        if pad_C > 0 or pad_K > 0:
            with model.graph.inserting_before(node):
                padded_input2 = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (input2, [0, pad_K, 0, pad_C]),
                )
            node.replace_input_with(input2, padded_input2)
            if (dtype := input2.meta.get("dtype")) is not None:
                padded_input2.meta["dtype"] = dtype

        # After GEMM, slice the output to remove the extra padded columns in the K dimension.
        user_node = next(iter(node.users))
        output_node = node

        if pad_K:
            with model.graph.inserting_before(user_node):
                sliced_output = model.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (output_node, -1, 0, K),
                )
            for user in list(output_node.users):
                if id(user) != id(sliced_output):
                    user.replace_input_with(output_node, sliced_output)
            if (dtype := output_node.meta.get("dtype")) is not None:
                sliced_output.meta["dtype"] = dtype

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
