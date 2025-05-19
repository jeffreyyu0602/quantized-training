import collections
import itertools
import logging
import operator
from itertools import repeat
from typing import Callable, List, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .mapping import replace_node_with_graph_module
from .mapping_utils import _is_elementwise_op
from ..pt2e_utils import get_aten_graph_module
from ..quantize_pt2e import create_getattr_from_value, export_model
from ..quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "convert_cat_and_stack_as_stack_on_dim0",
    "convert_cat_with_mismatched_shapes_to_stack",
    "convert_expand_to_memory_copy",
    "get_conv_bn_layers",
    "insert_permute_adapters_for_conv2d",
    "pad_gemm_inputs_to_hardware_unroll_size",
    "pad_conv2d_inputs_to_hardware_unroll_size",
    "pad_vit_embeddings_output",
    "replace_conv2d_with_im2col",
    "replace_target_with_vmap",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "replace_target",
    "rewrite_quantize_mx_for_lastdim",
    "transpose_linear_weights",
    "strip_softmax_dtype",
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


def get_parameter_or_buffer(model: torch.nn.Module, name: str):
    """Retrieve a parameter or buffer from the model by name."""
    if name in dict(model.named_parameters()):
        return model.get_parameter(name)
    if name in dict(model.named_buffers()):
        return model.get_buffer(name)
    if hasattr(model, name):
        return getattr(model, name)
    raise ValueError(f"Parameter or buffer '{name}' not found in the model.")


def fuse(model: torch.fx.GraphModule) -> torch.nn.Module:
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
            conv_node = node.args[0]

            conv_w = model.get_parameter(conv_node.args[1])
            conv_b = model.get_parameter(conv_node.args[2])

            bn_w = model.get_parameter(node.args[1])
            bn_b = model.get_parameter(node.args[2])
            bn_rm = model.get_buffer(node.args[3])
            bn_rv = model.get_buffer(node.args[4])
            bn_eps = node.args[7]

            fused_conv_w, fused_conv_b = fuse_conv_bn_weights(
                conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b
            )

            model.register_parameter(conv_node.args[1].target, fused_conv_w)
            model.register_parameter(conv_node.args[2].target, fused_conv_b)

            node.replace_all_uses_with(conv_node)
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
                            stacked_tensors.append(input.squeeze(dim) * 1)
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


def strip_softmax_dtype(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.softmax.int:
            node.args = node.args[:2]

    graph.lint()
    model.recompile()
    return model


def pad_gemm_inputs_to_hardware_unroll_size(
    model: torch.fx.GraphModule,
    C_unroll: int = 32,
    K_unroll: int = 32,
) -> torch.fx.GraphModule:
    """
    Pad inputs to GEMM (matrix multiplication) nodes in a torch.fx.GraphModule so that
    the dimensions C and K are multiples of the provided unroll factors. After the GEMM,
    the output is sliced to remove the extra padded columns in the K dimension.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        C_unroll (int): Unroll factor for the C (shared inner) dimension.
        K_unroll (int): Unroll factor for the K dimension.

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
        pad_C = (C_unroll - (C % C_unroll)) % C_unroll
        pad_K = (K_unroll - (K % K_unroll)) % K_unroll

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


def pad_conv2d_inputs_to_hardware_unroll_size(
    model: torch.fx.GraphModule,
    C_unroll: int = 32,
    K_unroll: int = 32,
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
    for node in list(model.graph.nodes):
        if node.target != torch.ops.aten.conv2d.default:
            continue

        if len(node.args) < 7 or node.args[6] != 1:
            continue

        input, weight = node.args[:2]
        C_in = input.shape[1]
        C_out = weight.shape[0]

        pad_C = (C_unroll - (C_in % C_unroll)) % C_unroll
        pad_K = (K_unroll - (C_out % K_unroll)) % K_unroll

        # Pad input along C dimension
        if pad_C > 0:
            pad_dims_input = [0, 0, 0, 0, 0, pad_C]
            with model.graph.inserting_before(node):
                padded_input = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (input, pad_dims_input),
                )
            node.replace_input_with(input, padded_input)
            if (dtype := input.meta.get("dtype")) is not None:
                padded_input.meta["dtype"] = dtype

        # Pad weight along K and C
        if pad_C > 0 or pad_K > 0:
            param = get_parameter_or_buffer(model, weight.target)
            pad_dims_weight = [0, 0, 0, 0, 0, pad_C, 0, pad_K]
            param.data = F.pad(param.data, pad_dims_weight)

            if len(node.args) > 2 and node.args[2] is not None:
                bias_param = get_parameter_or_buffer(model, node.args[2].target)
                bias_param.data = F.pad(bias_param.data, [0, pad_K])

        # Slice output along channel dimension to remove padding in C_out
        if pad_K:
            user_node = next(iter(node.users))
            output_node = node

            with model.graph.inserting_before(user_node):
                slice_node = model.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (output_node, 1, 0, C_out),
                )
            for user in list(output_node.users):
                if id(user) != id(slice_node):
                    user.replace_input_with(output_node, slice_node)
            if (dtype := output_node.meta.get("dtype")) is not None:
                slice_node.meta["dtype"] = dtype

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
    return torch.ops.aten.conv2d.default(
        input.permute(0, 3, 1, 2),
        weight.permute(3, 2, 0, 1),
        bias,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        groups,
    ).permute(0, 2, 3, 1)


def linear_transposed(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return torch.ops.aten.linear.default(input, weight.T, bias)


def is_conv2d(node: Node) -> bool:
    return node.op == "call_function" and node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d_mx.default
    ]


def dfs_collect_connected_conv2d_chain(start: Node, visited: Set[Node]) -> Set[Node]:
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
                torch.ops.quantized_ops.quantize_mx.default,
                operator.getitem,
            ]:
                stack.append(user)

    return chain


def insert_permute_adapters_for_conv2d(model: GraphModule):
    graph = model.graph
    visited: Set[Node] = set()

    torch.nn.functional.conv2d = conv2d_transposed

    for node in graph.nodes:
        if not is_conv2d(node) and node.target not in [
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.aten.max_pool2d.default,
        ]:
            continue

        if node in visited:
            continue

        conv_chain = dfs_collect_connected_conv2d_chain(node, visited)

        # Insert pre-permute before each input point
        pre_permute_nodes = {}
        post_permute_nodes = {}
        node_map = {}
        for n in conv_chain:
            for arg in n.all_input_nodes:
                if arg in conv_chain or arg.op == "get_attr" or arg in node_map:
                    continue
                logger.debug(f"Insert pre-permute for {arg}")
                if arg not in pre_permute_nodes:
                    with graph.inserting_after(arg):
                        pre_permute_nodes[arg] = graph.call_function(
                            torch.ops.aten.permute.default,
                            args=(arg, (0, 2, 3, 1))
                        )
                n.replace_input_with(arg, pre_permute_nodes[arg])

            for user in list(n.users.keys()):
                if user in conv_chain or user in node_map:
                    continue
                logger.debug(f"Insert post-permute for {n}")
                with graph.inserting_before(user):
                    permute_node = graph.call_function(
                        torch.ops.aten.permute.default,
                        args=(n, (0, 3, 1, 2))
                    )
                user.replace_input_with(n, permute_node)

            if n.target == torch.ops.quantized_ops.quantize_mx.default:
                n.args = n.args[:1] + (-1,) + n.args[2:]

            if not is_conv2d(n):
                continue

            logger.debug(f"Permuting weights for {n}")
            weight_node = n.args[1]
            param = get_parameter_or_buffer(model, weight_node.target)
            param.data = param.data.permute(2, 3, 1, 0)

            if n.target == torch.ops.quantized_ops.conv2d_mx.default:
                scale_node = n.kwargs.get("weight_scale")
                scale = get_parameter_or_buffer(model, scale_node.target)
                scale.data = scale.data.permute(2, 3, 1, 0)
                continue

            with graph.inserting_before(n):
                conv_node = graph.call_function(
                    torch.ops.quantized_ops.conv2d.default, args=n.args
                )

            logger.debug(f"Replace {n} with permute for {conv_node}")

            conv_node.meta = n.meta
            n.replace_all_uses_with(conv_node)
            graph.erase_node(n)

            node_map[conv_node] = n

    graph.lint()
    model.recompile()
    return model


def transpose_linear_weights(model: GraphModule, transpose_fc: bool = False):
    """
    Transpose the weights of linear layers in the given FX graph module.

    Args:
        model (GraphModule): The FX graph module to transform.
        transpose_fc (bool): If True, transpose the weights of fully connected layers.

    Returns:
        GraphModule: The transformed FX graph module with transposed weights.
    """
    torch.nn.functional.linear = linear_transposed

    for node in model.graph.nodes: 
        if node.target not in [
            torch.ops.aten.linear.default,
            torch.ops.quantized_ops.linear_mx.default,
        ]:
            continue

        input_node = node.args[0]
        input_shape = input_node.value.shape
        if not transpose_fc and sum(input_shape[:-1]) == 1:
            continue

        weight_node = node.args[1]
        weight = get_parameter_or_buffer(model, weight_node.target)
        weight.data = weight.data.T

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

        node.replace_all_uses_with(linear_node)
        model.graph.erase_node(node)

    model.graph.lint()
    model.recompile()
    return model


def replace_target(model: torch.fx.GraphModule, target_to_replace, new_target):
    graph = model.graph
    for node in graph.nodes:
        if node.target != target_to_replace:
            continue
        with graph.inserting_after(node):
            new_node = graph.call_function(new_target, node.args)
        new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()


def replace_conv2d_with_im2col(model: GraphModule):
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

        if input_node.value.shape[1] != 3:
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

        bias_node = args[2]
        if bias_node is not None:
            param = model.get_parameter(bias_node.target)
            param.data = param.data.reshape(C_out, 1)

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

    model.graph.lint()
    model.recompile()
    return model