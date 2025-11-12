import itertools
import logging
from typing import Callable, List

import torch
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .utils import _pair
from ..mapping import (
    get_parameter_or_buffer,
    propagate_shape,
    replace_node_with_graph_module,
)
from ..mapping_utils import is_nop
from ...pt2e_utils import get_aten_graph_module
from ...quantize_pt2e import create_getattr_from_value, export_model
from ...quantizer.xnnpack_quantizer_utils import _convert_scalars_to_attrs

logger = logging.getLogger(__name__)

__all__ = [
    "convert_cat_and_stack_as_stack_on_dim0",
    "convert_cat_with_mismatched_shapes_to_stack",
    "convert_expand_to_memory_copy",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "replace_target_with_vmap",
    "replace_conv2d_with_im2col",
    "extract_input_preprocessor",
    "rewrite_fx_graph",
    "inline_autocast_modules",
    "remove_softmax_dtype_cast",
]


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
                # Stack along the first dimension to create the expanded shape
                for dim, size in enumerate(sizes):
                    if input.shape[dim] == 1 and size > 1:
                        input = torch.stack([input.squeeze(dim)] * size, dim=dim)
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


def replace_rmsnorm_with_layer_norm(
    model: GraphModule,
    layer_norm: torch.nn.Module,
    example_input,
    convert_scalars_to_attrs=False,
):
    """Replace LLaMA RMSNorm with ATen layer_norm
    """
    original_graph = model.graph

    pattern = get_aten_graph_module(layer_norm, example_input)
    if convert_scalars_to_attrs:
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
    Extract the input preprocessing operations from the given FX GraphModule
    and create a separate GraphModule.

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


def inline_autocast_modules(model: torch.fx.GraphModule):
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


def remove_softmax_dtype_cast(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.softmax.int:
            node.args = node.args[:2]

    graph.lint()
    model.recompile()
    return model
