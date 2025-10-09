import copy
import itertools
import logging
import math
import operator
import os
import re
from collections import defaultdict
from typing import List, Dict, Callable, Union

import graphviz
import torch
from torch.fx import Node, Graph, GraphModule
from torch.fx.node import map_arg
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
from transformers.utils.import_utils import is_torch_greater_or_equal

from .mapping_utils import (
    is_addressing_op,
    is_conv2d,
    is_elementwise_op,
    is_gemm_op,
    is_indexing_or_concatenation_op,
    is_matmul,
    is_nop,
    is_reshape_op,
    map_node,
    set_output_field,
    set_tensor_field,
)
from .memory import MemoryAllocator, Segment, MemorySpace
from .param_pb2 import Model, Operation, Tensor
from .shape_prop import ShapeProp
from ..pt2e_utils import dtype_byte_size
from ..quantize_pt2e import create_getattr_from_value, export_model

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_SIZE = torch.finfo(torch.float32).max


def eliminate_dead_code(self):
    """
    Remove all dead code from the graph, based on each node's number of
    users, and whether the nodes have any side effects. The graph must be
    topologically sorted before calling.

    Returns:
        bool: Whether the graph was changed as a result of the pass.
    """
    # Lint the graph first to make sure its topologically sorted, otherwise
    # DCE below will not behave as expected.
    self.lint()

    # Reverse iterate so that when we remove a node, any nodes used as an
    # input to that node have an updated user count that no longer reflects
    # the removed node.
    changed = False
    for node in reversed(self.nodes):
        if node.op != 'output' and len(node.users) == 0:
            self.erase_node(node)
            changed = True

    return changed


def propagate_shape(node: Node, model: GraphModule = None):
    def load_arg(a):
        return torch.fx.graph.map_arg(a, lambda n: getattr(n, "value", n.meta.get("val")))

    def fetch_attr(target : str):
        target_atoms = target.split('.')
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    modules = dict(model.named_modules()) if model is not None else {}

    if node.op == 'get_attr':
        result = fetch_attr(node.target)
    elif node.op == 'call_function':
        result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
    elif node.op == 'call_method':
        self_obj, *args = load_arg(node.args)
        kwargs = load_arg(node.kwargs)
        result = getattr(self_obj, node.target)(*args, **kwargs)
    elif node.op == 'call_module':
        result = modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
    elif node.op == 'output':
        result = load_arg(node.args[0])

    if isinstance(result, torch.Tensor):
        node.shape = result.shape
        node.value = result.cpu().clone()
    elif isinstance(result, (tuple, list)):
        node.value = [x.cpu().clone() if isinstance(x, torch.Tensor) else x for x in result]
    else:
        node.value = result


def get_parameter_or_buffer(model: torch.nn.Module, name: str):
    """Retrieve a parameter or buffer from the model by name."""
    if name in dict(model.named_parameters()):
        return model.get_parameter(name)
    if name in dict(model.named_buffers()):
        return model.get_buffer(name)
    if hasattr(model, name):
        return getattr(model, name)
    raise ValueError(f"Parameter or buffer '{name}' not found in the model.")


def replace_node_with_graph_module(
    self: GraphModule, module: GraphModule, source: Node, value_remap=None
) -> List[Node]:
    if value_remap is None:
        value_remap = {}
    output = None
    args_iter = iter(source.all_input_nodes)
    for node in list(module.graph.nodes):
        if node.op == 'placeholder':
            value_remap[node] = next(args_iter, None)
        elif node.op == 'output':
            output = node.args[0]
            if len(output) == 1:
                source.replace_all_uses_with(value_remap[output[0]])
            else:
                for user in list(source.users):
                    assert user.target == operator.getitem
                    select_idx = user.args[1]
                    user.replace_all_uses_with(value_remap[output[select_idx]])
        else:
            with self.graph.inserting_before(source):
                if node.op == 'get_attr':
                    param = get_parameter_or_buffer(module, node.target)
                    value_remap[node] = create_getattr_from_value(
                        self, self.graph, "_tensor_constant_", param
                    )
                else:
                    value_remap[node] = self.graph.node_copy(
                        node, lambda n: value_remap[n]
                    )

            if (source_fn_st := node.meta.get('source_fn_stack', None)) is not None:
                source_fn = source_fn_st[-1]
                value_remap[node].meta['source_fn_stack'] = [
                    (value_remap[node].name, source_fn[1])
                ]

            propagate_shape(value_remap[node], self)

    return [value_remap[n] for n in output]


def _decompose_bmm(model: GraphModule, node: Node):
    assert node.op == 'call_function' and node.target == torch.ops.aten.matmul.default
    input1 = node.args[0].value
    input2 = node.args[1].value

    input1_dims = sum(1 for d in input1.shape if d > 1)
    input2_dims = sum(1 for d in input2.shape if d > 1)
    if input1_dims < 3 and input2_dims < 3:
        return None

    class BMM(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Loop through each element in the batch dimensions
            batch_shape = x.shape[:-2]
            result = []
            for idx in itertools.product(*[range(dim) for dim in batch_shape]):
                result.append(torch.matmul(x[idx], y[idx]))
            result = torch.stack(result)
            result = result.view(*batch_shape, *result.shape[-2:])
            return result

    gm = export_model(BMM(), (input1, input2))
    output_nodes = replace_node_with_graph_module(model, gm, node)
    model.graph.erase_node(node)
    return output_nodes[0]


def _decompose_bmm_mx(model: GraphModule, node: Node):
    assert (
        node.op == 'call_function' and
        node.target == torch.ops.quantized_ops.matmul_mx.default
    )

    input1 = node.args[0].value
    input2 = node.args[1].value

    input1_dims = sum(1 for d in input1.shape if d > 1)
    input2_dims = sum(1 for d in input2.shape if d > 1)
    if input1_dims < 3 and input2_dims < 3:
        return None

    block_size = node.kwargs['block_size']

    class BMM(torch.nn.Module):
        def forward(
                self, input: torch.Tensor, other: torch.Tensor, input_scale=None,
                weight_scale=None, input_code=None, weight_code=None):
            # Loop through each element in the batch dimensions
            batch_shape = input.shape[:-2]
            result = []
            for idx in itertools.product(*[range(dim) for dim in batch_shape]):
                result.append(torch.ops.quantized_ops.matmul_mx(
                    input[idx],
                    other[idx],
                    input_scale=input_scale[idx],
                    weight_scale=weight_scale[idx],
                    block_size=block_size,
                    input_code=input_code,
                    weight_code=weight_code,
                ))
            result = torch.stack(result)
            result = result.view(*batch_shape, *result.shape[-2:])
            return result

    input_code = node.kwargs.get('input_code', None)
    weight_code = node.kwargs.get('weight_code', None)

    kwargs = {
        'input_scale': node.kwargs['input_scale'].value,
        'weight_scale': node.kwargs['weight_scale'].value,
        'input_code': input_code.value if input_code is not None else None,
        'weight_code': weight_code.value if weight_code is not None else None,
    }

    gm = export_model(BMM(), (input1, input2), kwargs)

    # Remove unused placeholder nodes
    for n in gm.graph.nodes:
        if n.op == 'placeholder' and len(n.users) == 0:
            gm.graph.erase_node(n)
    gm.graph.lint()

    source_fn = node.meta['source_fn_stack'][-1]
    for n in gm.graph.nodes:
        if n.target == torch.ops.quantized_ops.matmul_mx.default:
            n.meta['source_fn_stack'] = [(n.name, source_fn[1])]

    output_nodes = replace_node_with_graph_module(model, gm, node)
    model.graph.erase_node(node)
    return output_nodes[0]


def split_multi_head_attention(model: GraphModule):
    graph = model.graph

    grouped_nodes = defaultdict(list)
    for node in list(graph.nodes):
        if node.target not in [
            torch.ops.aten.matmul.default,
            torch.ops.quantized_ops.matmul_mx.default,
        ]:
            continue

        if (nn_module_stack := node.meta.get('nn_module_stack', None)) is not None:
            bt = list(nn_module_stack.values())[-1]
            grouped_nodes[bt[0]].append(node)

    for nodes in grouped_nodes.values():
        if len(nodes) != 2:
            for node in nodes:
                if node.target == torch.ops.aten.matmul.default:
                    _decompose_bmm(model, node)
                else:
                    _decompose_bmm_mx(model, node)
            continue

        qk_matmul, av_matmul = nodes[0], nodes[1]

        query = qk_matmul.args[0]
        key = qk_matmul.args[1]
        value = av_matmul.args[1]
        query_scale = qk_matmul.kwargs.get('input_scale', None)
        key_scale = qk_matmul.kwargs.get('weight_scale', None)
        value_scale = av_matmul.kwargs.get('weight_scale', None)

        # Find the nodes between the qk and av matmuls
        def dfs(current_node, visited, max_depth=20):
            if len(visited) > max_depth:
                return None
            if current_node == av_matmul:
                return [visited]
            paths = []
            for user in current_node.users:
                if user not in visited:
                    if (result := dfs(user, visited + [user], max_depth)) is None:
                        return None
                    paths.extend(result)
            return paths

        paths = dfs(qk_matmul, [qk_matmul])

        # Decompose BMM into multiple matmuls
        if qk_matmul.target == torch.ops.aten.matmul.default:
            qk_output = _decompose_bmm(model, qk_matmul)
            av_output = _decompose_bmm(model, av_matmul)
        else:
            qk_output = _decompose_bmm_mx(model, qk_matmul)
            av_output = _decompose_bmm_mx(model, av_matmul)

        if paths is None:
            logger.warning(
                f"Failed to find paths between {qk_matmul} and {av_matmul}. "
                "Skipping fusion."
            )
            continue

        nodes_between = set()
        for path in paths:
            nodes_between.update(path[1:-1])

        # Sort the nodes between the qk and av matmuls
        order = {node: idx for idx, node in enumerate(graph.nodes)}
        nodes_between = sorted(nodes_between, key=lambda n: order[n])

        # Annotate the dtype of the new nodes in the graph
        def propagate_input_dtype(node):
            if (dtype := node.meta.get('dtype', None)) is None:
                return
            for user in node.users:
                if user.target == torch.ops.aten.select.int:
                    user.meta['dtype'] = dtype
                    propagate_input_dtype(user)

        def propagate_output_dtype(orig_node, new_node):
            if (dtype := orig_node.meta.get('dtype', None)) is None:
                return
            output_nodes = [new_node, new_node.args[0]] + new_node.args[0].args[0]
            for node in output_nodes:
                node.meta['dtype'] = dtype

        if qk_output is not None:
            propagate_input_dtype(query)
            propagate_input_dtype(key)
            propagate_output_dtype(qk_matmul, qk_output)

        if av_output is not None:
            propagate_input_dtype(value)
            propagate_output_dtype(av_matmul, av_output)

        if query_scale is not None:
            propagate_input_dtype(query_scale)

        if key_scale is not None:
            propagate_input_dtype(key_scale)

        if value_scale is not None:
            propagate_input_dtype(value_scale)

        if qk_output is None or av_output is None:
            continue

        # Duplicate the nodes between the qk and av matmuls to perform fusion
        qk_matmuls = qk_output.args[0].args[0]
        av_matmuls = av_output.args[0].args[0]

        for qk_matmul, av_matmul in zip(qk_matmuls, av_matmuls):
            value_remap = {qk_output: qk_matmul}
            for node in nodes_between:
                with graph.inserting_before(av_matmul):
                    value_remap[node] = graph.node_copy(node, lambda n: value_remap.get(n, n))

                if (source_fn_st := node.meta.get('source_fn_stack', None)) is not None:
                    source_fn = source_fn_st[-1]
                    value_remap[node].meta['source_fn_stack'] = [
                        (value_remap[node].name, source_fn[1])
                    ]

                propagate_shape(value_remap[node])

            av_matmul.replace_input_with(av_matmul.args[0], value_remap[nodes_between[-1]])

            if (scale_node := av_matmul.kwargs.get('input_scale', None)) is not None:
                av_matmul.replace_input_with(scale_node, value_remap[nodes_between[-2]])

    graph.lint()

    graph.eliminate_dead_code()
    model.recompile()

    return model


def _create_subgraph(nodes: List[Node]):
    new_args = []
    new_graph = torch.fx.Graph()
    value_remap = {}

    for node in nodes:
        for n in node.all_input_nodes:
            if n not in value_remap:
                value_remap[n] = new_graph.placeholder(n.name)
                new_args.append(n)
                value_remap[n].meta['source_node'] = n
        value_remap[node] = new_graph.node_copy(node, lambda n : value_remap[n])

    new_graph.output(value_remap[nodes[-1]])
    new_graph.lint()
    gm = torch.fx.GraphModule(torch.nn.Module(), new_graph)
    return gm, tuple(new_args)


OP_PARAM_ARG_INDEX = {
    torch.ops.aten.conv2d.default: 1,
    torch.ops.aten.linear.default: 1,
    torch.ops.aten.layer_norm.default: 2,
    torch.ops.quantized_ops.conv2d.default: 1,
    torch.ops.quantized_ops.linear.default: 1,
    torch.ops.quantized_ops.conv2d_mx.default: 1,
    torch.ops.quantized_ops.linear_mx.default: 1,
}


def get_unique_node_name(node: Node):
    """
    Generate a unique and meaningful name for the node based on its parameter.
    """
    if (pos := OP_PARAM_ARG_INDEX.get(node.target)) is not None:
        weight_node = node.args[pos]
        # There are cases where weights are sliced. Trace up to find the
        # get_attr node and use the parameter name
        while weight_node.target == torch.ops.aten.slice.Tensor:
            weight_node = weight_node.args[0]

        if weight_node.op == 'get_attr':
            return weight_node.name.split("_weight")[0]

    return node.name


def get_new_node_name_with_prefix(prefix: str):
    """
    Generate a new attribute name with a given prefix that is not already used
    in the module's graph.
    """
    prefix = prefix.replace(".", "_")

    def get_new_node_name(module: torch.nn.Module):
        existing_names = {n.name for n in module.graph.nodes}

        if prefix not in existing_names:
            return prefix

        i = 1
        while f"{prefix}_{i}" in existing_names:
            i += 1

        node_name = f"{prefix}_{i}"
        logger.debug(f"Generated new unique node name: {node_name}")
        return node_name

    return get_new_node_name


def get_submodule_name(module, nodes: List[Node]):
    prefix = "submodule"
    if is_torch_greater_or_equal("2.5"):
        node = next((n for n in nodes if (
            n.target in OP_PARAM_ARG_INDEX or is_gemm_op(n) or (
                n.op == 'call_function'
                and not is_nop(n)
                and not is_indexing_or_concatenation_op(n)
                and not is_reshape_op(n)
            )
        )), None)
        prefix = get_unique_node_name(node)
        if len(nodes) > 1:
            prefix += "_fused"

    get_new_node_name = get_new_node_name_with_prefix(prefix)
    return get_new_node_name(module)


def update_placeholder_meta(model, node, named_modules=None):
    """
    Update the metadata of all user nodes that consume the given node.
    """
    if named_modules is None:
        named_modules = dict(model.named_modules())

    for user in list(node.users):
        if user.op != "call_module":
            continue

        try:
            index = user.args.index(node)
        except ValueError:
            continue

        mod = named_modules[user.target]
        placeholders = [n for n in mod.graph.nodes if n.op == 'placeholder']
        if index >= len(placeholders):
            continue

        placeholder = placeholders[index]
        placeholder.name = node.name
        placeholder.meta['source_node'] = node


def rename_nodes_with_param_names(model: GraphModule):
    if not is_torch_greater_or_equal("2.5"):
        return
    graph = model.graph
    named_modules = dict(model.named_modules())
    for node in list(graph.nodes):
        if node.target in OP_PARAM_ARG_INDEX:
            node.name = get_submodule_name(model, [node])
            update_placeholder_meta(model, node, named_modules)
    graph.lint()
    model.recompile()


def _create_and_insert_subgraph(
    nodes: List[Node],
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module]
) -> Node:
    submodule, new_args = _create_subgraph(nodes)
    node_name = get_submodule_name(model, nodes)
    setattr(model, node_name, submodule)
    named_modules[node_name] = submodule
    with model.graph.inserting_after(nodes[-1]):
        new_node = model.graph.create_node(
            'call_module', node_name, new_args, {})
    nodes[-1].replace_all_uses_with(new_node)
    for node in reversed(nodes):
        model.graph.erase_node(node)
    new_node.meta['submodule'] = submodule
    if (dtype := nodes[-1].meta.get('dtype', None)) is not None:
        new_node.meta['dtype'] = dtype
    return new_node


def _nodes_sequential(nodes: List[Node], order: Dict[Node, int]):
    prev_node = nodes[0]
    for n in nodes[1:]:
        # Check if the current node is a user of the previous node
        if n not in prev_node.users:
            return False
        # Check if all the arguments of the current node are visited before the
        # previous node
        for arg in n.all_input_nodes:
            if id(arg) == id(prev_node):
                continue

            while is_nop(arg):
                arg = arg.all_input_nodes[0]

            if arg.op == "call_function" and order[arg] > order[prev_node]:
                return False
        prev_node = n
    return True


def find_sequential_nodes(model: GraphModule, pattern: List[List[Callable]]):
    graph = model.graph

    nodes_order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_fused: Dict[Node, None] = {}
    fused_nodes_list = []
    for wanted_sources in pattern:
        partitions = get_source_partitions(graph, wanted_sources)
        partitions = list(itertools.chain.from_iterable(partitions.values()))
        output_nodes = [p.output_nodes[0] for p in partitions]

        if len(output_nodes) == 0:
            continue

        if len(fused_nodes_list) == 0:
            fused_nodes_list = [[n] for n in output_nodes]
            continue

        fusion_candidates = []
        for nodes in fused_nodes_list:
            # If the last node in the group has multiple users, it cannot be fused
            last_node = nodes[-1]
            if len(last_node.users) > 1:
                fusion_candidates.append(nodes)
                continue

            # Include any NOP nodes after the last node in the group
            nop_nodes = []
            next_node = next(iter(last_node.users))
            while is_nop(next_node) and len(next_node.users) == 1:
                nop_nodes.append(next_node)
                next_node = next(iter(next_node.users))

            matched = False
            for node in output_nodes:
                if node in nodes_fused:
                    continue

                candidate = nodes + nop_nodes + [node]
                if _nodes_sequential(candidate, nodes_order):
                    matched = True
                    fusion_candidates.append(candidate)
                    for n in candidate:
                        nodes_fused[n] = None
                    if [node] in fused_nodes_list:
                        fused_nodes_list.remove([node])
                    if [node] in fusion_candidates:
                        fusion_candidates.remove([node])
                    output_nodes.remove(node)
                    break

            if not matched:
                fusion_candidates.append(nodes)

        fused_nodes_list = fusion_candidates + [[n] for n in output_nodes]

    return [fn for fn in fused_nodes_list if len(fn) > 1]


def is_tranpose(node: Node):
    """
    Transpose operations are characterized by swapping the last two dimensions
    """
    if node.target == torch.ops.aten.transpose.int:
        ndim = node.args[0].value.ndim
        axes = {x if x >= 0 else x + ndim for x in node.args[1:]}
        return (axes == {ndim - 2, ndim - 1})

    if node.target == torch.ops.aten.permute.default:
        permute_dims = node.args[1]
        tranpose_dims = list(range(len(permute_dims)))
        tranpose_dims[-2], tranpose_dims[-1] = tranpose_dims[-1], tranpose_dims[-2]
        return permute_dims == tranpose_dims

    return False


def is_mha_qkv_permute(node):
    """
    Check if the node is a permutation used in multi-head attention (MHA) operations.
    It has characteristics that last dimension is a power of 2 and the permuted
    dimensions are the middle two dimensions (2 and 3) of a 4D tensor.
    """
    import math

    # Don't support head dimension not being a power of 2
    if (
        not hasattr(node, 'shape') or
        len(node.shape) != 4 or
        not math.log2(node.shape[-1]).is_integer()
    ):
        return False

    if node.target == torch.ops.aten.permute.default:
        dims = node.args[1]
        return len(dims) == 4 and dims == [0, 2, 1, 3]

    if node.target == torch.ops.aten.transpose.int:
        dims = {x if x >= 0 else x + 4 for x in node.args[1:]}
        return node.value.ndim == 4 and dims == {1, 2}

    return False


def search_group(node, node_lists):
    for l in node_lists:
        if node in l:
            return l
    return None


def duplicate_shared_nodes(graph: torch.fx.Graph, nodes: List[Node]) -> List[Node]:
    """
    Ensures that nodes in the given list are independent by duplicating any node that has multiple users.

    This function processes the given list of nodes in topological order, identifying any node 
    with multiple users. If such a node exists, it is duplicated so that all nodes in the list 
    can be grouped together without affecting other nodes in the DAG.

    Args:
        graph (torch.fx.Graph): The FX graph being processed.
        nodes (List[Node]): A list of nodes to check for shared usage.

    Returns:
        List[Node]: A new list where shared nodes have been duplicated to ensure independence.
    """
    nodes_order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes = sorted(nodes, key=lambda n: nodes_order[n])

    for i in range(len(nodes) - 2, -1, -1):
        node = nodes[i]

        if len(node.users) == 1:
            continue

        user = nodes[i + 1]

        with graph.inserting_before(user):
            new_node = graph.node_copy(node, lambda n: n)

        user.replace_input_with(node, new_node)
        nodes[i] = new_node

       # Copy and update the metadata for tracking
        source_fn_st = node.meta.get('source_fn_stack', [])
        source_fn = source_fn_st[-1][1] if source_fn_st else new_node.target
        new_node.meta['source_fn_stack'] = [(new_node.name, source_fn)]

    return nodes

def _fuse_reshape_with_input_impl(
    graph: torch.fx.Graph,
    reshape_node: Node,
    current_node: Node = None,
    fused_nodes: List[Node] = None,
    candidates: List[List[Node]] = None,
    nodes_map: Dict[Node, Node] = None,
    simulate: bool = True
) -> Union[bool, List[Node]]:
    if current_node is None:
        current_node = reshape_node

    if fused_nodes is None:
        fused_nodes = []

    fused_nodes.append(current_node)

    # Check if fusion is valid
    has_no_tiled_shapes = "tiled_shapes" not in current_node.meta
    is_gemm_case = is_gemm_op(current_node) and (
        is_tranpose(reshape_node) or is_mha_qkv_permute(reshape_node)
    )
    is_elementwise_case = (
        is_elementwise_op(current_node) and not is_tranpose(reshape_node)
    )

    if has_no_tiled_shapes and (is_gemm_case or is_elementwise_case):
        if simulate:
            return True
        fused_nodes = duplicate_shared_nodes(graph, fused_nodes)
        if (group := search_group(current_node, candidates)) is not None:
            group.extend(n for n in fused_nodes if n not in group)
        else:
            candidates.append(fused_nodes)
        nodes_map[fused_nodes[0]] = fused_nodes[-2]
        return [fused_nodes[0]]

    is_select_after_tranpose = (
        is_tranpose(reshape_node)
        and current_node.target == torch.ops.aten.select.int
        and current_node.args[1] == 0
    )

    if (
        id(current_node) != id(reshape_node)
        and not is_nop(current_node)
        and not is_select_after_tranpose
    ):
        logger.info(f"Cannot fuse {reshape_node} with {current_node}")
        return False if simulate else []

    all_results = []
    for user in list(current_node.users):
        result = _fuse_reshape_with_input_impl(
            graph,
            reshape_node,
            current_node=user,
            fused_nodes=fused_nodes.copy(),
            candidates=candidates,
            nodes_map=nodes_map,
            simulate=simulate
        )
        if simulate:
            if not result:
                return False
        else:
            all_results.extend(result)

    return True if simulate else all_results


def fuse_reshape_with_input(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    reshape_node: Node
):
    # First pass: simulate fusion to ensure all users can be fused
    can_fuse_all = _fuse_reshape_with_input_impl(
        graph, reshape_node, simulate=True
    )

    if can_fuse_all:
        # Second pass: perform actual fusion
        return _fuse_reshape_with_input_impl(
            graph, reshape_node, simulate=False,
            candidates=candidates, nodes_map=nodes_map
        )
    else:
        logger.info(f"Skipping fusion for {reshape_node} due to unfusable path")
        return []


def fuse_reshape_with_output(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    reshape_node: Node
) -> bool:
    if not is_mha_qkv_permute(reshape_node):
        return False

    curr_node = reshape_node.all_input_nodes[0]
    fused_nodes = [reshape_node]

    while not (is_gemm_op(curr_node) or is_elementwise_op(curr_node)):
        if (
            len(curr_node.users) > 1
            or len(curr_node.all_input_nodes) != 1
            or not is_nop(curr_node)
        ):
            logger.debug(f"Cannot fuse {reshape_node} with {curr_node}")
            return False

        fused_nodes.insert(0, curr_node)
        curr_node = curr_node.all_input_nodes[0]

    def _is_tiled(n): 
        return "tiled_shapes" in getattr(n, "meta", {})

    if len(curr_node.users) > 1 or _is_tiled(curr_node):
        return False

    group = search_group(curr_node, candidates)

    if group is not None:
        if any(_is_tiled(n) for n in group):
            return False
        else:
            group.extend(n for n in fused_nodes if n not in group)
    else:
        candidates.append([curr_node, *fused_nodes])

    nodes_map[reshape_node] = curr_node

    return True


def move_transpose_after_select(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    transpose_node: Node,
):
    ndim = transpose_node.args[0].value.ndim
    dims = [x if x >= 0 else x + ndim for x in transpose_node.args[1:]]

    select_nodes = []

    curr_user = next(iter(transpose_node.users))
    while curr_user.target == torch.ops.aten.select.int and curr_user.args[1] == 0:
        select_nodes.append(curr_user)
        curr_user = next(iter(curr_user.users))
        dims = [x - 1 for x in dims]

    if not select_nodes:
        return False

    with graph.inserting_before(curr_user):
        new_node = graph.call_function(
            torch.ops.aten.transpose.int, (select_nodes[-1], *dims),
        )

    curr_user.replace_input_with(select_nodes[-1], new_node)
    select_nodes[0].replace_input_with(transpose_node, transpose_node.args[0])
    graph.erase_node(transpose_node)

    group = search_group(transpose_node, candidates)
    group.append(new_node)
    group.remove(transpose_node)

    # Propagate shape in the order of nodes appearing in graph
    for node in select_nodes:
        group.remove(node)
        propagate_shape(node)

    propagate_shape(new_node)

    mapped_node = nodes_map.pop(transpose_node, None)
    nodes_map[new_node] = (
        new_node if mapped_node == select_nodes[-1] else mapped_node
    )

    return True


def fuse_op_with_input(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    node_to_fuse: Node,
    current_node: Node = None,
    fused_nodes: Dict[Node, Node] = None,
):
    if current_node is None:
        current_node = node_to_fuse

    if fused_nodes is None:
        fused_nodes = []

    fused_nodes.append(current_node)

    if id(current_node) != id(node_to_fuse) and is_elementwise_op(current_node):
        # Only address generator 0 support slicing op
        group = search_group(current_node, candidates)
        if (
            node_to_fuse.target == torch.ops.aten.slice.Tensor and
            group is not None and
            current_node.prev in group
        ):
            logger.info(f"Cannot fuse {node_to_fuse} with {current_node}")
            return

        fused_nodes = duplicate_shared_nodes(graph, fused_nodes)
        if (group := search_group(current_node, candidates)) is not None:
            group.extend(n for n in fused_nodes if n not in group)
        else:
            candidates.append(fused_nodes)
        nodes_map[fused_nodes[0]] = fused_nodes[-2]
        return

    if id(current_node) != id(node_to_fuse) and not is_nop(current_node):
        logger.info(f"Cannot fuse {node_to_fuse} with {current_node}")
        return

    for user in list(current_node.users):
        fuse_op_with_input(
            graph, candidates, nodes_map, node_to_fuse, user, fused_nodes.copy()
        )


def fuse_operator(
    model: GraphModule,
    operations: List[List[Callable]] = None,
    fuse_reshape: bool = True
):
    """
    Fuse reshape, slicing, and dequantize operations with their immediate users.

    The logic for fusing reshape operations is that we first trace up the graph
    until we reach a GEMM or elementwise operation. If we encounter a reshape operation
    or a node that has either multiple inputs or users along the way, we stop and
    try to fuse the node with its immediate user. For fusing a node with its user,
    we trace down the graph until we reach a GEMM or elementwise operation. If we
    encounter a node with multiple users, we duplicate all the elements on the path
    and perform fusion on each branch.
    """
    graph = model.graph
    named_modules = dict(model.named_modules(remove_duplicate=False))
    named_buffers = dict(model.named_buffers())

    nodes_map = {}
    fused_nodes_list = []

    if operations is not None:
        fused_nodes_list = find_sequential_nodes(model, operations)

    if fuse_reshape:
        for node in list(graph.nodes):
            # Try to fuse MHA QKV permute with preceeding GEMM
            if fuse_reshape_with_output(
                graph, fused_nodes_list, nodes_map, node
            ):
                continue

            # Attempt to fuse it with its immediate user
            if is_reshape_op(node):
                fused_reshape_nodes = fuse_reshape_with_input(
                    graph, fused_nodes_list, nodes_map, node
                )

                for n in fused_reshape_nodes:
                    if n.target == torch.ops.aten.transpose.int:
                        move_transpose_after_select(
                            graph, fused_nodes_list, nodes_map, n
                        )

    for node in list(graph.nodes):
        if node.target not in [
            torch.ops.aten.slice.Tensor, torch.ops.aten.select.int
        ]:
            continue

        dim = node.args[1]
        ndim = node.args[0].value.ndim
        if dim == ndim - 1 or dim == -1:
            continue

        if (
            is_nop(node) or
            node.target == torch.ops.aten.select.int and
            all(d == 1 for d in node.args[0].shape[:node.args[1]])
        ):
            continue

        fuse_op_with_input(graph, fused_nodes_list, nodes_map, node)

    for node in list(graph.nodes):
        if node.target != torch.ops.quantized_ops.dequantize.default:
            continue

        # If the node is already fused, skip it
        if search_group(node, fused_nodes_list) is not None:
            continue

        fuse_op_with_input(graph, fused_nodes_list, nodes_map, node)

    # Fuse nodes that appear earlier in the graph first
    nodes_order = {node: idx for idx, node in enumerate(graph.nodes)}
    for nodes in fused_nodes_list:
        nodes.sort(key=lambda n: nodes_order[n])
    fused_nodes_list.sort(key=lambda fn: nodes_order[fn[-1]])

    nodes_map = {v.name: k.name for k, v in nodes_map.items()}

    for fused_nodes in fused_nodes_list:
        node = _create_and_insert_subgraph(fused_nodes, model, named_modules)
        gm = named_modules[node.target]

        for n in list(gm.graph.nodes):
            if (name := nodes_map.get(n.name, None)) is None:
                continue

            fused_node = next(iter(n for n in gm.graph.nodes if n.name == name))
            fused_node.meta['fused'] = True

            if is_reshape_op(fused_node):
                if next(iter(fused_node.users)).op == 'output':
                    node.meta['reshape'] = fused_node
                else:
                    n.meta['reshape'] = fused_node

            if is_indexing_or_concatenation_op(fused_node):
                n.meta['slicing'] = fused_node

            if fused_node.target == torch.ops.quantized_ops.dequantize.default:
                n.meta['dq_scale'] = named_buffers[fused_node.args[1].target]

            if next(iter(fused_node.users)).op != 'output':
                n.meta['input_node'] = fused_node.args[0]

        update_placeholder_meta(model, node)

        args = map_arg(node.args, lambda n: n.value)
        kwargs = map_arg(node.kwargs, lambda n: n.value)
        output = gm(*args, **kwargs)

        if isinstance(output, torch.Tensor):
            node.shape = output.shape
            node.value = output.cpu().clone()
        elif isinstance(output, (tuple, list)):
            node.value = [x.cpu().clone() for x in output]
        else:
            node.value = output

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def get_node_bytes(n: Node):
    if (dtype := n.meta.get('dtype', None)) is None:
        if isinstance(n.value, (list, tuple)):
            return (dtype_byte_size(t.dtype) for t in (n.value))
        else:
            return dtype_byte_size(n.value.dtype)

    if isinstance(dtype, (list, tuple)):
        dtypes = [
            t if t is not None else v.dtype for t, v in zip(dtype, n.value)
        ]
        return (dtype_byte_size(t) for t in dtypes)

    return dtype_byte_size(dtype if dtype is not None else n.value.dtype)


def get_tiled_tensor(arg, tiled_shapes=None):
    if not isinstance(arg, torch.fx.Node):
        return arg

    if tiled_shapes is None or arg not in tiled_shapes:
        return arg.value.clone()

    tensor = arg.value.clone()
    shape = tiled_shapes[arg]
    n = len(shape)
    slices = [slice(None)] * (tensor.ndim - n) + [slice(0, s) for s in shape]
    return tensor[tuple(slices)]


def conv2d_layout(shape, is_weight=False, do_transpose=False):
    assert len(shape) == 4, "Conv2d shape must be 4D"
    if not do_transpose:
        return shape
    if is_weight:
        return (shape[2], shape[3], shape[1], shape[0])
    else:
        return (shape[0], shape[2], shape[3], shape[1])


def normalize_shape(node, shape):
    from torch.fx.operator_schemas import normalize_function

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
    shape = {
        n: shape[k] for n, k in node_to_key.items() if k in shape
    }
    return shape


def run_fused_op_l2_tiling(
    node, module, tiled_shapes, allocator, unroll_dims, align_banks=True,
):
    from .utils import (
        get_arg_or_kwarg, get_valid_tiling, get_tiled_shape, construct_tiled_shape
    )

    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    first_node = next(n for n in module.graph.nodes if n.op == "call_function")

    if (
        not is_gemm_op(first_node) and
        not is_elementwise_op(first_node) and
        first_node.target not in [
            torch.ops.aten.softmax.int,
            torch.ops.aten.layer_norm.default,
            torch.ops.quantized_ops.calculate_mx_qparam.default,
            torch.ops.quantized_ops.quantize_mx.default,
        ]
    ):
        return tiled_shapes

    if not tiled_shapes:
        tiled_shapes = {n: n.value.shape for n in node.all_input_nodes}
        if isinstance(node.value, torch.Tensor):
            tiled_shapes[node] = node.value.shape
        else:
            tiled_shapes[node] = tuple(t.shape for t in node.value)
    else:
        tiled_shapes = {
            n: k for n, k in tiled_shapes.items()
            if n in node.all_input_nodes or n == node
        }

    is_gemm = is_gemm_op(first_node)
    min_sizes = None
    transposed = first_node.meta.get("transposed", False)

    if is_gemm:
        args = map_arg(node.args, lambda n: get_tiled_tensor(n, tiled_shapes))
        ShapeProp(module).propagate(*args)
        output_shape = first_node.value.shape

        # We are not doing tiling on Y, X and C dimensions for conv layers here
        if is_conv2d(first_node):
            if transposed:
                min_sizes = output_shape[:-1] + (unroll_dims[0],)
            else:
                min_sizes = (unroll_dims[0],) + output_shape[2:]
        else:
            min_sizes = (unroll_dims[0],)
    elif isinstance(node.value, torch.Tensor):
        output_shape = tiled_shapes[node]
    else:
        output_shape = tiled_shapes[node][1]

    tilings = get_valid_tiling(output_shape, reverse=is_gemm, min_sizes=min_sizes)

    for tiled_output_shape, tiling in tilings:
        new_shapes = {}

        if is_gemm:
            bs = first_node.kwargs.get("block_size", 1)
            if is_conv2d(first_node):
                N, tile_y, tile_x, tile_k = conv2d_layout(
                    tiled_output_shape, False, not transposed
                )
                kH, kW, _, tile_c = conv2d_layout(
                    first_node.args[1].shape, True, not transposed
                )

                stride = get_arg_or_kwarg(first_node, 3, "stride", (1, 1))
                dilation = get_arg_or_kwarg(first_node, 5, "dilation", (1, 1))

                tile_iy = (tile_y - 1) * stride[0] + (kH - 1) * dilation[0] + 1
                tile_ix = (tile_x - 1) * stride[1] + (kW - 1) * dilation[1] + 1

                new_shapes = {
                    "input": (N, tile_c, tile_iy, tile_ix),
                    "weight": (tile_k, tile_c, kH, kW),
                    "bias": (tile_k,),
                    "input_scale": (N, tile_c // bs, tile_iy, tile_ix),
                    "weight_scale": (tile_k, tile_c // bs, kH, kW),
                }

                new_shapes = {
                    k: conv2d_layout(v, "weight" in k, transposed) if k != "bias" else v
                    for k, v in new_shapes.items()
                }
            else:
                x_tiled = math.prod(tiled_output_shape[:-1])
                k_tiled = tiled_output_shape[-1]
                c_tiled = first_node.args[0].value.shape[-1]

                input_value = first_node.args[0].value
                tiled_input_shape = construct_tiled_shape(
                    input_value.shape, x_tiled, list(range(input_value.ndim))[:-1]
                )

                weight_transposed = is_matmul(first_node) or transposed
                weight_shape = (
                    (c_tiled, k_tiled) if weight_transposed else (k_tiled, c_tiled)
                )
                weight_scale_shape = (
                    (c_tiled // bs, k_tiled) if weight_transposed
                    else (k_tiled, c_tiled // bs)
                )

                weight_key = "other" if is_matmul(first_node) else "weight"

                new_shapes = {
                    "input": tiled_input_shape[:-1] + (c_tiled,),
                    weight_key: weight_shape,
                    "bias": (k_tiled,),
                    "input_scale": tiled_input_shape[:-1] + (c_tiled // bs,),
                    "weight_scale": weight_scale_shape,
                }

                tiling = (math.prod(tiling[:-1]), tiling[-1])

            new_shapes = normalize_shape(first_node, new_shapes)

        for n in node.all_input_nodes:
            if n not in new_shapes and not n.name.startswith("code") and n.value.numel() > 1:
                new_shapes[n] = get_tiled_shape(tuple(tiled_shapes[n]), tiling)

        if isinstance(node.value, (tuple, list)):
            new_shapes[node] = [get_tiled_shape(s, tiling) for s in tiled_shapes[node]]
        else:
            assert isinstance(node.value, torch.Tensor)
            new_shapes[node] = tiled_output_shape

        total_size = sum(
            allocator.get_tensor_size(n, s, align_banks, n == node)
            for n, s in new_shapes.items() if n in node.all_input_nodes or n == node
        )

        if total_size <= allocator.total_memory:
            if (orig_tiling := first_node.meta.get("l2_tiling")) is not None:
                assert len(orig_tiling) == len(tiling), (
                    f"Original tiling {orig_tiling} and new tiling {tiling} "
                    "have different ranks"
                )
                tiling = tuple(a * b for a, b in zip(orig_tiling, tiling))

            if math.prod(tiling) == 1:
                return {}

            first_node.meta["l2_tiling"] = tiling
            return new_shapes

    logger.warning(f"Failed to adjust tiling for {node}")
    return {}


def get_tiled_input_shape(shape1, shape2):
    """
    Given:
      - shape1: tiled shape of tensor A
      - shape2: original shape of tensor B (broadcastable to the original shape
                of tensor A)

    Returns:
      - shape2_tiled: shape for B that is broadcastable to the tiled shape of A
    """
    # Align shape2 to the same number of dims as new_shape1
    shape2 = list(shape2)
    ndiff = len(shape1) - len(shape2)
    # For each dim, if shape2 dim is 1 or matches, keep it, else set to 1
    shape2_tiled = []
    for i in range(len(shape2)):
        shape2_tiled.append(
            shape1[i + ndiff] if i + ndiff >= 0 and shape2[i] != 1 else shape2[i]
        )
    return tuple(shape2_tiled)


def run_memory_mapping(
    model: GraphModule,
    allocator: MemoryAllocator = None,
    cache_size: int = None,
    num_banks: int = None,
    bank_width: int = None,
    unroll_dims=None
):
    graph = model.graph
    named_modules = dict(model.named_modules(remove_duplicate=False))

    if allocator is None:
        allocator = MemoryAllocator(DEFAULT_MEMORY_SIZE)

    # Store all the weights in memory if persistent is enabled
    for node in model.graph.nodes:
        if node.op == "get_attr" and not re.fullmatch(r"code_\d+", node.name):
            node.meta["memory"] = allocator.allocate_memory(node)

    # Store inputs to the model in memory
    for node in model.graph.nodes:
        if node.op == "placeholder":
            node.meta["memory"] = allocator.allocate_memory(node)

    allocator.snapshot()

    eliminate_dead_code(model.graph)

    # Run through reverse nodes and record the first instance of a use
    # of a given node. This represents the *last* use of the node in the
    # execution order of the program, which we will use to free unused
    # values
    node_to_last_use : Dict[Node, Node] = {}
    user_to_last_uses : Dict[Node, List[Node]] = {}

    def register_last_uses(n: Node, user: Node):
        if n not in node_to_last_use and n.op != "get_attr":
            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)

            if (
                is_nop(n) or
                is_indexing_or_concatenation_op(n) or
                n.target == operator.getitem
            ):
                for arg in n.all_input_nodes:
                    register_last_uses(arg, user)

    for node in reversed(model.graph.nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    def get_unused_values(user: Node):
        """
        Delete values after their last use. This ensures that values that are
        not used in the remainder of the code are freed and the memory usage
        of the code is optimal.
        """
        nodes_to_delete = user_to_last_uses.get(user, [])
        return nodes_to_delete

    def get_path_to_target(node: torch.fx.Node, targets):
        if not isinstance(targets, (list, tuple)):
            targets = [targets]

        for user in node.users:
            if user.target in targets:
                return [node, user]

            if is_nop(user) or is_addressing_op(user):
                path = get_path_to_target(user, targets)
                if path is not None:
                    return [node] + path
        return None

    def allocate_scratchpad(node: Node):
        from .utils import get_tiled_shape

        if cache_size is None:
            return

        bank_size = cache_size // num_banks if num_banks is not None else None
        sp_allocator = MemoryAllocator(
            cache_size, bank_width, bank_size, MemorySpace.SCRATCHPAD
        )

        if node.op == "call_module":
            mod = named_modules[node.target]
            first_node = next(iter(n for n in mod.graph.nodes if n.op == "call_function"))
        else:
            first_node = node

        tiled_shapes = normalize_shape(
            first_node, first_node.meta.get("tiled_shapes", {})
        )

        # Calculate tiled shape for other input/output nodes
        if node.op == "call_module" and tiled_shapes:
            output_shape = tiled_shapes[first_node]
            for n in node.all_input_nodes:
                if n not in tiled_shapes and not n.name.startswith("code"):
                    tiled_shapes[n] = get_tiled_input_shape(output_shape, n.value.shape)

            l2_tiling = first_node.meta.get("l2_tiling")
            if isinstance(node.value, torch.Tensor):
                tiled_shapes[node] = get_tiled_shape(node.shape, l2_tiling)
            else:
                tiled_shapes[node] = tuple(
                    get_tiled_shape(t.shape, l2_tiling) for t in node.value
                )

        if node.op == "call_module":
            mod = named_modules[node.target]
            new_shapes = run_fused_op_l2_tiling(
                node, mod, tiled_shapes, sp_allocator, unroll_dims
            )

            if not new_shapes:
                new_shapes = run_fused_op_l2_tiling(
                    node, mod, tiled_shapes, sp_allocator, unroll_dims, False
                )

            tiled_shapes = new_shapes

            for n in list(mod.graph.nodes):
                if n != first_node:
                    n.meta.pop("l2_tiling", None)

        if tiled_shapes:
            node.meta["tiled_shapes"] = tiled_shapes

        tensor_sizes = {
            n: sp_allocator.get_tensor_size(n, tiled_shapes.get(n))
            for n in node.all_input_nodes
            if not n.name.startswith("code") and (
                isinstance(n.value, torch.Tensor) and n.value.numel() > 1
                or n.op != "get_attr"
            )
        }

        if isinstance(node.value, torch.Tensor):
            tensor_sizes[node] = sp_allocator.get_tensor_size(
                node, tiled_shapes.get(node), is_scratch_output=True
            )
        elif isinstance(node.value, (tuple, list)):
            output_shapes = tiled_shapes.get(node, [tuple(t.shape) for t in node.value])
            tensor_sizes[node] = sp_allocator.get_tensor_size(
                node, output_shapes, is_scratch_output=True
            )

        tensor_sizes = dict(sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True))

        bytes_to_allocate = sum(tensor_sizes.values())
        remaining_cache_size = cache_size

        scratchpad_mem = {}
        unaligned_tensors = []

        # Allocate large tensors first for better cache utilization
        for n, size in tensor_sizes.items():
            aligned_size = sp_allocator.align_size(size, True)

            bytes_to_allocate -= size
            align_bank = bytes_to_allocate <= remaining_cache_size - aligned_size
            remaining_cache_size -= aligned_size if align_bank else size

            if not align_bank:
                unaligned_tensors.append(n.name)

            scratchpad_mem[n] = sp_allocator.allocate_memory(
                n,
                shape=tiled_shapes.get(n),
                align_bank=align_bank,
                expand_on_failure=True,
                is_scratch_output=(n == node),
            )

        if sp_allocator.total_memory != cache_size:
            logger.warning(
                f"[MEM_ALLOC_FAIL] {node}: expanding cache size from "
                f"{cache_size} to {sp_allocator.total_memory}."
            )

        if unaligned_tensors:
            names = ', '.join(unaligned_tensors)
            logger.warning(
                f"[BANK_ASSIGN_FAIL] {node}: tensors {names} could not "
                f"be assigned to individual banks"
            )

        node.meta["scratchpad_mem"] = scratchpad_mem

    def allocate_for_stack_op(node: Node):
        """
        For stacked layers, place them next to each other so that we can read
        them using a single memory access in the next operation
        """
        nodes = get_path_to_target(
            node, [torch.ops.aten.stack.default, torch.ops.aten.cat.default]
        )

        if nodes is not None:
            # TODO: should not duplicate complex ops like conv2d
            nodes = duplicate_shared_nodes(graph, nodes)
            for n in nodes:
                propagate_shape(n, model)

            stack_node = nodes[-1]
            if (memory := stack_node.meta.get("memory", None)) is None:
                memory = allocate_for_stack_op(stack_node)

            tensor_sizes = [n.value.numel() * get_node_bytes(n) for n in stack_node.args[0]]

            index = stack_node.args[0].index(nodes[-2])
            start_offset = memory.start + sum(tensor_sizes[:index])
            size = tensor_sizes[index]
            segment = Segment(start_offset, start_offset + size, allocator.memory_space)

            for n in reversed(nodes[:-1]):
                n.meta["memory"] = segment
                allocate_scratchpad(n)

            # If the first node is a param node, we need to copy it to the new location
            input_node = node.all_input_nodes[0]
            if is_nop(node) and input_node.meta["memory"].start != segment.start:
                with graph.inserting_before(node):
                    copy_node = graph.call_function(
                        torch.ops.aten.add.Scalar, (input_node, 0)
                    )

                node.replace_input_with(input_node, copy_node)
                register_last_uses(copy_node, node)

                copy_node.meta["memory"] = segment
                copy_node.value, copy_node.shape = input_node.value, input_node.shape

                allocate_scratchpad(copy_node)
        elif node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            node.meta["memory"] = allocator.allocate_memory(node)

        return node.meta.get("memory")

    for node in list(model.graph.nodes):
        if node.op not in ["call_function", "call_module"]:
            continue

        if "memory" in node.meta:
            continue

        skip_allocation = False

        # Propagate memory metadata for nop nodes
        if is_nop(node):
            assert "memory" in node.args[0].meta, (
                f"Node {node} does not have memory metadata, "
            )
            node.meta["memory"] = copy.deepcopy(node.args[0].meta["memory"])
            skip_allocation = True

        if node.target == operator.getitem:
            input_node = node.args[0]
            output_sizes = input_node.meta["output_sizes"]
            start_offset = input_node.meta["memory"].start + sum(output_sizes[:node.args[1]])
            size = output_sizes[node.args[1]]
            node.meta["memory"] = Segment(start_offset, start_offset + size, allocator.memory_space)
            skip_allocation = True

        # We do not allocate new memory for select operations. Instead, calculate
        # the memory offset from the select index
        # TODO: Fuse select operation with its user if possible. Do not handle it here
        if (
            node.target == torch.ops.aten.select.int and
            all(d == 1 for d in node.args[0].value.shape[:node.args[1]])
        ):
            size = node.value.numel() * get_node_bytes(node)
            start_offset = node.args[0].meta["memory"].start + node.args[2] * size
            node.meta["memory"] = Segment(start_offset, start_offset + size, allocator.memory_space)
            skip_allocation = True

        # We use the partition of the first input tensor since it preallocates
        # memory for all the tensors in the stack operation
        if node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            assert len(node.args) != 1 and node.args[1] != 0, (
                f"Only support stacking along the first dimension, got {node.args[1]} for {node}"
            )
            continue

        allocate_for_stack_op(node)

        if skip_allocation:
            continue

        if node.meta.get("memory") is None:
            node.meta["memory"] = allocator.allocate_memory(node)
            allocator.snapshot()

        for n in get_unused_values(node):
            allocator.free_memory(n)

        if node.meta.get("scratchpad_mem") is None:
            allocate_scratchpad(node)


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    named_modules = dict(model.named_modules(remove_duplicate=False))

    ShapeProp(model).propagate(*args)
    model_params = Model()

    for node in model.graph.nodes:
        node_value = getattr(node, 'value', None)
        if not isinstance(node_value, (torch.Tensor, tuple, list)):
            continue

        op = Operation()

        if node.op == 'placeholder':
            tensor = Tensor()
            set_tensor_field(tensor, node, output_dir)
            model_params.inputs.append(tensor)
            continue
        elif node.op == 'get_attr':
            tensor = Tensor()
            set_tensor_field(tensor, node, output_dir)
            if "memory" in node.meta:
                model_params.parameters.append(tensor)
            continue
        elif node.op == 'call_function':
            op.op.CopyFrom(map_node(node))
        elif node.op == 'call_module':
            gm = named_modules[node.target]
            assert isinstance(gm, torch.fx.GraphModule)

            if (tiled_shapes := node.meta.get('tiled_shapes')):
                args = map_arg(node.args, lambda n: get_tiled_tensor(n, tiled_shapes))
                ShapeProp(gm).propagate(*args)

                for n in gm.graph.nodes:
                    if hasattr(n, "shape"):
                        tiled_shapes.setdefault(n, n.shape)

            args = map_arg(node.args, lambda n: n.value.clone())
            ShapeProp(gm).propagate(*args)

            scratchpad_mem = node.meta.get("scratchpad_mem")

            operators = []
            for n in gm.graph.nodes:
                if n.op != 'call_function' or n.meta.get('fused', False) or is_nop(n):
                    continue

                n.meta["tiled_shapes"] = tiled_shapes
                n.meta["scratchpad_mem"] = scratchpad_mem

                operators.append(map_node(n))

            op.fused_op.name = node.name
            op.fused_op.op_list.extend(operators)
        else:
            continue

        set_output_field(op, node, output_dir)

        model_params.ops.append(op)

    return model_params


def gen_compute_graph(model, output_file="compute_graph", max_users=10):
    nodes = {}
    edges = []
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.op == "get_attr" and node.name.startswith("code_"):
            continue

        header = node.name

        if isinstance(node.value, torch.Tensor):
            header += f"&#92;n{str(tuple(node.value.shape))}"
            if (dtype := node.meta.get("dtype", None)) is not None:
                header += f"&#92;n{dtype}"
        elif isinstance(node.value, (tuple, list)):
            shape_str = ", ".join([str(tuple(t.shape)) for t in node.value])
            header += f"&#92;n{shape_str}"

            dtypes = [t.dtype for t in node.value]
            if (dtype := node.meta.get("dtype", None)) is not None:
                dtypes = [dt or dtypes[i] for i, dt in enumerate(dtype)]

            if any(dtype not in [torch.float, torch.bfloat16] for dtype in dtypes):
                header += f"&#92;n{', '.join([str(d) for d in dtypes])}"
        else:
            continue

        body = None
        if node.op == "call_module":
            gm = named_modules[node.target]
            if isinstance(gm, torch.fx.GraphModule):
                body = "&#92;n".join([
                    n.name for n in gm.graph.nodes if n.op == "call_function"
                ])
        label = f"{{{header}}}" if body is None else f"{{{header}|{body}}}"
        label = label.replace("<", "\<").replace(">", "\>")

        nodes[node.name] = {
            "label": label,
            "shape": "Mrecord",
        }

        users = list(node.users)
        num_users = len(users)
        if num_users > max_users:
            num_splits = (num_users + max_users - 1) // max_users
            for i in range(num_splits):
                sub_node = f"{node.name}_split_{i}"
                sub_label = f"{{{sub_node}}}"
                sub_label = sub_label.replace("<", "\<").replace(">", "\>")

                # Create a sub-node for this group of users
                nodes[sub_node] = {
                    "label": sub_label,
                    "shape": "Mrecord",
                }

                edges.append((node.name, sub_node))

                # Add edges from sub-node to its users
                start_idx = i * max_users
                end_idx = min(start_idx + max_users, num_users)
                for u in users[start_idx:end_idx]:
                    edges.append((sub_node, u.name))
        else:
            for u in users:
                edges.append((node.name, u.name))

    g = graphviz.Digraph()
    # g.attr(bgcolor="transparent")

    for node, attrs in nodes.items():
        g.node(node, **attrs)

    g.edges(edges)

    g.render(output_file, format='svg', cleanup=True)
