import copy
import itertools
import logging
import operator
import os
from collections import defaultdict
from typing import List, Dict, Callable

import graphviz
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx import Node, Graph, GraphModule
from torch.fx.node import map_arg
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .mapping_utils import (
    _is_elementwise_op,
    _is_gemm_op,
    _is_indexing_or_concatenation_op,
    _is_nop,
    _is_reshape_op,
    map_node,
    set_output_field,
    set_tensor_field,
)
from .memory import MemoryManager, Partition
from .param_pb2 import Model, Operation, Tensor
from .shape_prop import ShapeProp
from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_SIZE = torch.finfo(torch.float32).max


def graph_copy(self: GraphModule, graph: Graph, source: Node) -> List[Node]:
    args_iter = iter(source.all_input_nodes)
    value_remap = {}
    for node in list(graph.nodes):
        if node.op == 'placeholder':
            value_remap[node] = next(args_iter)
        elif node.op == 'output':
            source.replace_all_uses_with(value_remap[node.args[0][0]])
        else:
            with self.graph.inserting_before(source):
                value_remap[node] = self.graph.node_copy(node, lambda n: value_remap[n])

            if (source_fn_st := node.meta.get('source_fn_stack', None)) is not None:
                source_fn = source_fn_st[-1]
                value_remap[node].meta['source_fn_stack'] = [
                    (value_remap[node].name, source_fn[1])
                ]

    output_node = list(graph.nodes)[-1]
    return [value_remap[n] for n in output_node.args[0]]


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

    gm = capture_pre_autograd_graph(BMM(), (input1, input2))
    output_nodes = graph_copy(model, gm.graph, node)
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

    gm = capture_pre_autograd_graph(BMM(), (input1, input2), kwargs)

    # Remove unused placeholder nodes
    for n in gm.graph.nodes:
        if n.op == 'placeholder' and len(n.users) == 0:
            gm.graph.erase_node(n)
    gm.graph.lint()

    source_fn = node.meta['source_fn_stack'][-1]
    for n in gm.graph.nodes:
        if n.target == torch.ops.quantized_ops.matmul_mx.default:
            n.meta['source_fn_stack'] = [(n.name, source_fn[1])]

    output_nodes = graph_copy(model, gm.graph, node)
    model.graph.erase_node(node)
    return output_nodes[0]


def split_multi_head_attention(model: GraphModule):
    graph = model.graph
    partitions = get_source_partitions(graph, [torch.matmul])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    grouped_nodes = defaultdict(list)
    for partition in partitions:
        matmul_node = partition.output_nodes[0]
        if (nn_module_stack := matmul_node.meta.get('nn_module_stack', None)) is None:
            continue

        bt = list(nn_module_stack.values())[-1]
        grouped_nodes[bt[0]].append(matmul_node)

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
        def dfs(current_node, visited):
            if current_node == av_matmul:
                return [visited]
            paths = []
            for user in current_node.users:
                if user not in visited:
                    paths.extend(dfs(user, visited + [user]))
            return paths

        paths = dfs(qk_matmul, [qk_matmul])

        nodes_between = set()
        for path in paths:
            nodes_between.update(path[1:-1])

        # Sort the nodes between the qk and av matmuls
        order = {node: idx for idx, node in enumerate(graph.nodes)}
        nodes_between = sorted(nodes_between, key=lambda n: order[n])

        # Decompose BMM into multiple matmuls
        if qk_matmul.target == torch.ops.aten.matmul.default:
            qk_output = _decompose_bmm(model, qk_matmul)
            av_output = _decompose_bmm(model, av_matmul)
        else:
            qk_output = _decompose_bmm_mx(model, qk_matmul)
            av_output = _decompose_bmm_mx(model, av_matmul)

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

        nodes_between[0].replace_input_with(qk_output, qk_matmuls[0])
        av_matmuls[0].replace_input_with(av_matmuls[0].args[0], nodes_between[-1])

        if (scale_node := av_matmuls[0].kwargs.get('input_scale', None)) is not None:
            av_matmuls[0].replace_input_with(scale_node, nodes_between[-2])

        for qk_matmul, av_matmul in zip(qk_matmuls[1:], av_matmuls[1:]):
            value_remap = {qk_matmuls[0]: qk_matmul}
            for node in nodes_between:
                with graph.inserting_before(av_matmul):
                    value_remap[node] = graph.node_copy(node, lambda n: value_remap.get(n, n))

                if (source_fn_st := node.meta.get('source_fn_stack', None)) is not None:
                    source_fn = source_fn_st[-1]
                    value_remap[node].meta['source_fn_stack'] = [
                        (value_remap[node].name, source_fn[1])
                    ]

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


def _create_and_insert_subgraph(
    nodes: List[Node],
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module]
) -> Node:
    submodule, new_args = _create_subgraph(nodes)
    get_new_node_name = get_new_attr_name_with_prefix('submodule_')
    node_name = get_new_node_name(model)
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

            while _is_nop(arg):
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

        if len(fused_nodes_list) == 0:
            fused_nodes_list = [[p.output_nodes[0]] for p in partitions]
            continue

        if len(partitions) == 0:
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
            while _is_nop(next_node) and len(next_node.users) == 1:
                nop_nodes.append(next_node)
                next_node = next(iter(next_node.users))

            matched = False
            for p in partitions:
                output_node = p.output_nodes[0]
                if output_node in nodes_fused:
                    continue

                candidate = nodes + nop_nodes + [output_node]
                if _nodes_sequential(candidate, nodes_order):
                    matched = True
                    fusion_candidates.append(candidate)
                    for n in candidate:
                        nodes_fused[n] = None
                    if [output_node] in fused_nodes_list:
                        fused_nodes_list.remove([output_node])
                    break

            if matched:
                partitions.remove(p)
            else:
                fusion_candidates.append(nodes)

        fused_nodes_list = fusion_candidates + [[p.output_nodes[0]] for p in partitions]

    return [fn for fn in fused_nodes_list if len(fn) > 1]


def is_tranpose(node: Node):
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
        source_fn = source_fn_st[-1] if source_fn_st else new_node.target
        new_node.meta['source_fn_stack'] = [(new_node.name, source_fn[1])]

    return nodes


def fuse_reshape_with_input(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    reshape_node: Node,
    current_node: Node = None,
    fused_nodes: Dict[Node, Node] = None,
):
    """
    Recursively fuses a reshape node with its immediate user that is not a NOP.
    
    This function traverses the graph downward from the reshape node, collecting nodes along the path.
    If the path ends in a GEMM or elementwise operation, the nodes are added to a fusion group.
    
    Args:
        graph (torch.fx.Graph): The FX graph being processed.
        candidates (List[List[Node]]): List of fusion groups.
        nodes_map (Dict[Node, Node]): A mapping from nodes to be fused to their immediate user.
        reshape_node (Node): The reshape node to be fused.
        node (Node, optional): The current node being processed. Defaults to `reshape_node`.
        fused_nodes (List[Node], optional): List of nodes being fused in the current path. Defaults to an empty list.
    
    Returns:
        List[Node]: List of reshape nodes that were successfully fused.
    """
    if current_node is None:
        current_node = reshape_node

    if fused_nodes is None:
        fused_nodes = []

    fused_nodes.append(current_node)

    # Base case: Stop if the node is a GEMM or an elementwise operation
    if (
        _is_gemm_op(current_node) or
        (not is_tranpose(reshape_node) and _is_elementwise_op(current_node))
    ):
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

    # Reshape can be fused with aten.select.int only if the select index is 0
    if (
        id(current_node) != id(reshape_node)
        and not _is_nop(current_node)
        and not is_select_after_tranpose
    ):
        logger.warning(f"Cannot fuse {reshape_node} with {current_node}")
        return []

    # If there's a branching point, perform fusion on each branch
    # TODO is it worth performing fusion if there are more than one user?
    all_reshape_nodes = []
    for user in list(current_node.users):
        nodes = fuse_reshape_with_input(
            graph, candidates, nodes_map, reshape_node, user, fused_nodes.copy()
        )
        all_reshape_nodes.extend(nodes)
    return all_reshape_nodes


def swap_tranpose_and_select(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    node: Node,
):
    ndim = node.args[0].value.ndim
    dims = [x if x >= 0 else x + ndim for x in node.args[1:]]

    select_nodes = []

    user_node = next(iter(node.users))
    while (
        user_node.target == torch.ops.aten.select.int
        and user_node.args[1] == 0
    ):
        select_nodes.append(user_node)
        user_node = next(iter(user_node.users))
        dims = [x - 1 for x in dims]

    if len(select_nodes) > 0:
        with graph.inserting_before(user_node):
            new_node = graph.call_function(
                torch.ops.aten.transpose.int, (select_nodes[-1], *dims),
            )

        user_node.replace_input_with(select_nodes[-1], new_node)
        select_nodes[0].replace_input_with(node, node.args[0])
        graph.erase_node(node)

        group = search_group(node, candidates)

        group.remove(node)
        for n in select_nodes:
            group.remove(n)
        group.append(new_node)

        user_node = nodes_map.pop(node, None)
        nodes_map[new_node] = (
            new_node if user_node == select_nodes[-1] else user_node
        )


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

    if id(current_node) != id(node_to_fuse) and _is_elementwise_op(current_node):
        fused_nodes = duplicate_shared_nodes(graph, fused_nodes)
        if (group := search_group(current_node, candidates)) is not None:
            group.extend(n for n in fused_nodes if n not in group)
        else:
            candidates.append(fused_nodes)
        nodes_map[fused_nodes[0]] = fused_nodes[-2]
        return

    if id(current_node) != id(node_to_fuse) and not _is_nop(current_node):
        logger.warning(f"Cannot fuse {node_to_fuse} with {current_node}")
        return

    for user in list(current_node.users):
        fuse_op_with_input(
            graph, candidates, nodes_map, node_to_fuse, user, fused_nodes.copy()
        )


def fuse_operator(model: GraphModule, mapping=None):
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
    if mapping is None:
        mapping = {}

    nodes_map = {}

    graph = model.graph
    named_modules = dict(model.named_modules(remove_duplicate=False))

    fused_nodes_list = find_sequential_nodes(model, mapping.values())

    partitions = get_source_partitions(graph, ['permute', 'transpose'])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        reshape_node = partitions.pop(0).output_nodes[0]

        if not is_tranpose(reshape_node):
            match_found = True
            fused_nodes = [reshape_node]
            input_node = reshape_node.all_input_nodes[0]

            while not _is_gemm_op(input_node) and not _is_elementwise_op(input_node):
                if not _is_nop(input_node):
                    logger.warning(f"Cannot fuse {reshape_node} with {input_node}")
                    match_found = False
                    break

                # Reshape cannot be fused with a node that has multiple users or inputs
                if (
                    len(input_node.users) > 1 or
                    len(input_node.all_input_nodes) != 1 or
                    _is_reshape_op(input_node)
                ):
                    match_found = False
                    break

                fused_nodes.insert(0, input_node)
                input_node = input_node.all_input_nodes[0]

            if match_found and len(input_node.users) == 1:
                nodes_map[reshape_node] = input_node
                if (group := search_group(input_node, fused_nodes_list)) is not None:
                    group.extend(n for n in fused_nodes if n not in group)
                else:
                    fused_nodes.insert(0, input_node)
                    fused_nodes_list.append(fused_nodes)
                continue

        nodes = fuse_reshape_with_input(graph, fused_nodes_list, nodes_map, reshape_node)

        for n in nodes:
            if n.target == torch.ops.aten.transpose.int:
                swap_tranpose_and_select(graph, fused_nodes_list, nodes_map, n)

    partitions = get_source_partitions(graph, [operator.getitem])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        slice_node = partitions.pop(0).output_nodes[0]

        if slice_node.target not in [torch.ops.aten.slice.Tensor, torch.ops.aten.select.int]:
            logger.warning(f"Unrecognized getitem operation: {slice_node.target}")
            continue

        if _is_nop(slice_node) or (
            slice_node.target == torch.ops.aten.select.int and slice_node.args[1] == 0
        ):
            logger.info(f"Node {slice_node} is a nop or can be handled by memory planner.")
            continue

        fuse_op_with_input(graph, fused_nodes_list, nodes_map, slice_node)

    partitions = get_source_partitions(graph, [torch.ops.quantized_ops.dequantize.default])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        dequantized_node = partitions.pop(0).output_nodes[0]

        # If the node is already fused, skip it
        if search_group(dequantized_node, fused_nodes_list) is not None:
            continue

        fuse_op_with_input(graph, fused_nodes_list, nodes_map, dequantized_node)

    # Fuse nodes that appear earlier in the graph first
    nodes_order = {node: idx for idx, node in enumerate(graph.nodes)}
    for nodes in fused_nodes_list:
        nodes.sort(key=lambda n: nodes_order[n])
    fused_nodes_list.sort(key=lambda fn: nodes_order[fn[-1]])

    nodes_map = {v.name: k.name for k, v in nodes_map.items()}

    for fused_nodes in fused_nodes_list:
        node = _create_and_insert_subgraph(fused_nodes, model, named_modules)
        named_modules = dict(model.named_modules(remove_duplicate=False))
        gm = named_modules[node.target]

        buffers = dict(model.named_buffers())

        nodes = list(gm.graph.nodes)
        for n in nodes:
            if (name := nodes_map.get(n.name, None)) is None:
                continue
            source_node = next(iter(n for n in nodes if n.name == name))

            # A small trick to skip generating the param for this node
            source_node.meta['fused'] = True

            if _is_reshape_op(source_node):
                if next(iter(source_node.users)).op == 'output':
                    node.meta['reshape'] = source_node
                else:
                    n.meta['reshape'] = source_node
            elif _is_indexing_or_concatenation_op(source_node):
                n.meta['slicing'] = source_node
            elif source_node.target == torch.ops.quantized_ops.dequantize:
                # get_attr_node is a placeholder node. We assume that the attribute
                # name is equal to the placeholder name.
                get_attr_node = source_node.args[1]
                n.meta['dq_scale'] = buffers[get_attr_node.target]

    graph.lint()

    graph.eliminate_dead_code()
    model.recompile()

    return model


def allocate_weights(model: GraphModule, manager: MemoryManager = None):
    if manager is None:
        manager = MemoryManager(DEFAULT_MEMORY_SIZE)

    for node in model.graph.nodes:
        if node.op == "get_attr" and "code" not in node.name:
            node.meta["memory"] = manager.allocate_memory(node)


def allocate_activations(model: GraphModule, manager: MemoryManager = None):
    if manager is None:
        manager = MemoryManager(DEFAULT_MEMORY_SIZE)

    for node in model.graph.nodes:
        if node.op == "placeholder":
            node.meta["memory"] = manager.allocate_memory(node)

    manager.take_snapshot()

    # Run through reverse nodes and record the first instance of a use
    # of a given node. This represents the *last* use of the node in the
    # execution order of the program, which we will use to free unused
    # values
    node_to_last_use : Dict[Node, Node] = {}
    user_to_last_uses : Dict[Node, List[Node]] = {}

    def register_last_uses(n: Node, user: Node):
        if n.op != 'get_attr' and n not in node_to_last_use:
            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)

    for node in reversed(model.graph.nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    def delete_unused_values(user: Node):
        """
        Delete values after their last use. This ensures that values that are
        not used in the remainder of the code are freed and the memory usage
        of the code is optimal.
        """
        nodes_to_delete = user_to_last_uses.get(user, [])
        for n in list(nodes_to_delete):
            if _is_nop(n) or _is_indexing_or_concatenation_op(n) or n.target == operator.getitem:
                nodes_to_delete.extend(delete_unused_values(n))
        return nodes_to_delete
    
    def get_num_bytes(n: Node):
        if n.meta.get('dtype', None) is not None:
            return dtype_byte_size(n.meta['dtype'])
        else:
            return dtype_byte_size(n.value.dtype)

    for node in model.graph.nodes:
        if node.op not in ["call_function", "call_module"]:
            continue

        # Propagate memory metadata for nop nodes
        if _is_nop(node):
            node.meta["memory"] = copy.deepcopy(node.args[0].meta["memory"])
            continue

        if node.target == operator.getitem:
            input_node = node.args[0]
            if "dtype" not in node.meta:
                dtypes = [str(tensor.dtype).split(".")[1] for tensor in input_node.value]
            else:
                dtypes = input_node.meta["dtype"]
            sizes = [t.numel() * dtype_byte_size(dtypes[i]) for i, t in enumerate(input_node.value)]
            start_offset = input_node.meta["memory"].start + sum(sizes[:node.args[1]])
            size = sizes[node.args[1]]
            node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
            continue

        # We do not allocate new memory for select operations. Instead, calculate
        # the memory offset from the select index
        if node.target == torch.ops.aten.select.int and node.args[1] == 0:
            size = node.value.numel() * get_num_bytes(node)
            start_offset = node.args[0].meta["memory"].start + node.args[2] * size
            node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
            continue

        # We use the partition of the first input tensor since it preallocates
        # memory for all the tensors in the stack operation (read below)
        if node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            if len(node.args) != 1 and node.args[1] != 0:
                logger.warning("Only support stacking or concatenating on the first dimension")
            else:
                node.meta["memory"] = copy.deepcopy(node.args[0][0].meta["memory"])
                continue

        # For stacked layers, place them next to each other so that we can
        # read them using a single memory access in the next operation
        next_node = next(iter(node.users))
        if next_node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            first_node = next_node.args[0][0]
            tensor_sizes = [n.value.numel() * get_num_bytes(n) for n in next_node.args[0]]
            if (memory := first_node.meta.get("memory", None)) is None:
                memory = manager.allocate_memory(first_node, sum(tensor_sizes))
                first_node.meta["memory"] = memory

            index = next_node.args[0].index(node)
            if index > 0:
                start_offset = memory.start + sum(tensor_sizes[:index])
                size = tensor_sizes[index]
                node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
        else:
            node.meta["memory"] = manager.allocate_memory(node)

        nodes_to_delete = delete_unused_values(node)
        for n in nodes_to_delete:
            if n in manager.tensor_memory_map:
                manager.free_memory(n)

        manager.take_snapshot()


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
            set_tensor_field(tensor, node, output_dir, True)
            model_params.inputs.append(tensor)
            continue
        elif node.op == 'get_attr' and "memory" in node.meta:
            tensor = Tensor()
            set_tensor_field(tensor, node, output_dir, True)
            model_params.parameters.append(tensor)
            continue
        elif node.op == 'call_function':
            op.op.CopyFrom(map_node(node, output_dir))
        elif node.op == 'call_module':
            gm = named_modules[node.target]
            assert isinstance(gm, torch.fx.GraphModule)
            submodule_args = torch.fx.node.map_arg(node.args, lambda n: n.value.clone())
            ShapeProp(gm).propagate(*submodule_args)

            operators = []
            for n in gm.graph.nodes:
                if n.op != 'call_function' or n.meta.get('fused', False) or _is_nop(n):
                    continue
                operators.append(map_node(n, output_dir))

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
        if node.op == "get_attr" and "code" in node.name:
            continue

        if node.op == "get_attr":
            continue

        header = node.name
        if hasattr(node, "value") and isinstance(node.value, (tuple, list)):
            shape_str = ", ".join([str(tuple(t.shape)) for t in node.value])
            header += f"&#92;n{shape_str}"
        elif hasattr(node, "shape"):
            header += f"&#92;n{str(tuple(node.shape))}"
        if (dtype := node.meta.get("dtype", None)) is not None:
            header += f"&#92;n{dtype}"

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
