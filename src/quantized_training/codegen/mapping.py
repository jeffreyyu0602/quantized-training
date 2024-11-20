import copy
import itertools
import logging
import operator
import os
from collections import defaultdict
from typing import List, Dict, Type, Callable

import graphviz
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx import Node, Graph, GraphModule
from torch.fx.node import map_arg
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)

from .mapping_utils import (
    OP_TO_MAPPING_FUNC,
    _is_elementwise_op,
    _is_gemm_op,
    _is_indexing_or_concatenation_op,
    _is_nop,
    _is_reshape_op,
    _is_slicing_nop,
    _set_tensor_field,
)
from .memory import MemoryManager, Partition
from .param_pb2 import (
    AcceleratorParam,
    MatrixOp,
    ModelParams,
    Nop,
    PoolingOp,
    ReduceOp,
    ReshapeOp,
    SlicingOp,
    Tensor,
    VectorOp,
)
from .shape_prop import ShapeProp
from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_SIZE = torch.finfo(torch.float32).max


def replace_elementwise_with_vmap(
    model: GraphModule,
    mapping: Dict[str, Callable],
) -> GraphModule:
    mapped_sources = list(itertools.chain.from_iterable(mapping.values()))
    for node in list(model.graph.nodes):
        if not _is_elementwise_op(node) or len(node.all_input_nodes) > 1:
            continue

        source_fn_st = node.meta.get("source_fn_stack", [])
        if source_fn_st and source_fn_st[-1][1] in mapped_sources:
            continue

        logger.info(f"Replace {node.target} with value mapping")

        values = torch.arange(2 ** 16, dtype=torch.int16).view(torch.bfloat16)
        args = list(node.args)
        val_map = node.target(values, *args[1:])

        from ..quantize_pt2e import create_getattr_from_value

        with model.graph.inserting_before(node):
            get_attr_node = create_getattr_from_value(
                model, model.graph, f'_tensor_constant_', val_map
            )
            new_node = model.graph.call_function(
                torch.ops.quantized_ops.vmap, (args[0], get_attr_node)
            )

        node.replace_all_uses_with(new_node)
        model.graph.erase_node(node)

    model.graph.lint()

    model.graph.eliminate_dead_code()
    model.recompile()

    return model


def _decompose_node(model: GraphModule, gm: GraphModule, source_node: Node) -> List[Node]:
    args_iter = iter(source_node.args)
    value_remap = {}
    for node in list(gm.graph.nodes):
        if node.op == 'placeholder':
            value_remap[node] = next(args_iter)
        elif node.op == 'output':
            source_node.replace_all_uses_with(value_remap[node.args[0][0]])
        else:
            with model.graph.inserting_before(source_node):
                value_remap[node] = model.graph.node_copy(node, lambda n: value_remap[n])

            if (source_fn_st := node.meta.get('source_fn_stack', None)) is not None:
                source_fn = source_fn_st[-1]
                value_remap[node].meta['source_fn_stack'] = [(value_remap[node].name, source_fn[1])]

    output_node = list(gm.graph.nodes)[-1]
    return [value_remap[n] for n in output_node.args[0]]


def _decompose_bmm(model: GraphModule, node: Node):
    assert node.op == 'call_function' and node.target == torch.ops.aten.matmul.default
    input1 = node.args[0].meta['val']
    input2 = node.args[1].meta['val']

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
    output_nodes = _decompose_node(model, gm, node)
    model.graph.erase_node(node)
    return output_nodes[0]


def split_nodes_on_path(graph: Graph, nodes: List[Node]):
    node_groups = [[] for _ in range(len(nodes[-1].users))]
    for node in reversed(nodes):
        orig_users = list(node.users)
        for i, user in enumerate(orig_users):
            with graph.inserting_before(user):
                new_node = graph.node_copy(node, lambda n: n)
            user.replace_input_with(node, new_node)
            node_groups[i].insert(0, new_node)

            # Copy and update the source_fn_stack
            if (source_fn_st := node.meta.get('source_fn_stack', None)) is not None:
                source_fn = source_fn_st[-1]
                new_node.meta['source_fn_stack'] = source_fn_st + [(new_node.name, source_fn[1])]
        graph.erase_node(node)
    return node_groups


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
        if len(nodes) == 1:
            _decompose_bmm(model, nodes[0])
            continue

        assert len(nodes) == 2
        matmul_1 = nodes[0]
        matmul_2 = nodes[1]

        query = matmul_1.args[0]
        key = matmul_1.args[1]
        value = matmul_2.args[1]

        # Find all the nodes between two matmul operations
        nodes_in_path = []
        next_node = next(iter(matmul_1.users))
        while len(next_node.users) == 1 and next_node.target != torch.ops.aten.matmul.default:
            nodes_in_path.append(next_node)
            next_node = next(iter(next_node.users))

        assert id(next_node) == id(matmul_2)

        output_1 = _decompose_bmm(model, matmul_1)
        output_2 = _decompose_bmm(model, matmul_2)

        def propagate_input_dtype(node):
            if (dtype := node.meta.get('dtype', None)) is None:
                return
            select_nodes = list(node.users)
            while len(select_nodes) > 0:
                select = select_nodes.pop(0)
                select.meta['dtype'] = dtype
                select_nodes.extend([
                    n for n in select.users if n.target != torch.ops.aten.matmul.default
                ])

        def propagate_output_dtype(matmul_node, output_node):
            if (dtype := matmul_node.meta.get('dtype', None)) is None:
                return
            output_nodes = [output_node, output_node.args[0]] + output_node.args[0].args[0]
            for node in output_nodes:
                node.meta['dtype'] = dtype

        if output_1 is not None:
            propagate_input_dtype(query)
            propagate_input_dtype(key)
            propagate_output_dtype(matmul_1, output_1)

        if output_2 is not None:
            propagate_input_dtype(value)
            propagate_output_dtype(matmul_2, output_2)

        if output_1 is None or output_2 is None:
            continue

        # Before splitting the nodes:
        # select -> matmul \                          / select_1 -> select_2
        #       ...
        # select -> matmul -> stack -> view -> group -> select_1 -> select_2
        # select -> matmul /                          \ select_1 -> select_2

        # After splitting the nodes:
        # select -> matmul \                 / group 1   -> select_1 -> select_2
        #       ...
        # select -> matmul -> stack -> view -> group N-1 -> select_1 -> select_2
        # select -> matmul /                 \ group N   -> select_1 -> select_2
        # 
        # We can match the decomposed matmul nodes with a specific path and remove
        # the stack and view nodes. Each group perform a single head attention.

        # Final graph after removing the stack and view nodes:
        # select -> matmul -> group 1   -> matmul
        #       ...
        # select -> matmul -> group N-1 -> matmul
        # select -> matmul -> group N   -> matmul

        node_groups = split_nodes_on_path(graph, nodes_in_path)
        matmul_nodes = output_1.args[0].args[0]
        for node in matmul_nodes:
            # Input select for the first set of matmul nodes
            select = node.args[0]
            for group in node_groups:
                # Input select for the second set of matmul nodes
                select_1 = next(iter(group[-1].users))
                select_2 = next(iter(select_1.users))
                if select_2.args[1:] == select.args[1:]:
                    group[0].replace_input_with(group[0].args[0], node)
                    select_2.replace_all_uses_with(group[-1])
                    break

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
        for arg in n.args:
            if (
                isinstance(arg, Node)
                and id(arg) != id(prev_node)
                and arg.op == "call_function"
                and order[arg] > order[prev_node]
            ):
                return False
        prev_node = n
    return True


def make_partition(nodes: List[Node], module_type: Type = None) -> SourcePartition:
    if isinstance(nodes, Node):
        nodes = [nodes]
        source_fn_st = nodes[0].meta.get("source_fn_stack", None)
        if module_type is None and source_fn_st is not None:
            module_type = source_fn_st[-1][1]

    input_nodes = set()
    output_nodes = set()
    params = set()
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, Node) and arg not in nodes:
                input_nodes.add(arg)

        if node.op == "get_attr":
            params.add(node)

        for user in node.users.keys():
            if user not in nodes:
                output_nodes.add(node)

    return SourcePartition(
        nodes,
        module_type,
        list(input_nodes),
        list(output_nodes),
        list(params),  # type: ignore[arg-type]
    )


def find_sequential_nodes(model: GraphModule, pattern: List[List[Callable]]):
    graph = model.graph

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
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

                candidate = [*nodes, *nop_nodes, output_node]
                if _nodes_sequential(candidate, node_order):
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


def transform_concat_nodes(model: GraphModule):
    graph = model.graph

    partitions = get_source_partitions(graph, [torch.stack, torch.cat])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        cat_node = partitions.pop(0).output_nodes[0]
        if cat_node.target not in [
            torch.ops.aten.cat.default, torch.ops.aten.stack.default
        ]:
            continue

        input_shape = list(cat_node.args[0][0].shape)

        concat_dim = 0 if len(cat_node.args) == 1 else cat_node.args[1]
        if concat_dim < 0:
            concat_dim += len(input_shape)

        if concat_dim == 0:
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

        graph.erase_node(cat_node)

    graph.lint()

    graph.eliminate_dead_code()
    model.recompile()
    return model


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
    transform_concat_nodes(model)

    graph = model.graph
    named_modules = dict(model.named_modules(remove_duplicate=False))

    if mapping is None:
        mapping = {}

    fused_nodes_list = find_sequential_nodes(model, mapping.values())
    node_map = {}

    def search_group(node):
        for g in fused_nodes_list:
            if node in g:
                return g
        return None

    partitions = get_source_partitions(graph, ['permute', 'transpose'])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        output_node = partitions.pop(0).output_nodes[0]

        is_tranpose = False

        if output_node.target == torch.ops.aten.transpose.int:
            ndim = output_node.args[0].meta['val'].ndim
            axes = {x if x >= 0 else x + ndim for x in output_node.args[1:]}
            is_tranpose = (axes == {ndim - 2, ndim - 1})

        if not is_tranpose:
            match_found = True
            fused_nodes = [output_node]
            prev_node = output_node.all_input_nodes[0]

            while not _is_gemm_op(prev_node) and not _is_elementwise_op(prev_node):
                if not _is_nop(prev_node) and not _is_slicing_nop(prev_node):
                    logger.warning(f"Cannot fuse reshape operation with {prev_node.target}")
                    match_found = False
                    break

                # Reshape cannot be fused with a node that has multiple users or inputs
                if (
                    len(prev_node.users) > 1 or
                    len(prev_node.all_input_nodes) != 1 or
                    _is_reshape_op(prev_node)
                ):
                    match_found = False
                    break

                fused_nodes.insert(0, prev_node)
                prev_node = prev_node.all_input_nodes[0]

            if match_found and len(prev_node.users) == 1:
                node_map[output_node] = prev_node
                if (group := search_group(prev_node)) is not None:
                    group.extend(fused_nodes)
                else:
                    fused_nodes.insert(0, prev_node)
                    fused_nodes_list.append(fused_nodes)
                continue

        match_found = True
        fused_nodes = []
        next_node = output_node

        while not _is_gemm_op(next_node) and not _is_elementwise_op(next_node):
            # Reshape can be fused with aten.select.int if and only if the select index is 0
            if (
                id(next_node) != id(output_node)
                and not _is_nop(next_node)
                and not _is_slicing_nop(next_node)
                and not (next_node.target == torch.ops.aten.select.int and next_node.args[1] == 0)
            ):
                logger.warning(f"Cannot fuse {next_node} with {output_node}")
                match_found = False
                break

            fused_nodes.append(next_node)

            # If there is a divergent path in the graph, duplicate the node
            # and perform fusion on each branch
            if len(next_node.users) != 1:
                match_found = False
                paths = split_nodes_on_path(graph, fused_nodes)
                for p in reversed(paths):
                    partitions.insert(0, make_partition(p[0]))
                break

            node_map[output_node] = next_node
            next_node = next(iter(next_node.users))

        if not match_found:
            node_map.pop(output_node, None)
            continue

        if (group := search_group(next_node)) is not None:
            group.extend(fused_nodes)
        else:
            fused_nodes.append(next_node)
            fused_nodes_list.append(fused_nodes)

        # Switch order of transpose and select nodes
        if output_node.target == torch.ops.aten.transpose.int:
            select_nodes = []
            next_node = next(iter(output_node.users))
            ndim = output_node.args[0].meta['val'].ndim
            axes = [x if x >= 0 else x + ndim for x in output_node.args[1:]]

            while (
                next_node.target == torch.ops.aten.select.int
                and next_node.args[1] == 0
            ):
                select_nodes.append(next_node)
                next_node = next(iter(next_node.users))
                axes = [x - 1 for x in axes]

            if len(select_nodes) > 0:
                with graph.inserting_before(next_node):
                    new_node = graph.call_function(
                        torch.ops.aten.transpose.int, (select_nodes[-1], *axes),
                    )

                next_node.replace_input_with(select_nodes[-1], new_node)
                select_nodes[0].replace_input_with(output_node, output_node.args[0])
                graph.erase_node(output_node)

                group = search_group(output_node)
                group.insert(0, new_node)
                group.remove(output_node)
                for n in select_nodes:
                    group.remove(n)

                user_node = node_map.pop(output_node, None)
                node_map[new_node] = new_node if user_node == select_nodes[-1] else user_node

    partitions = get_source_partitions(graph, [operator.getitem])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        slice_node = partitions.pop(0).output_nodes[0]

        if slice_node.target not in [torch.ops.aten.slice.Tensor, torch.ops.aten.select.int]:
            print(f"Unrecognized getitem operation: {slice_node.target}")
            continue

        if _is_slicing_nop(slice_node) or (
            slice_node.target == torch.ops.aten.select.int and slice_node.args[1] == 0
        ):
            print(f"Node {slice_node.name} is a nop or can be handled by memory planner.")
            continue

        match_found = True
        fused_nodes = []
        next_node = slice_node

        # Slicing can only be fused with vector operations
        while not _is_elementwise_op(next_node):
            if (
                id(next_node) != id(slice_node)
                and not _is_nop(next_node)
                and not _is_slicing_nop(next_node)
            ):
                logger.warning(f"Cannot fuse {next_node} with {slice_node}")
                match_found = False
                break

            fused_nodes.append(next_node)

            if len(next_node.users) != 1:
                match_found = False
                paths = split_nodes_on_path(graph, fused_nodes)
                for p in reversed(paths):
                    partitions.insert(0, make_partition(p[0]))
                break

            node_map[slice_node] = next_node
            next_node = next(iter(next_node.users))

        if not match_found:
            node_map.pop(slice_node, None)
            continue

        if (group := search_group(next_node)) is not None:
            group.extend(fused_nodes)
        else:
            fused_nodes.append(next_node)
            fused_nodes_list.append(fused_nodes)

    partitions = get_source_partitions(graph, [torch.ops.quantized_ops.dequantize])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        dequantized_node = partitions.pop(0).output_nodes[0]

        # If the node is already fused, skip it
        if search_group(dequantized_node) is not None:
            continue

        match_found = True
        fused_nodes = []
        next_node = dequantized_node

        while not _is_elementwise_op(next_node) or id(next_node) == id(dequantized_node):
            if (
                id(next_node) != id(dequantized_node)
                and not _is_nop(next_node)
                and not _is_slicing_nop(next_node)
            ):
                logger.warning(f"Cannot fuse {next_node} with {dequantized_node}")
                match_found = False
                break

            fused_nodes.append(next_node)

            if len(next_node.users) != 1:
                match_found = False
                paths = split_nodes_on_path(graph, fused_nodes)
                for p in reversed(paths):
                    partitions.insert(0, make_partition(p[0]))
                break

            node_map[dequantized_node] = next_node
            next_node = next(iter(next_node.users))

        if not match_found:
            node_map.pop(dequantized_node, None)
            continue

        if (group := search_group(next_node)) is not None:
            group.extend([n for n in fused_nodes if n not in group])
        else:
            fused_nodes.append(next_node)
            fused_nodes_list.append(fused_nodes)

    # Fuse nodes that appear earlier in the graph first
    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    for nodes in fused_nodes_list:
        nodes.sort(key=lambda n: node_order[n])
    fused_nodes_list.sort(key=lambda fn: node_order[fn[-1]])

    node_map = {v.name: k.name for k, v in node_map.items()}

    for fused_nodes in fused_nodes_list:
        node = _create_and_insert_subgraph(fused_nodes, model, named_modules)
        named_modules = dict(model.named_modules(remove_duplicate=False))
        gm = named_modules[node.target]

        buffers = dict(model.named_buffers())

        nodes = list(gm.graph.nodes)
        for n in nodes:
            if (name := node_map.get(n.name, None)) is None:
                continue
            source_node = next(iter(n for n in nodes if n.name == name))

            # A small trick to skip generating the param for this node
            source_node.meta['fused'] = True

            if _is_reshape_op(source_node):
                if next(iter(source_node.users)).op == 'output':
                    node.meta['reshape'] = source_node
                else:
                    n.meta['reshape'] = source_node
            elif source_node.target == torch.ops.quantized_ops.dequantize:
                get_attr_node = source_node.args[1]
                # get_attr_node is a placeholder node. We assume that the attribute
                # name is equal to the placeholder name.
                n.meta['dq_scale'] = buffers[get_attr_node.target]
            elif _is_indexing_or_concatenation_op(source_node):
                n.meta['indexing'] = source_node

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
    
    def get_node_byte_size(n: Node):
        if n.meta.get('dtype', None) is not None:
            return dtype_byte_size(n.meta['dtype'])
        return dtype_byte_size(n.value.dtype)

    for node in model.graph.nodes:
        if node.op not in ["call_function", "call_module"]:
            continue

        # Propagate memory metadata for nop nodes
        if _is_nop(node) or _is_slicing_nop(node):
            node.meta["memory"] = copy.deepcopy(node.args[0].meta["memory"])
            continue

        if node.target == operator.getitem:
            sizes = [t.numel() * dtype_byte_size(t.dtype) for t in node.args[0].value]
            start_offset = node.args[0].meta["memory"].start + sum(sizes[:node.args[1]])
            size = sizes[node.args[1]]
            node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
            continue

        # We do not allocate new memory for select operations. Instead, calculate
        # the memory offset from the select index
        if node.target == torch.ops.aten.select.int and node.args[1] == 0:
            size = node.value.numel() * get_node_byte_size(node)
            start_offset = node.args[0].meta["memory"].start + node.args[2] * size
            node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
            continue

        # We use the partition of the first input tensor since it preallocates
        # memory for all the tensors in the stack operation (read below)
        if node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            assert len(node.args) == 1 or node.args[1] == 0, (
                "Only support stack/cat on the first dimension"
            )
            node.meta["memory"] = copy.deepcopy(node.args[0][0].meta["memory"])
            continue

        # For stacked layers, place them next to each other so that we can
        # read them using a single memory access in the next operation
        next_node = next(iter(node.users))
        if next_node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            first_node = next_node.args[0][0]
            tensor_sizes = [n.value.numel() * get_node_byte_size(n) for n in next_node.args[0]]
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


def _map_operation(node: Node, output_dir: str):
    if node.op != "call_function":
        return None
    for mapping_fn in OP_TO_MAPPING_FUNC.values():
        param = mapping_fn(node, output_dir)
        if param is not None:
            return param
    raise NotImplementedError(f"Unsupported operation: {node.target}")


def _compose_accelerator_param(node: Node, params: List, output_dir: str):
    accelerator_param = AcceleratorParam()
    accelerator_param.name = node.name
    if isinstance(node.value, (tuple, list)):
        partition = node.meta["memory"]
        offset = partition.start
        for i, tensor in enumerate(node.value):
            tensor_param = Tensor()
            tensor_param.node = f"{node.name}_{i}"
            tensor_param.dtype = str(tensor.dtype).split(".")[1]
            tensor_param.shape.extend(tuple(tensor.shape))
            tensor_param.memory.partition = partition.partition_id
            tensor_param.memory.offset = offset
            offset += tensor.numel() * dtype_byte_size(tensor.dtype)
            accelerator_param.outputs.tensors.append(tensor_param)
    else:
        _set_tensor_field(accelerator_param.output, node, output_dir, True)

    if isinstance(params[0], MatrixOp):
        accelerator_param.matrix_param.CopyFrom(params[0])
        if len(params) > 1:
            accelerator_param.vector_params.extend(params[1:])
        return accelerator_param

    if isinstance(params[0], VectorOp):
        accelerator_param.vector_params.extend(params)
        return accelerator_param

    assert len(params) == 1, f"{str(params[0].opcode)} does not support fusion"
    if params[0].opcode == "layer_norm":
        accelerator_param.matrix_param.CopyFrom(params[0])
    elif isinstance(params[0], PoolingOp):
        accelerator_param.pooling_param.CopyFrom(params[0])
    elif isinstance(params[0], ReduceOp):
        accelerator_param.reduce_param.CopyFrom(params[0])
    elif isinstance(params[0], ReshapeOp):
        accelerator_param.reshape_param.CopyFrom(params[0])
    elif isinstance(params[0], SlicingOp):
        accelerator_param.slicing_param.CopyFrom(params[0])
    elif isinstance(params[0], Nop):
        accelerator_param.nop_param.CopyFrom(params[0])
    return accelerator_param


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    named_modules = dict(model.named_modules(remove_duplicate=False))

    ShapeProp(model).propagate(*args)
    model_params = ModelParams()
    for node in model.graph.nodes:
        params = []
        if node.op == 'call_function':
            params.append(_map_operation(node, output_dir))
        elif node.op == 'call_module':
            gm = named_modules[node.target]
            assert isinstance(gm, torch.fx.GraphModule)
            n_args = torch.fx.node.map_arg(node.args, lambda n: n.value)
            ShapeProp(gm).propagate(*n_args)
            for n in gm.graph.nodes:
                if _is_nop(n) or _is_reshape_op(n) or n.meta.get('fused', False):
                    continue
                param = _map_operation(n, output_dir)
                if param is not None and not isinstance(param, Nop):
                    params.append(param)

        if len(params) == 0:
            continue

        param = _compose_accelerator_param(node, params, output_dir)
        model_params.params.append(param)

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
    g.attr(bgcolor="transparent")

    for node, attrs in nodes.items():
        g.node(node, **attrs)

    g.edges(edges)

    g.render(output_file, format='svg', cleanup=True)
