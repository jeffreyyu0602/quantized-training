import copy
import itertools
import logging
import os
from collections import defaultdict
from typing import List, Dict, Type

import graphviz
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx import Node, Graph, GraphModule
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
    SourcePartition,
)

from .mapping_utils import (
    OP_TO_MAPPING_FUNC,
    _set_tensor_field,
    _is_gemm_op,
    _is_elementwise_op,
    _is_nop,
)
from .memory import MemoryManager, Partition
from .param_pb2 import (
    AcceleratorParam,
    ModelParams,
    MatrixParam,
    VectorParam,
    PoolingParam,
    ReduceParam,
    ReshapeParam,
    NopParam,
)
from .shape_prop import ShapeProp
from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_SIZE = 1024 ** 4


def _decompose_node(model: GraphModule, gm: GraphModule, orig_node: Node) -> List[Node]:
    arg_index = 0
    value_remap = {}
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            value_remap[node] = orig_node.args[arg_index]
            arg_index += 1
        elif node.op == 'output':
            orig_node.replace_all_uses_with(value_remap[node.args[0][0]])
        else:
            with model.graph.inserting_before(orig_node):
                new_node = model.graph.node_copy(node, lambda n: value_remap[n])
            value_remap[node] = new_node

            source_fn_st = new_node.meta.setdefault('source_fn_stack', [])
            source_fn = source_fn_st[-1][1] if len(source_fn_st) > 0 else new_node.target
            source_fn_st.append((new_node.name, source_fn))

            if (nn_module_stack := orig_node.meta.get('nn_module_stack', None)) is not None:
                new_node.meta.setdefault('nn_module_stack', nn_module_stack)

            if node.op != "get_attr":
                continue

            # TODO: might have duplicate target names
            if node.target in gm._buffers:
                if new_node.target in model._buffers:
                    logger.warning(f"Duplicate buffer name: {new_node.target}")
                model.register_buffer(new_node.target, gm.get_buffer(node.target))
            if node.target in gm._parameters:
                if new_node.target in model._parameters:
                    logger.warning(f"Duplicate parameter name: {new_node.target}")
                model.register_parameter(new_node.target, gm.get_parameter(node.target))

    output_node = list(gm.graph.nodes)[-1]
    return [value_remap[n] for n in output_node.args[0]]


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


def _decompose_bmm(model: GraphModule, node: Node):
    assert node.op == 'call_function' and node.target == torch.ops.aten.matmul.default
    input1 = node.args[0].meta['val']
    input2 = node.args[1].meta['val']

    input1_dims = sum(1 for d in input1.shape if d > 1)
    input2_dims = sum(1 for d in input2.shape if d > 1)
    if input1_dims < 3 and input2_dims < 3:
        return None

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

    graph.eliminate_dead_code()
    graph.lint()
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
    new_node.meta['source_module'] = submodule
    if (dtype := nodes[-1].meta.get('dtype', None)) is not None:
        new_node.meta['dtype'] = dtype
    return new_node


def _partitions_sequential(partitions: List[SourcePartition], node_order: Dict[Node, int]):
    prev_partition = None
    for partition in partitions:
        if prev_partition is None:
            prev_partition = partition
            continue
        if prev_partition is not None and not check_subgraphs_connected(
            prev_partition, partition
        ):
            return False
        prev_output_node = prev_partition.output_nodes[0]
        output_node = partition.output_nodes[0]
        # Check if the two partitions are the same
        if id(prev_output_node) == id(output_node):
            return False
        # Check if all the arguments of the output node are visited before the
        # previous output node
        for arg in output_node.args:
            if (
                isinstance(arg, Node)
                and arg.op == "call_function"
                and id(arg) != id(prev_output_node)
                and node_order[arg] > node_order[prev_output_node]
            ):
                return False
        prev_partition = partition
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


def fuse_operator(model: GraphModule, pipeline=None):
    if pipeline is None:
        pipeline = {}

    graph = model.graph

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    named_modules = dict(model.named_modules(remove_duplicate=False))
    fused_nodes: Dict[Node, None] = {}

    fused_partitions = []
    for wanted_sources in pipeline.values():
        partitions = get_source_partitions(graph, wanted_sources)
        partitions = list(itertools.chain.from_iterable(partitions.values()))

        if len(fused_partitions) == 0:
            fused_partitions = partitions
            continue

        if len(partitions) == 0:
            continue

        fusion_candidates = []
        for fp in fused_partitions:
            # If a node has already been fused at this round, skip it
            if isinstance(fp, SourcePartition) and fp.output_nodes[0] in fused_nodes:
                continue

            # If the last node in the group has multiple users, it cannot be fused
            last_node = fp[-1].output_nodes[0] if isinstance(fp, list) else fp.output_nodes[0]
            if len(last_node.users) > 1:
                fusion_candidates.append(fp)
                continue

            # Include any NOP nodes after the last node in the group
            nops = []
            next_node = next(iter(last_node.users))
            while _is_nop(next_node) and len(next_node.users) == 1:
                nops.append(make_partition(next_node))
                next_node = next(iter(next_node.users))

            matched = False
            for p in partitions:
                if p.output_nodes[0] in fused_nodes:
                    continue
                candidate = [*fp, *nops, p] if isinstance(fp, list) else [fp, *nops, p]
                if _partitions_sequential(candidate, node_order):
                    matched = True
                    fusion_candidates.append(candidate)
                    for c in candidate:
                        fused_nodes[c.output_nodes[0]] = None
                    break
            if not matched:
                fusion_candidates.append(fp)

        fused_partitions = [
            p for p in fusion_candidates + partitions
            if not (isinstance(p, SourcePartition) and p.output_nodes[0] in fused_nodes)
        ]

    fused_partitions = [
        [p.output_nodes[0] for p in fp] for fp in fused_partitions if isinstance(fp, list)
    ]

    def search_group(node):
        for g in fused_partitions:
            if node in g:
                return g
        return None

    partitions = get_source_partitions(graph, ['permute', 'transpose'])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        partition = partitions.pop(0)
        output_node = partition.output_nodes[0]

        # The logic for fusing reshape operations is that we first trace up the graph
        # until we reach a GEMM or elementwise operation. If we encounter a reshape operation
        # or a node that has either multiple inputs or users along the way, we stop and
        # try to fuse the node with its immediate user. For fusing a node with its user,
        # we trace down the graph until we reach a GEMM or elementwise operation. If we
        # encounter a node with multiple users, we duplicate all the elements on the path
        # and perform fusion on each branch.
        reshape_nodes = [output_node]
        prev_node = output_node.all_input_nodes[0]
        fused = True
        while _is_nop(prev_node) or prev_node.target in [
            torch.ops.aten.cat.default,
            torch.ops.aten.select.int,
            torch.ops.aten.stack.default,
        ]:
            # Reshape cannot be fused with a node that has multiple users or inputs
            if (
                len(prev_node.users) > 1 or
                len(prev_node.all_input_nodes) != 1
            ):
                fused = False
                break

            reshape_nodes.insert(0, prev_node)
            prev_node = prev_node.all_input_nodes[0]

        # We don't support fusing more than one reshape operation.
        if fused and prev_node.target not in [
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
        ]:
            if prev_node in fused_nodes:
                group = search_group(prev_node)
                group.extend(reshape_nodes)
            else:
                reshape_nodes.insert(0, prev_node)
                fused_partitions.append(reshape_nodes)

            for n in reshape_nodes:
                fused_nodes[n] = None
            continue

        reshape_nodes = []
        next_node = output_node
        fused = True
        while id(next_node) == id(output_node) or _is_nop(next_node) or next_node.target in [
            torch.ops.aten.cat.default,
            torch.ops.aten.select.int,
            torch.ops.aten.stack.default,
        ]:
            reshape_nodes.append(next_node)

            # If the node is the last node, stop.
            if len(next_node.users) == 0:
                fused = False
                break

            # If there is a divergence in the graph, duplicate the path
            # and perform fusion on each branch
            if len(next_node.users) != 1:
                fused = False
                groups = split_nodes_on_path(graph, reshape_nodes)
                for g in groups:
                    partitions.insert(0, make_partition(g[0]))
                break

            next_node = next(iter(next_node.users))

        if not fused or next_node.target in [
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
        ]:
            continue

        if next_node in fused_nodes:
            group = search_group(next_node)
            for n in reversed(reshape_nodes):
                group.insert(0, n)
        else:
            reshape_nodes.append(next_node)
            fused_partitions.append(reshape_nodes)
            group = reshape_nodes

        for n in reshape_nodes:
            fused_nodes[n] = None

        # Switch order of transpose and select nodes
        reshape_node = group[0]
        if reshape_node.target == torch.ops.aten.transpose.int:
            ndim = reshape_node.args[0].meta['val'].ndim
            axes = [x if x >= 0 else x + ndim for x in reshape_node.args[1:]]
            unfused_nodes = []
            next_node = next(iter(reshape_node.users))
            while (
                next_node.target == torch.ops.aten.select.int
                and next_node.args[1] == 0
                and next_node.args[1] not in axes
            ):
                unfused_nodes.append(next_node)
                next_node = next(iter(next_node.users))
                axes = [x - 1 for x in axes]

            if len(unfused_nodes) > 0:
                with graph.inserting_before(next_node):
                    new_node = graph.node_copy(reshape_node, lambda _: unfused_nodes[-1])
                next_node.replace_input_with(unfused_nodes[-1], new_node)
                unfused_nodes[0].replace_input_with(reshape_node, reshape_node.args[0])

                graph.erase_node(reshape_node)

                for n in unfused_nodes:
                    group.remove(n)
                group.remove(reshape_node)
                group.insert(0, new_node)

    partitions = get_source_partitions(graph, [torch.ops.quantized_ops.dequantize_symmetric])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        p = partitions.pop(0)
        node = p.output_nodes[0]
        # If the node is already fused, skip it
        if search_group(node) is not None:
            continue
        # Fuse the dequantize node with its user
        fused_nodes = []
        next_node = node
        fused = True
        while id(next_node) == id(node) or _is_nop(next_node) or next_node.target in [
            torch.ops.aten.cat.default,
            torch.ops.aten.select.int,
            torch.ops.aten.stack.default,
        ]:
            fused_nodes.append(next_node)
            if len(next_node.users) != 1:
                fused = False
                groups = split_nodes_on_path(graph, fused_nodes)
                for g in groups:
                    partitions.insert(0, make_partition(g[0]))
                break
            next_node = next(iter(next_node.users))

        if not fused:
            continue

        if (group := search_group(next_node)) is not None:
            for n in reversed(fused_nodes):
                if n not in group:
                    user_node = next(iter(n.users))
                    group.insert(group.index(user_node), n)
        else:
            fused_nodes.append(next_node)
            fused_partitions.append(fused_nodes)

    # Fuse nodes that appear earlier in the graph first
    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    fused_partitions.sort(key=lambda x: node_order[x[0]])

    for partition in fused_partitions:
        node = _create_and_insert_subgraph(partition, model, named_modules)
        named_modules = dict(model.named_modules(remove_duplicate=False))
        gm = named_modules[node.target]

        # If a permute operation is at the output, annotate the output node.
        # Otherwise, annotate the immediate user of the permute operation.
        for n in gm.graph.nodes:
            if n.op != 'call_function' or n.target not in [
                torch.ops.aten.permute.default,
                torch.ops.aten.transpose.int,
            ]:
                continue

            user = next(iter(n.users))
            if user.op == 'output':
                node.meta['reshape'] = n
            else:
                input_node = n
                while _is_nop(user):
                    input_node = user
                    user = next(iter(user.users))
                input_node.meta['reshape'] = n

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def allocate_weights(model: GraphModule, manager: MemoryManager = None):
    if manager is None:
        manager = MemoryManager(DEFAULT_MEMORY_SIZE)

    for node in model.graph.nodes:
        if node.op == "get_attr" and "quant_map" not in node.name:
            node.meta["memory"] = manager.allocate_memory(node)


def allocate_activations(model: GraphModule, manager: MemoryManager = None):
    if manager is None:
        manager = MemoryManager(DEFAULT_MEMORY_SIZE)

    for node in model.graph.nodes:
        if node.op == "placeholder":
            node.meta["memory"] = manager.allocate_memory(node)

    # Allocate memory for intermediate tensors
    visited: Dict[Node, None] = {}
    for node in model.graph.nodes:
        if node.op not in ["call_function", "call_module"]:
            continue

        # Propagate memory metadata for nop nodes
        # TODO: slice op and non-zero select op should be handled separately
        if (
            _is_nop(node)
            or node.target == torch.ops.aten.slice.Tensor
            or node.target == torch.ops.aten.select.int and node.args[1] != 0
        ):
            node.meta["memory"] = copy.deepcopy(node.args[0].meta["memory"])
            continue

        def get_node_byte_size(n):
            if n.meta.get('dtype', None) is not None:
                return dtype_byte_size(n.meta['dtype'])
            return dtype_byte_size(n.value.dtype)

        # We do not allocate new memory for select operations. Instead, calculate
        # the memory offset from the select index
        if node.target == torch.ops.aten.select.int:
            assert node.args[1] == 0, "Only support select on the first dimension"
            size = manager.calculate_tensor_size(node.shape) * get_node_byte_size(node)
            start_offset = node.args[0].meta["memory"].start + node.args[2] * size
            node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
            continue

        # We use the partition of the first input tensor since it preallocates
        # memory for all the tensors in the stack operation (read below)
        if node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            node.meta["memory"] = copy.deepcopy(node.args[0][0].meta["memory"])
            continue

        # For stacked layers, place them next to each other so that we can
        # read them using a single memory access in the next operation
        maybe_stack_node = next(iter(node.users))
        if maybe_stack_node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
            first_node = maybe_stack_node.args[0][0]
            if (memory := first_node.meta.get("memory", None)) is None:
                size = sum(
                    manager.calculate_tensor_size(n.shape) * get_node_byte_size(n)
                    for n in maybe_stack_node.args[0]
                )
                memory = manager.allocate_memory(first_node, size)
                first_node.meta["memory"] = memory

            index = maybe_stack_node.args[0].index(node)
            if index > 0:
                start_offset = memory.start + sum(
                    manager.calculate_tensor_size(n.shape) * get_node_byte_size(n)
                    for n in maybe_stack_node.args[0][:index]
                )
                size = manager.calculate_tensor_size(node.shape) * get_node_byte_size(node)
                node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
        else:
            node.meta["memory"] = manager.allocate_memory(node)
        visited[node] = None

        # We treat cat, select, slice, and stack operations as nops. Therefore,
        # we need to find if their immediate users have been visited.
        def _node_visited(n):
            if _is_nop(n) or n.target in [
                torch.ops.aten.cat.default,
                torch.ops.aten.select.int,
                torch.ops.aten.slice.Tensor,
                torch.ops.aten.stack.default,
            ]:
                return all(_node_visited(user) for user in n.users)
            return n in visited

        # Free the memory of nodes whose users have all been visited.
        active_nodes = list(manager.tensor_memory_map.keys())
        for n in active_nodes:
            if n.op not in ["placeholder", "call_function", "call_module"]:
                continue
            if all(_node_visited(user) for user in n.users):
                manager.free_memory(n)


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
    _set_tensor_field(accelerator_param.output, node, output_dir, True)

    if isinstance(params[0], MatrixParam):
        accelerator_param.matrix_param.CopyFrom(params[0])
        if len(params) > 1:
            accelerator_param.vector_params.extend(params[1:])
        return accelerator_param

    if isinstance(params[0], VectorParam):
        accelerator_param.vector_params.extend(params)
        return accelerator_param

    assert len(params) == 1, f"{str(params[0].opcode)} does not support fusion"
    if params[0].opcode == "layer_norm":
        accelerator_param.matrix_param.CopyFrom(params[0])
    elif isinstance(params[0], PoolingParam):
        accelerator_param.pooling_param.CopyFrom(params[0])
    elif isinstance(params[0], ReduceParam):
        accelerator_param.reduce_param.CopyFrom(params[0])
    elif isinstance(params[0], ReshapeParam):
        accelerator_param.reshape_param.CopyFrom(params[0])
    elif isinstance(params[0], NopParam):
        accelerator_param.nop.CopyFrom(params[0])
    return accelerator_param


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    named_modules = dict(model.named_modules(remove_duplicate=False))
    buffers = dict(model.named_buffers())

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
                if _is_nop(n) or n.target in [
                    torch.ops.aten.permute.default,
                    torch.ops.aten.transpose.int,
                ]:
                    continue

                # Skip dequantize nodes that are fused inputs
                if (
                    n.target == torch.ops.quantized_ops.dequantize_symmetric
                    and n.args[0].op == 'placeholder'
                ):
                    qparam_node = n.args[1]
                    if qparam_node.op == 'placeholder':
                        qparam_node = qparam_node.meta['source_node']
                    n.meta['dq_scale'] = buffers[qparam_node.target]
                    continue

                param = _map_operation(n, output_dir)
                if param is not None and not isinstance(param, NopParam):
                    params.append(param)
        else:
            continue

        param = _compose_accelerator_param(node, params, output_dir)
        model_params.params.append(param)

    return model_params


def gen_compute_graph(model, output_file="compute_graph"):
    nodes = {}
    edges = []
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.op == "get_attr" and "code" in node.name:
            continue

        # Skip nodes with too many users
        if len(node.users) > 10:
            continue

        if node.op == "get_attr":
            continue

        header = node.name
        if hasattr(node, "shape"):
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
        for n in node.users:
            edges.append((node.name, n.name))

    g = graphviz.Digraph()
    g.attr(bgcolor="transparent")

    for node, attrs in nodes.items():
        g.node(node, **attrs)

    for edge in edges:
        g.edge(edge[0], edge[1])

    g.render(output_file, format='svg', cleanup=True)
