import copy
import itertools
import operator
import os
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
)
from .shape_prop import ShapeProp


OPERATOR_MAPPINGS = {
    "add": ["add", "add_", operator.add, torch.add, operator.iadd],
    "sub": ["sub", "sub_", operator.sub, torch.sub, operator.isub],
    "mul": ["mul", "mul_", operator.mul, torch.mul, operator.imul],
    "div": ["div", "div_", operator.truediv, torch.div, operator.itruediv],
    "exp": [torch.exp],
    "relu": [torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_],
    "gelu": [torch.nn.GELU, torch.nn.functional.gelu],
    "gemm": [torch.nn.Conv2d, torch.nn.Linear, torch.matmul],
}

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
                new_node = model.graph.node_copy(node, lambda n : value_remap[n])
            value_remap[node] = new_node

            if node.op == "get_attr":
                model.register_buffer(new_node.target, gm.get_buffer(node.target))

            # Update the node name in the source_fn_stack, which is used in get_source_partitions
            if (source_fn_st := new_node.meta.get('source_fn_stack', None)) is not None:
                source_fn = source_fn_st[-1]
                source_fn_st[-1] = (new_node.name, source_fn[1])

            # TODO copy other metadata?
            if (nn_module_stack := orig_node.meta.get('nn_module_stack', None)) is not None:
                new_node.meta.setdefault('nn_module_stack', copy.deepcopy(nn_module_stack))

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

            # The node_copy function performs a shallow copy of the metadata dictionary.
            # As a result, modifying the source_fn metadata could affect other nodes. To
            # prevent these side effects, we perform a deep copy of the source_fn_stack.
            source_fn_st = copy.deepcopy(node.meta['source_fn_stack'])
            source_fn = source_fn_st[-1]
            source_fn_st[-1] = (new_node.name, source_fn[1])
            new_node.meta['source_fn_stack'] = source_fn_st
        graph.erase_node(node)
    return node_groups


def split_multi_head_attention(model: GraphModule):
    graph = model.graph
    partitions = get_source_partitions(graph, [torch.matmul])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    count = 0
    for partition in partitions:
        # we group matmul into pairs. This logic assumes that torch.matmul
        # is only used in multi-head attention. This logic should be improved.
        count += 1
        if count % 2 == 0:
            continue

        # Fine the path between two matmul operations
        output_node = partition.output_nodes[0]
        user_node = next(iter(output_node.users))
        fused_nodes = [output_node, user_node]
        while user_node.target != torch.ops.aten.matmul.default:
            user_node = next(iter(user_node.users))
            fused_nodes.append(user_node)

        output_1 = _decompose_bmm(model, fused_nodes[0])
        output_2 = _decompose_bmm(model, fused_nodes[-1])
        if output_1 is None and output_2 is None:
            continue

        node_groups = split_nodes_on_path(graph, fused_nodes[1:-1])

        # match the select index of inputs to first matmul and second matmul
        matmul_1 = output_1.args[0].args[0]
        for node in matmul_1:
            select = node.args[0]
            for group in node_groups:
                last_node = group[-1]
                select_2 = next(iter(last_node.users))
                select_2 = next(iter(select_2.users))
                if select_2.args[1:] != select.args[1:]:
                    continue
                group[0].replace_input_with(group[0].args[0], node)
                select_2.replace_all_uses_with(group[-1])

        from torch.fx.experimental import proxy_tensor

        # Update tensor metadata of each node
        for group in node_groups:
            for node in group:
                args = torch.fx.node.map_aggregate(
                    node.args, lambda n: n.meta['val'] if isinstance(n, Node) else n
                )
                out = node.target(*args)
                node.meta['val'] = proxy_tensor.extract_val(out)

    graph.eliminate_dead_code()
    graph.lint()


def _create_subgraph(nodes: List[Node]):
    new_args = []
    new_graph = torch.fx.Graph()
    value_remap = {}

    def process_arg(arg):
        if isinstance(arg, Node) and arg not in value_remap:
            value_remap[arg] = new_graph.placeholder(arg.name)
            new_args.append(arg)
            value_remap[arg].meta['source_node'] = arg
        elif isinstance(arg, (list, tuple)):
            for n in arg:
                process_arg(n)

    for node in nodes:
        process_arg(node.args)
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


def _make_node_partition(node: Node) -> SourcePartition:
    module_type = None
    if (source_fn_st := node.meta.get("source_fn_stack", None)) is not None:
        module_type = source_fn_st[-1][1]

    def make_partition(nodes: List[Node], module_type: Type) -> SourcePartition:
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

    return make_partition([node], module_type)


def fuse_operator(model: GraphModule, vector_stages=None):
    if vector_stages is None:
        vector_stages = {}

    split_multi_head_attention(model)

    graph = model.graph

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    named_modules = dict(model.named_modules(remove_duplicate=False))
    fused_nodes: Dict[Node, None] = {}

    fused_partitions = []
    for stage, ops in vector_stages.items():
        # If there is no corresponding mapping, we directly append the op string
        wanted_sources = [item for op in ops for item in OPERATOR_MAPPINGS.get(op, [op])]
        partitions = get_source_partitions(graph, wanted_sources)
        partitions = list(itertools.chain.from_iterable(partitions.values()))

        if len(fused_partitions) == 0:
            fused_partitions = partitions
            continue

        if len(partitions) == 0:
            continue

        fusion_candidates = []
        for fp in fused_partitions:
            if isinstance(fp, SourcePartition) and fp.output_nodes[0] in fused_nodes:
                continue
            # There could be some nop nodes between the two partitions. We need to
            # manually insert them into the fusion candidates.
            last_node = fp[-1] if isinstance(fp, list) else fp
            last_node = last_node.output_nodes[0]
            node_iter = next(iter(last_node.users))
            nop_nodes = []
            while _is_nop(node_iter.target) and len(node_iter.users) == 1:
                nop_nodes.append(_make_node_partition(node_iter))
                node_iter = next(iter(node_iter.users))

            matched = False
            for p in partitions:
                if p.output_nodes[0] in fused_nodes:
                    continue
                candidate = [*fp, *nop_nodes, p] if isinstance(fp, list) else [fp, *nop_nodes, p]
                if _partitions_sequential(candidate, node_order):
                    fusion_candidates.append(candidate)
                    fused_nodes[p.output_nodes[0]] = None
                    if isinstance(fp, SourcePartition):
                        fused_nodes[fp.output_nodes[0]] = None
                    matched = True
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

    def _search_partition_group(node):
        for g in fused_partitions:
            if node in g:
                return g
        return None

    def get_input_nodes(args):
        input_nodes = []
        for arg in args:
            if isinstance(arg, Node):
                input_nodes.append(arg)
            elif isinstance(arg, (list, tuple)):
                input_nodes.extend(get_input_nodes(arg))
        return input_nodes

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
        reshape_nodes = []
        prev_node = output_node
        has_reshape_or_multiple_inputs = False
        while not _is_gemm_op(prev_node.target) and not _is_elementwise_op(prev_node.target):
            reshape_nodes.insert(0, prev_node)

            # right now we don't support fusing more than one reshape operation.
            # Require codegen schema change for this.
            if id(prev_node) != id(output_node) and prev_node.target in [
                torch.ops.aten.permute.default,
                torch.ops.aten.transpose.int,
            ]:
                has_reshape_or_multiple_inputs = True
                break

            # if a previous node has multiple users, we cannot fuse it with the reshape
            if id(prev_node) != id(output_node) and len(prev_node.users) > 1:
                has_reshape_or_multiple_inputs = True
                break

            # lastly, if a node take inputs from multiple nodes, the reshape op cannot
            # be fused with any single input branch.
            input_nodes = get_input_nodes(prev_node.args)
            if len(input_nodes) > 1:
                has_reshape_or_multiple_inputs = True
                break
            prev_node = input_nodes[0]

        if not has_reshape_or_multiple_inputs:
            if prev_node in fused_nodes:
                group = _search_partition_group(prev_node)
                group.extend(reshape_nodes)
                for n in reshape_nodes:
                    fused_nodes[n] = None
            else:
                group = [prev_node] + reshape_nodes
                fused_partitions.append(group)
                for n in group:
                    fused_nodes[n] = None
            continue

        if len(output_node.users) == 0:
            continue

        user = output_node
        reshape_nodes = []
        has_reshape_or_multiple_users = False
        while not _is_gemm_op(user.target) and not _is_elementwise_op(user.target):
            reshape_nodes.append(user)

            # we cannot fuse multiple reshape operations together
            if id(user) != id(output_node) and user.target in [
                torch.ops.aten.permute.default,
                torch.ops.aten.transpose.int,
            ]:
                has_reshape_or_multiple_users = True
                break

            # if there is a divergence in the graph, duplicate the path
            # and perform fusion on each branch
            if len(user.users) > 1:
                has_reshape_or_multiple_users = True
                input_node = output_node.args[0]
                split_nodes_on_path(graph, reshape_nodes)
                for new_user in input_node.users:
                    partitions.insert(0, _make_node_partition(new_user))
                break

            user = next(iter(user.users))

        if has_reshape_or_multiple_users:
            continue

        if user in fused_nodes:
            group = _search_partition_group(user)
            for n in reversed(reshape_nodes):
                group.insert(0, n)
                fused_nodes[n] = None
        else:
            group = reshape_nodes + [user]
            fused_partitions.append(group)
            for n in group:
                fused_nodes[n] = None

        # switch order of transpose and select ops
        if group[0].target == torch.ops.aten.transpose.int:
            transpose_node = group.pop(0)
            num_nodes = 0
            ndim = transpose_node.meta['val'].ndim
            dims = [d % ndim for d in transpose_node.args[1:]]
            for n in group:
                if n.target != torch.ops.aten.select.int:
                    break
                select_dim = n.args[1]
                if select_dim in dims:
                    break
                with graph.inserting_after(n):
                    new_node = graph.node_copy(transpose_node, lambda _: n)
                orig_users = [user for user in n.users.keys() if id(user) != id(new_node)]
                for user in orig_users:
                    user.replace_input_with(n, new_node)
                transpose_node.replace_all_uses_with(transpose_node.args[0])
                graph.erase_node(transpose_node)
                transpose_node = new_node
                dims = [d - 1 if d > select_dim else d for d in dims]
                num_nodes += 1
            del group[:num_nodes]
            group.insert(0, transpose_node)

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
                while not _is_gemm_op(user.target) and not _is_elementwise_op(user.target):
                    input_node = user
                    user = next(iter(user.users))
                input_node.meta['reshape'] = n

    graph.lint()
    graph.eliminate_dead_code()
    return GraphModule(model, graph)


def _map_operation(node: Node, output_dir: str):
    if node.op != "call_function":
        return None
    for mapping_fn in OP_TO_MAPPING_FUNC.values():
        param = mapping_fn(node, output_dir)
        if param is not None:
            return param
    print(f"WARNING: unsupported operation {node.name}: {node.target}")
    return None


def _compose_accelerator_param(params: List):
    params = [p for p in params if p is not None]
    if len(params) == 0:
        return None

    accelerator_param = AcceleratorParam()
    if isinstance(params[0], MatrixParam):
        accelerator_param.matrix_param.CopyFrom(params[0])
        if len(params) > 1:
            accelerator_param.vector_params.extend(params[1:])
    elif isinstance(params[0], VectorParam):
        accelerator_param.vector_params.extend(params)
    else:
        assert len(params) == 1, f"{str(params[0].opcode)} does not support fusion"
        if params[0].opcode == "layer_norm":
            accelerator_param.matrix_param.CopyFrom(params[0])
        if isinstance(params[0], PoolingParam):
            accelerator_param.pooling_param.CopyFrom(params[0])
        if isinstance(params[0], ReduceParam):
            accelerator_param.reduce_param.CopyFrom(params[0])
        if isinstance(params[0], ReshapeParam):
            accelerator_param.reshape_param.CopyFrom(params[0])
    return accelerator_param


def allocate_weights(model: GraphModule, manager: MemoryManager = None):
    if manager is None:
        manager = MemoryManager(DEFAULT_MEMORY_SIZE)

    for node in model.graph.nodes:
        if node.op == "get_attr":
            node.meta["memory"] = manager.allocate_memory(node)


def allocate_activations(model: GraphModule, manager: MemoryManager = None):
    if manager is None:
        manager = MemoryManager(DEFAULT_MEMORY_SIZE)

    for node in model.graph.nodes:
        if node.op == "placeholder":
            node.meta["memory"] = manager.allocate_memory(node)

    named_modules = dict(model.named_modules(remove_duplicate=False))

    # Allocate memory for intermediate tensors
    visited: Dict[Node, None] = {}
    for node in model.graph.nodes:
        if node.op not in ["call_function", "call_module"]:
            continue

        # Propagate memory metadata for nop nodes
        # TODO: slice op and non-zero select op should be handled separately
        if (
            _is_nop(node.target)
            or node.target == torch.ops.aten.slice.Tensor
            or node.target == torch.ops.aten.select.int and node.args[1] != 0
        ):
            node.meta["memory"] = copy.deepcopy(node.args[0].meta["memory"])
            continue

        # We do not allocate new memory for select operations. Instead, calculate
        # the memory offset from the select index
        if node.target == torch.ops.aten.select.int:
            assert node.args[1] == 0, "Only support select on the first dimension"
            size = manager.calculate_tensor_size(node.shape)
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
                size = sum(manager.calculate_tensor_size(n.shape) for n in maybe_stack_node.args[0])
                memory = manager.allocate_memory(first_node, size)
                first_node.meta["memory"] = memory

            index = maybe_stack_node.args[0].index(node)
            if index > 0:
                start_offset = memory.start + sum(
                    manager.calculate_tensor_size(n.shape) for n in maybe_stack_node.args[0][:index]
                )
                size = manager.calculate_tensor_size(node.shape)
                node.meta["memory"] = Partition(start_offset, start_offset + size, manager.partition_id)
        else:
            node.meta["memory"] = manager.allocate_memory(node)
        visited[node] = None

        # Propagate memory metadata for the inputs of submodules. Since reshape
        # operations are fused within each submodule, we propagate memory metadata
        # for these nodes as well.
        if node.op == "call_module" and isinstance(named_modules[node.target], GraphModule):
            gm = named_modules[node.target]
            node_args = iter(node.args)
            for n in gm.graph.nodes:
                if n.op == 'placeholder':
                    n.meta['memory'] = next(node_args).meta.get('memory', None)
                elif _is_nop(n.target) or n.target in [
                    torch.ops.aten.permute.default,
                    torch.ops.aten.transpose.int,
                ]:
                    n.meta['memory'] = n.args[0].meta.get('memory', None)

        # We treat cat, select, slice, and stack operations as nops. Therefore,
        # we need to find if their immediate users have been visited.
        def _node_visited(n):
            if _is_nop(n.target) or n.target in [
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


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    named_modules = dict(model.named_modules(remove_duplicate=False))

    ShapeProp(model).propagate(*args)
    model_params = ModelParams()
    for node in model.graph.nodes:
        params = []
        if node.op == 'call_module':
            gm = named_modules[node.target]
            if isinstance(gm, torch.fx.GraphModule):
                n_args = torch.fx.node.map_aggregate(
                    node.args, lambda n: n.value if isinstance(n, Node) else n
                )
                ShapeProp(gm).propagate(*n_args)
                for n in gm.graph.nodes:
                    param = _map_operation(n, output_dir)
                    if isinstance(param, (MatrixParam, VectorParam)):
                        params.append(param)
        elif node.op == 'call_function':
            params.append(_map_operation(node, output_dir))

        param = _compose_accelerator_param(params)
        if param is not None:
            param.name = node.name
            _set_tensor_field(param.output, node, output_dir)
            model_params.params.append(param)

    return model_params


def gen_compute_graph(model, output_file="compute_graph"):
    nodes = {}
    edges = []
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.op == "get_attr" and "code" in node.name:
            continue

        if node.op in ["placeholder", "get_attr"]:
            continue

        label = ""
        if node.op == "call_module":
            gm = named_modules[node.target]
            if isinstance(gm, torch.fx.GraphModule):
                label = "&#92;n".join([
                    str(n.target) for n in gm.graph.nodes if n.op == "call_function"
                ])
        else:
            label = str(node.target)
        node_str = node.name
        if hasattr(node, "shape"):
            node_str += f"&#92;n{str(tuple(node.shape))}"
        label = f"{{{node_str}}}" if label == "" else f"{{{node_str}|{label}}}"
        label = label.replace("<", "\<").replace(">", "\>")

        nodes[node.name] = {
            "label": label,
            "shape": "Mrecord",
        }
        for n in node.users:
            edges.append((node.name, n.name))

    g = graphviz.Digraph()

    for node, attrs in nodes.items():
        g.node(node, **attrs)

    for edge in edges:
        g.edge(edge[0], edge[1])

    g.render(output_file, format='svg', cleanup=True)
