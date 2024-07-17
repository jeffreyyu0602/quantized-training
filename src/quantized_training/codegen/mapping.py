import itertools
import operator
import os
from functools import reduce
from typing import List, Dict, Type

import graphviz
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.fx import Node, Graph, GraphModule
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
    SourcePartition,
)

from .mapping_utils import (
    OP_TO_MAPPING_FUNC,
    _set_tensor_field,
    _is_nop,
    _is_gemm_op,
    _is_elementwise_op,
)
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


def _decompose_node(graph: Graph, gm: GraphModule, args: List[Node], output_node: Node):
    arg_count = 0
    value_remap = {}
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            value_remap[node] = args[arg_count]
            arg_count += 1
        elif node.op == 'output':
            output_node.replace_all_uses_with(value_remap[node.args[0][0]])
        elif node.op == 'call_function':
            with graph.inserting_before(output_node):
                value_remap[node] = graph.node_copy(node, lambda n : value_remap[n])
            source_fn_st = value_remap[node].meta.setdefault('source_fn_stack', [])
            source_fn = source_fn_st[-1]
            source_fn_st[-1] = (value_remap[node].name, source_fn[1])


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


def _split_bmm(model: GraphModule):
    """ Split batched matrx multiply into multiple GEMM operations
    """
    graph = model.graph
    nodes = list(graph.nodes)
    for n in nodes:
        if n.op != 'call_function' or n.target != torch.ops.aten.matmul.default:
            continue
        input1 = torch.randn(n.args[0].meta['val'].size())
        input2 = torch.randn(n.args[1].meta['val'].size())

        input1_dims = sum(1 for d in input1.shape if d > 1)
        input2_dims = sum(1 for d in input2.shape if d > 1)
        if input1_dims < 3 and input2_dims < 3:
            continue

        gm = capture_pre_autograd_graph(BMM(), (input1, input2))
        _decompose_node(graph, gm, n.args, n)
        graph.erase_node(n)

    graph.lint()
    model.recompile()


class MultiHeadAttention(torch.nn.Module):
    def forward(self, query, key, scale_factor, attention_mask, value) -> torch.Tensor:
        batch_shape = query.shape[:-2]
        outputs = []
        for idx in itertools.product(*[range(dim) for dim in batch_shape]):
            attention_scores = torch.matmul(query[idx], key[idx])
            attention_scores = attention_scores / scale_factor
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            # if head_mask is not None:
            #     attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value[idx])
            outputs.append(context_layer)
        outputs = torch.stack(outputs)
        outputs = outputs.view(*batch_shape, *outputs.shape[-2:])
        return outputs


def _split_multi_head_attention(model: GraphModule):
    graph = model.graph
    partitions = get_source_partitions(graph, [torch.matmul])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    for partition in partitions:
        output_node = partition.output_nodes[0]
        # if a node is already fused, it will have no users
        if len(output_node.users) == 0:
            continue
        # TODO: check if all operations match exactly
        user_node = next(iter(output_node.users))
        fused_nodes = [output_node, user_node]
        while user_node.target != torch.ops.aten.matmul.default:
            user_node = next(iter(user_node.users))
            fused_nodes.append(user_node)
        args = tuple(a for n in fused_nodes for a in n.args if isinstance(a, Node) and a not in fused_nodes)
        args_value = torch.fx.node.map_aggregate(
            args, lambda n: torch.randn(n.meta['val'].size())
        )
        gm = capture_pre_autograd_graph(MultiHeadAttention(), args_value)
        _decompose_node(graph, gm, args, fused_nodes[-1])
        for n in reversed(fused_nodes):
            graph.erase_node(n)
    graph.lint()


def _create_subgraph(nodes: List[Node]):
    new_args = []
    new_graph = torch.fx.Graph()
    value_remap = {}

    def process_arg(arg):
        if isinstance(arg, Node) and arg not in value_remap:
            value_remap[arg] = new_graph.placeholder(arg.name)
            new_args.append(arg)
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


def fuse_operator(model: GraphModule):
    _split_multi_head_attention(model)
    _split_bmm(model)

    # TODO: this should be passed in as an argument
    vector_stages = {
        0: ["gemm"],
        1: ["add", "sub", "mul", "div"],
        2: ["exp"],
        3: ["add", "mul", "div"],
        4: ["relu", "gelu"],
    }

    graph = model.graph

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    named_modules = dict(model.named_modules(remove_duplicate=False))
    fused_nodes: Dict[Node, None] = {}

    fused_partitions = []
    for stage, ops in vector_stages.items():
        wanted_sources = [item for op in ops for item in OPERATOR_MAPPINGS[op]]
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
            matched = False
            for p in partitions:
                if p.output_nodes[0] in fused_nodes:
                    continue
                candidate = [*fp, p] if isinstance(fp, list) else [fp, p]
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

    def _create_copy_of_path(graph: Graph, nodes: List[Node]):
        for node in reversed(nodes):
            orig_users = list(node.users)
            for user in orig_users:
                with graph.inserting_before(user):
                    new_node = graph.node_copy(node, lambda n: n)
                user.replace_input_with(node, new_node)
            graph.erase_node(node)
        graph.lint()

    def get_input_nodes(args):
        input_nodes = []
        for arg in args:
            if isinstance(arg, Node):
                input_nodes.append(arg)
            elif isinstance(arg, (list, tuple)):
                input_nodes.extend(get_input_nodes(arg))
        return input_nodes

    graph.print_tabular()
    partitions = get_source_partitions(graph, ['permute', 'transpose'])
    partitions = list(itertools.chain.from_iterable(partitions.values()))
    while len(partitions) > 0:
        partition = partitions.pop(0)
        output_node = partition.output_nodes[0]

        print("processing node:", output_node)

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
                print(f"{output_node} encountered a reshape operation: {prev_node}")
                has_reshape_or_multiple_inputs = True
                break

            # if a previous node has multiple users, we cannot fuse it with the reshape
            if id(prev_node) != id(output_node) and len(prev_node.users) > 1:
                print(f"{output_node} has multiple users: {prev_node.users}")
                has_reshape_or_multiple_inputs = True
                break

            # lastly, if a node take inputs from multiple nodes, the reshape op cannot
            # be fused with any single input branch.
            input_nodes = get_input_nodes(prev_node.args)
            if len(input_nodes) > 1:
                print(f"{output_node} has multiple inputs: {input_nodes}")
                has_reshape_or_multiple_inputs = True
                break
            prev_node = input_nodes[0]

        if not has_reshape_or_multiple_inputs:
            if prev_node in fused_nodes:
                group = _search_partition_group(prev_node)
                group.extend(reshape_nodes)
                print("Fuse reshape with last node:", group, end="\n\n")
            else:
                fused_partitions.append([prev_node] + reshape_nodes)
                print("Create new group:", fused_partitions[-1], end="\n\n")
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
                _create_copy_of_path(graph, reshape_nodes)
                for new_user in input_node.users:
                    if (source_fn_st := new_user.meta.get("source_fn_stack", None)):
                        module_type = source_fn_st[-1][1]
                    else:
                        module_type = str(new_user.target).split(".")[-2]
                    par = make_partition([new_user], module_type)
                    partitions.insert(0, par)
                break

            user = next(iter(user.users))

        if has_reshape_or_multiple_users:
            continue

        if user in fused_nodes:
            group = _search_partition_group(user)
            for n in reversed(reshape_nodes):
                group.insert(0, n)
            print("Fuse reshape with next op:", group, end="\n\n")
        else:
            group = reshape_nodes + [user]
            fused_partitions.append(group)
            print("Create new group:", reshape_nodes + [user], end="\n\n")

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

    graph.print_tabular()

    for partition in fused_partitions:
        _create_and_insert_subgraph(partition, model, named_modules)

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


# HACK a temporary function that make sure inputs and weights don't overlap
# within each operation. Should be removed once we have a proper memory planner.
# Store bias in RRAM and output in reference memory.
def _write_memory_offsets(accelerator_param: AcceleratorParam):
    def get_tensor_size(tensor):
        return reduce(operator.mul, tensor.shape)

    offset = 0
    param_type = accelerator_param.WhichOneof("param_type")
    if param_type is not None:
        param = getattr(accelerator_param, param_type)
        if param.HasField("input"):
            param.input.memory.partition = 0
            param.input.memory.offset = 0
            offset += 2 * get_tensor_size(param.input)

        if param_type == "matrix_param":
            if param.HasField("mx_input"):
                param.mx_input.input.memory.partition = 0
                param.mx_input.input.memory.offset = 0
                offset += 2 * get_tensor_size(param.mx_input.input)
                param.mx_input.scale.memory.partition = 0
                param.mx_input.scale.memory.offset = 0
                offset += 2 * get_tensor_size(param.mx_input.scale)
            if param.HasField("mx_weight"):
                param.mx_weight.input.memory.partition = 0
                param.mx_weight.input.memory.offset = offset
                offset += 2 * get_tensor_size(param.mx_weight.input)
                param.mx_weight.scale.memory.partition = 0
                param.mx_weight.scale.memory.offset = offset
                offset += 2 * get_tensor_size(param.mx_weight.scale)
            if param.HasField("weight"):
                param.weight.memory.partition = 0
                param.weight.memory.offset = offset
                offset += 2 * get_tensor_size(param.weight)
            if param.HasField('bias'):
                param.bias.memory.partition = 1
                param.bias.memory.offset = offset
                offset += 2 * get_tensor_size(param.bias)

    for vector_param in accelerator_param.vector_params:
        vector_param.input.memory.partition = 0
        vector_param.input.memory.offset = offset
        offset += 2 * get_tensor_size(vector_param.input)
        if vector_param.HasField("other"):
            vector_param.other.memory.partition = 0
            vector_param.other.memory.offset = offset
            offset += 2 * get_tensor_size(vector_param.other)

    accelerator_param.output.memory.partition = 0
    accelerator_param.output.memory.offset = offset


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    model_params = ModelParams()

    ShapeProp(model).propagate(*args)
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        has_permute = False
        params = []
        if node.op == 'call_module':
            gm = named_modules[node.target]
            if isinstance(gm, torch.fx.GraphModule):
                n_args = torch.fx.node.map_aggregate(
                    node.args, lambda n: n.value if isinstance(n, Node) else n
                )
                ShapeProp(gm).propagate(*n_args)

                # If a permute operation is at the output, put permute information in the output
                # node. Otherwise, put it at the input node of GEMM or vector operations.
                for n in gm.graph.nodes:
                    if n.op != 'call_function' or n.target not in [
                        torch.ops.aten.permute.default,
                        torch.ops.aten.transpose.int,
                    ]:
                        continue

                    has_permute = True
                    user = next(iter(n.users))
                    if user.op == 'output':
                        node.meta['permute'] = n
                    else:
                        input_node = n
                        while not _is_gemm_op(user.target) and not _is_elementwise_op(user.target):
                            input_node = user
                            user = next(iter(user.users))
                        input_node.meta['permute'] = n

                if has_permute:
                    gm.graph.print_tabular()

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
        
        if has_permute:
            print(param)

    for param in model_params.params:
        _write_memory_offsets(param)
    return model_params


def gen_compute_graph(model, output_file="compute_graph"):
    nodes = {}
    edges = []
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.op == "get_attr" and "qvalues" in node.name:
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
        label = f"{{{node}}}" if label == "" else f"{{{node}|{label}}}"
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
