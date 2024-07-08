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
from torch.fx import Node, GraphModule
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
    SourcePartition,
)

from .mapping_utils import (
    OP_TO_MAPPING_FUNC,
    _is_elementwise_op,
    _is_gemm_op,
    _is_nop,
    _set_tensor_field,
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

class MatmulDecomposed(torch.nn.Module):
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
    # Split batched matrx multiply into multiple GEMM operations
    graph = model.graph
    nodes = list(graph.nodes)
    for n in nodes:
        if n.op != 'call_function' or n.target != torch.ops.aten.matmul.default:
            continue
        inputs1 = torch.randn(n.args[0].meta['val'].size())
        inputs2 = torch.randn(n.args[1].meta['val'].size())
        gm = capture_pre_autograd_graph(MatmulDecomposed(), (inputs1, inputs2))

        value_remap = {}
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                value_remap[node] = n.args[0] if node.name == 'arg0_1' else n.args[1]
            elif node.op == 'output':
                n.replace_all_uses_with(value_remap[node.args[0][0]])
            elif node.op == 'call_function':
                with graph.inserting_before(n):
                    value_remap[node] = graph.node_copy(node, lambda n : value_remap[n])
        graph.erase_node(n)

    graph.lint()
    model.recompile()


def _split_multi_head_attention(gm: GraphModule):
    fused_partitions = find_sequential_partitions(
        gm, [torch.matmul, operator.truediv, operator.add, torch.nn.functional.softmax]
    )
    for partition in fused_partitions:
        print(partition)


def _is_node_visited(node, visited):
    if isinstance(node, (tuple, list)):
        return all(_is_node_visited(a, visited) for a in node)
    if isinstance(node, Node):
        if node.op != "call_function" or "_tensor_constant_" in node.name:
            return True
        return node in visited
    return True


def _create_subgraph(nodes: List[Node]):
    new_args = []
    new_graph = torch.fx.Graph()
    value_remap = {}

    def process_arg(arg):
        if isinstance(arg, Node) and arg not in value_remap:
            value_remap[arg] = new_graph.create_node('placeholder', arg.name)
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
        prev_output_node = prev_partition.output_nodes[0] if prev_partition else None
        output_node = partition.output_nodes[0]
        # Check if the two partitions are the same
        if (
            prev_output_node is not None
            and id(prev_output_node) == id(output_node)
        ):
            return False
        # Check if all the arguments of the output node are visited before the
        # previous output node
        for arg in output_node.args:
            if (
                isinstance(arg, Node)
                and arg.op == "call_function"
                and arg != prev_output_node
                and node_order[arg] > node_order[prev_output_node]
            ):
                return False
        prev_partition = partition
    return True


def fuse_operation(model: GraphModule):
    # TODO: this should be passed in as an argument
    vector_stages = {
        0: ["gemm"],
        1: ["add", "sub", "mul"],
        2: ["exp"],
        3: ["add", "mul"],
        4: ["relu", "gelu"],
    }

    node_order = {node: idx for idx, node in enumerate(model.graph.nodes)}
    named_modules = dict(model.named_modules(remove_duplicate=False))
    fused_nodes: Dict[Node, None] = {}

    fused_partitions = []
    for stage, ops in vector_stages.items():
        wanted_sources = [item for op in ops for item in OPERATOR_MAPPINGS[op]]
        partitions = get_source_partitions(model.graph, wanted_sources)
        partitions = list(itertools.chain.from_iterable(partitions.values()))
        if len(fused_partitions) == 0:
            fused_partitions = partitions
            continue
        if len(partitions) == 0:
            continue
        fusion_candidates = []
        for x in fused_partitions:
            if isinstance(x, SourcePartition) and x.output_nodes[0] in fused_nodes:
                continue
            matched = False
            for y in partitions:
                if y.output_nodes[0] in fused_nodes:
                    continue
                candidate = [*x, y] if isinstance(x, list) else [x, y]
                if _partitions_sequential(candidate, node_order):
                    fusion_candidates.append(candidate)
                    matched = True
                    fused_nodes[y.output_nodes[0]] = None
                    if isinstance(x, SourcePartition):
                        fused_nodes[x.output_nodes[0]] = None
                    break
            if not matched:
                fusion_candidates.append(x)
        partitions = [p for p in partitions if p.output_nodes[0] not in fused_nodes]
        fused_partitions = [
            p for p in fusion_candidates
            if not (isinstance(p, SourcePartition) and p.output_nodes[0] in fused_nodes)
        ]
        fused_partitions = fused_partitions + partitions

    for partition in fused_partitions:
        if not isinstance(partition, list):
            continue
        nodes = [p.output_nodes[0] for p in partition]
        _create_and_insert_subgraph(nodes, model, named_modules)

    return GraphModule(model, model.graph)


def fuse_operator(model: GraphModule):
    _split_bmm(model)
    return fuse_operation(model)
    # _split_multi_head_attention(model)
    visited : Dict[Node, None] = {}
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.op != 'call_function':
            visited[node] = None
            continue
        nodes = [node]
        cur_node = node
        is_gemm_or_elwise_op = _is_gemm_op(node.target) or _is_elementwise_op(node.target)
        # Perform fusion if
        # 1) all arguments are visited
        # 2) the root node is a GEMM or elementwise op and the user is a elwise operation
        # or the user is a nop operation
        while len(cur_node.users) == 1:
            user = next(iter(cur_node.users))
            if not all(
                _is_node_visited(arg, visited) for arg in user.args if arg != cur_node
            ) or not (
                _is_nop(user.target)
                or is_gemm_or_elwise_op and _is_elementwise_op(user.target)
            ):
                break
            nodes.append(user)
            cur_node = user
        if len(nodes) == 1:
            visited[node] = None
            continue
        new_node = _create_and_insert_subgraph(nodes, model, named_modules)
        visited[new_node] = None
    return GraphModule(model, model.graph)


def _map_operation(node: Node, output_dir: str):
    if node.op != "call_function":
        return None
    for mapping_fn in OP_TO_MAPPING_FUNC.values():
        param = mapping_fn(node, output_dir)
        if param is not None:
            return param
    print(f"WARNING: unsupported operation {node.name}: {node.target}")
    return None


def _compose_param(params: List):
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


def _get_size(tensor):
    return reduce(lambda x, y: x * y, tensor.shape)


# HACK a temporary function that make sure inputs and weights don't overlap
# within each operation. Should be removed once we have a proper memory planner.
# Store bias in RRAM and output in reference memory.
def _plan_memory(accelerator_param: AcceleratorParam):
    offset = 0
    param_type = accelerator_param.WhichOneof("param_type")
    if param_type is not None:
        param = getattr(accelerator_param, param_type)
        param.input.memory.partition = 0
        param.input.memory.offset = 0
        offset += 2 * _get_size(param.input)

        if param_type == "matrix_param":
            param.weight.memory.partition = 0
            param.weight.memory.offset = offset
            offset += 2 * _get_size(param.weight)
            if param.HasField('bias'):
                param.bias.memory.partition = 1
                param.bias.memory.offset = offset
                offset += 2 * _get_size(param.bias)

    for vector_param in accelerator_param.vector_params:
        vector_param.input.memory.partition = 0
        vector_param.input.memory.offset = offset
        offset += 2 * _get_size(vector_param.input)
        if vector_param.HasField("other"):
            vector_param.other.memory.partition = 0
            vector_param.other.memory.offset = offset
            offset += 2 * _get_size(vector_param.other)

    accelerator_param.output.memory.partition = 0
    accelerator_param.output.memory.offset = offset


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    model_params = ModelParams()

    ShapeProp(model).propagate(*args)
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        params = []
        if node.op == 'call_module':
            gm = named_modules[node.target]
            if isinstance(gm, torch.fx.GraphModule):
                n_args = torch.fx.node.map_aggregate(
                    node.args, lambda x: x.value if isinstance(x, Node) else x
                )
                ShapeProp(gm).propagate(*n_args)
                params = [_map_operation(n, output_dir) for n in gm.graph.nodes]
        elif node.op == 'call_function':
            params.append(_map_operation(node, output_dir))

        param = _compose_param(params)
        if param is not None:
            param.name = node.name
            _set_tensor_field(param.output, node, output_dir)
            model_params.params.append(param)

    for param in model_params.params:
        _plan_memory(param)
    return model_params


def gen_compute_graph(model, output_file="compute_graph"):
    nodes = {}
    edges = []
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
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
