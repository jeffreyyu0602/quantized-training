import os
from functools import reduce
from typing import List, Dict

import graphviz
import torch
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx import Node, GraphModule

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
    VectorParam,
    PoolingParam,
    ReduceParam,
    ShapeParam,
)
from .shape_prop import ShapeProp


def _is_node_visited(node, visited):
    if isinstance(node, (tuple, list)):
        return all(_is_node_visited(a, visited) for a in node)
    if isinstance(node, Node):
        if node.op != "call_function" or "_tensor_constant_" in node.name:
            return True
        return node in visited
    return True


def _create_subgraph(nodes: List[Node]):
    graph : torch.fx.Graph = torch.fx.Graph()
    value_remap = {}
    gm_args = []
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, Node) and arg not in value_remap:
                x : torch.fx.Node = graph.create_node('placeholder', arg.name)
                value_remap[arg] = x
                gm_args.append(arg)
        value_remap[node] = graph.node_copy(node, lambda n : value_remap[n])
    graph.output(value_remap[nodes[-1]])
    graph.lint()
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, tuple(gm_args)


def fuse_operator(model: GraphModule):
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
        submodule, submodule_args = _create_subgraph(nodes)
        get_new_node_name = get_new_attr_name_with_prefix('submodule_')
        node_name = get_new_node_name(model)
        setattr(model, node_name, submodule)
        named_modules[node_name] = submodule
        with model.graph.inserting_after(nodes[-1]):
            new_node = model.graph.create_node(
                'call_module', node_name, submodule_args, {})
        nodes[-1].replace_all_uses_with(new_node)
        for node in reversed(nodes):
            model.graph.erase_node(node)
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


def _compose_param(params: List[AcceleratorParam]):
    params = [p for p in params if p is not None]
    if len(params) == 0:
        return None

    param = AcceleratorParam()
    if params[0].opcode in ["conv2d", "linear", "matmul"]:
        param.matrix_param.CopyFrom(params[0])
        if len(params) > 1:
            param.vector_params.extend(params[1:])
    elif isinstance(params[0], VectorParam):
        param.vector_params.extend(params)
    else:
        assert len(params) == 1, f"{str(params[0].opcode)} does not support fusion"
        if params[0].opcode == "layer_norm":
            param.matrix_param.CopyFrom(params[0])
        if isinstance(params[0], PoolingParam):
            param.pooling_param.CopyFrom(params[0])
        if isinstance(params[0], ReduceParam):
            param.reduce_param.CopyFrom(params[0])
        if isinstance(params[0], ShapeParam):
            param.shape_param.CopyFrom(params[0])
    return param


def _get_size(tensor):
    return reduce(lambda x, y: x * y, tensor.shape)


# HACK a temporary function that make sure inputs and weights don't overlap
# within each operation. Should be removed once we have a proper memory planner.
# Store bias in RRAM and output in reference memory.
def _plan_memory(param: AcceleratorParam):
    param_type = param.WhichOneof("param_type")
    if param_type is not None:
        matrix_param = getattr(param, param_type)
        matrix_param.input.memory.partition = 0
        matrix_param.input.memory.offset = 0
        offset = 2 * _get_size(matrix_param.input)

        if param_type == "matrix_param":
            matrix_param.weight.memory.partition = 0
            matrix_param.weight.memory.offset = offset
            offset += 2 * _get_size(matrix_param.weight)
            if matrix_param.HasField('bias'):
                matrix_param.bias.memory.partition = 1
                matrix_param.bias.memory.offset = offset
                offset += 2 * _get_size(matrix_param.bias)

    for vector_param in param.vector_params:
        if vector_param.HasField("other"):
            vector_param.other.memory.partition = 0
            vector_param.other.memory.offset = offset
            offset += 2 * _get_size(vector_param.other)

    param.output.memory.partition = 0
    param.output.memory.offset = offset


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
