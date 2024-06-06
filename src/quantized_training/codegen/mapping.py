import os
from typing import List, Dict

import graphviz
import torch
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx import Node, GraphModule

from .mapping_utils import OP_TO_MAPPING_FUNC, _is_elementwise_op, _is_gemm_op, _is_nop
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
        # 2) the user is a NOP operation
        # 3) OR the root node is a GEMM or elwise op and the user is a elwise operation
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


def map_operation(nodes: List[Node], name, output_dir):
    params = []
    for node in nodes:
        param = None
        for mapping_func in OP_TO_MAPPING_FUNC.values():
            param = mapping_func(node, output_dir)
            if param is not None:
                params.append(param)
                break
        if param is None:
            print(f"Unsupported operation {node.name}: {node.target}")

    if len(params) == 0:
        return None

    from .param_pb2 import (
        AcceleratorParam,
        PoolingParam,
        ReduceParam,
        ShapeParam,
        VectorParam,
    )

    param = AcceleratorParam()
    param.name = name

    if params[0].opcode in ["conv2d", "linear", "matmul"]:
        param.matrix_param.CopyFrom(params[0])
        if len(params) > 1:
            param.vector_params.extend(params[1:])
            param.fused = True
        return param

    if isinstance(params[0], VectorParam):
        param.vector_params.extend(params)
        return param

    assert len(params) == 1, f"{str(node.target)} cannot be fused with other operations"

    if params[0].opcode == "layer_norm":
        param.matrix_param.CopyFrom(params[0])
    if isinstance(params[0], PoolingParam):
        param.pooling_param.CopyFrom(params[0])
    if isinstance(params[0], ReduceParam):
        param.reduce_param.CopyFrom(params[0])
    if isinstance(params[0], ShapeParam):
        param.shape_param.CopyFrom(params[0])
    return param


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    from .param_pb2 import ModelParams
    params = ModelParams()

    sp = ShapeProp(model)
    sp.propagate(*args)
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        param = None
        if node.op == 'call_module':
            gm = named_modules[node.target]
            if isinstance(gm, torch.fx.GraphModule):
                gm_args = sp.load_arg(node.args)
                ShapeProp(gm).propagate(*gm_args)
                nodes = [n for n in gm.graph.nodes if n.op == 'call_function']
                param = map_operation(nodes, node.name, output_dir)
        elif node.op == 'call_function':
            param = map_operation([node], node.name, output_dir)

        if param is not None:
            params.params.append(param)
    return params


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
