import logging
from typing import Dict, List

import torch
from torch import nn
from torch.fx import GraphModule, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

from .accelerator.map_operation import (
    GEMM_OPS_MAPPING,
    COMPLEX_VECTOR_OPS_MAPPING,
    map_operation
)

logger = logging.getLogger(__name__)


def _get_anchor_ops():
    return set(GEMM_OPS_MAPPING.keys() | COMPLEX_VECTOR_OPS_MAPPING.keys())


class Attribute(nn.Module):
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def forward(self, args, kwargs):
        pass

    def __repr__(self):
        return f"attr: {' -> '.join([str(node) for node in self.nodes])}"


class FusedOperations(nn.Module):
    def __init__(self, nodes: List[Node], args=None):
        super().__init__()
        self.nodes = nodes
        self.args = args
        self.operations = []

    def forward(self, args_list, kwargs_list):
        result = None
        self.args = []
        for i, node in enumerate(self.nodes):
            assert node.op == 'call_function', "Only call_function is supported"
            args = tuple(arg if arg != 'placeholder' else result for arg in args_list[i])
            result = node.target(*args, **kwargs_list[i])
            self.args.append(args)
        return result

    def __repr__(self):
        return f"fused ops: {' -> '.join([str(node) for node in self.nodes])}"


def _check_arg_computed(arg, visited):
    if isinstance(arg, List):
        return all(_check_arg_computed(a, visited) for a in arg)
    return arg in visited or not isinstance(arg, Node)


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.env: Dict[str, Node] = {}

    def load_arg(self, a):
        return torch.fx.graph.map_arg(a, lambda n: self.env[n.name])

    def fetch_attr(self, target: str):
        target_atoms = target.split('.')
        attr_itr = self.mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def propagate(self, *args):
        args_iter = iter(args)

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = self.fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*self.load_arg(node.args),
                                     **self.load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = self.load_arg(node.args)
                kwargs = self.load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](
                    *self.load_arg(node.args), **self.load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            self.env[node.name] = result

        return self.load_arg(list(self.graph.nodes)[-1].args)

    def transform(self):
        anchor_ops = _get_anchor_ops()
        def _is_anchor(node):
            return node.target in anchor_ops

        visited : Dict[Node, None] = {}
        for node in self.graph.nodes:
            if _is_anchor(node):
                fused_ops = [node]
                new_args = [node.args]
                cur_node = node
                while len(cur_node.users) == 1:
                    user = next(iter(cur_node.users))
                    all_args_computed = all(
                        _check_arg_computed(arg, visited) for arg in user.args if arg != cur_node
                    )
                    if _is_anchor(user) or user.target == 'output' or not all_args_computed:
                        break
                    fused_ops.append(user)
                    new_args.append(tuple(arg if arg != cur_node else 'placeholder' for arg in user.args))
                    cur_node = user
                if len(fused_ops) == 1:
                    visited[node] = None
                    continue
                new_kwargs = tuple([n.kwargs for n in fused_ops])
                fused_mod = FusedOperations(fused_ops)
                get_new_node_name = get_new_attr_name_with_prefix('fused_op_')
                node_name = get_new_node_name(self.mod)
                setattr(self.mod, node_name, fused_mod)
                self.modules[node_name] = fused_mod
                with self.graph.inserting_before(node):
                    new_node = self.graph.create_node(
                        'call_module', node_name, (new_args, new_kwargs), {})
                fused_ops[-1].replace_all_uses_with(new_node)
                for node in reversed(fused_ops):
                    self.graph.erase_node(node)
                visited[new_node] = None
            else:
                visited[node] = None
                # TODO: fuse get_attr with shape mutation operations

        self.mod = GraphModule(self.mod, self.graph)

    def get_operations(self):
        all_ops = []
        for node in self.graph.nodes:
            if (
                node.op == 'call_module' and
                isinstance(self.modules[node.target], FusedOperations)
            ):
                all_ops.append(self.modules[node.target])
            elif node.op == 'call_function':
                op = FusedOperations([node], [self.load_arg(node.args)])
                all_ops.append(op)
        return all_ops

    def gen_code(self):
        from .accelerator.build import param_pb2
        params = param_pb2.ModelParams()
        for node in self.graph.nodes:
            if (
                node.op == 'call_module' and
                isinstance(self.modules[node.target], FusedOperations)
            ):
                params.params.append(map_operation(self.modules[node.target]))
            elif node.op == 'call_function':
                op = FusedOperations([node], [self.load_arg(node.args)])
                if (param := map_operation(op)) is not None:
                    params.params.append(param)
        return params

    def gen_compute_graph(self, output_file="compute_graph"):
        nodes = {}
        edges = []
        for node in self.graph.nodes:
            if node.op in ["placeholder", "get_attr"]:
                continue

            label = "{" + str(node) + "|"
            if (
                node.op == "call_module"
                and (mod := self.modules[node.target])
                and isinstance(mod, FusedOperations)
            ):
                label += "&#92;n".join([str(n.target) for n in mod.nodes])
            else:
                label += str(node.target)
            label += "}"
            label = label.replace("<", "\<").replace(">", "\>")

            nodes[node.name] = {
                "label": label,
                "shape": "Mrecord",
            }
            for n in node.users:
                edges.append((node.name, n.name))

        import graphviz
        dot = graphviz.Digraph()

        # Add nodes with attributes to the graph
        for node, attrs in nodes.items():
            dot.node(node, **attrs)

        # Add edges to the graph
        for edge in edges:
            dot.edge(edge[0], edge[1])

        # Render and save the graph
        dot.render(output_file, format='png', cleanup=True)