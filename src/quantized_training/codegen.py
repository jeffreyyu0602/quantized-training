import logging
from typing import Dict, List

import torch
from torch import nn
from torch.fx import GraphModule, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

logger = logging.getLogger(__name__)


ANCHOR_OPS = [
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.convolution.default,
]


class FusedNode(nn.Module):
    def __init__(self, nodes: List[Node]):
        super().__init__()
        self.nodes = nodes

    def forward(self, args_list, kwargs_list):
        result = None
        for i, node in enumerate(self.nodes):
            assert node.op == 'call_function', "Only call_function is supported"
            args = tuple(arg if arg != 'placeholder' else result for arg in args_list[i])
            result = node.target(*args, **kwargs_list[i])
        return result
    
    def __repr__(self):
        return f"FusedNode: {' -> '.join([str(node) for node in self.nodes])}"


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
        def _is_anchor(node):
            return node.target in ANCHOR_OPS

        visited : Dict[Node, None] = {}
        for node in self.graph.nodes:
            if _is_anchor(node):
                fused_ops = [node]
                new_args = [node.args]
                cur_node = node
                while len(cur_node.users) == 1:
                    user = next(iter(cur_node.users))
                    # TODO: if all the args are computed before they are used, we can fuse them
                    if (
                        _is_anchor(user) or
                        user.target == 'output' or
                        any(not arg in visited for arg in user.args if isinstance(arg, Node) and arg != cur_node)
                    ):
                        break
                    fused_ops.append(user)
                    new_args.append(tuple(arg if arg != cur_node else 'placeholder' for arg in user.args))
                    cur_node = user
                if len(fused_ops) == 1:
                    visited[node] = None
                    continue
                new_kwargs = tuple([n.kwargs for n in fused_ops])
                node_module = FusedNode(fused_ops)
                get_new_node_name = get_new_attr_name_with_prefix('fused_node_')
                node_name = get_new_node_name(self.mod)
                setattr(self.mod, node_name, node_module)
                logger.debug(node_name + " " + str(node_module))
                self.modules[node_name] = node_module
                with self.graph.inserting_before(node):
                    new_node = self.graph.create_node(
                        'call_module', node_name, (new_args, new_kwargs), {})
                fused_ops[-1].replace_all_uses_with(new_node)
                for node in reversed(fused_ops):
                    self.graph.erase_node(node)
                visited[new_node] = None
            elif node.op == 'get_attr':
                visited[node] = None
                # TODO: fuse get_attr with shape mutation operations
            else:
                visited[node] = None
                logger.debug(f"ignore non-anchor node: {node.op} {node.name}")

        self.mod = GraphModule(self.mod, self.graph)

    def gen_code(self):
        pass
