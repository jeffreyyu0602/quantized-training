import logging
from typing import Dict, List

import torch
from torch.fx.graph import map_arg
from torch.fx.node import Node

logger = logging.getLogger(__name__)


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

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

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

        for node in reversed(self.graph.nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        def delete_unused_values(user : Node):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            nodes_to_delete = user_to_last_uses.get(user, [])
            for n in nodes_to_delete:
                env.pop(n.name)

        for node in self.graph.nodes:
            try:
                if node.op == 'placeholder':
                    result = next(args_iter)
                elif node.op == 'get_attr':
                    result = fetch_attr(node.target)
                elif node.op == 'call_function':
                    result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
                elif node.op == 'call_method':
                    self_obj, *args = load_arg(node.args)
                    kwargs = load_arg(node.kwargs)
                    result = getattr(self_obj, node.target)(*args, **kwargs)
                elif node.op == 'call_module':
                    result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            except:
                print(f"Error in node {node}")
                raise

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.value = result.cpu().clone()
            elif isinstance(result, (tuple, list)):
                node.value = [x.cpu().clone() if isinstance(x, torch.Tensor) else x for x in result]
            else:
                node.value = result
                logger.warning(f"Node {node} produced non-tensor output {result}")

            env[node.name] = result

            delete_unused_values(node)

        return load_arg(list(self.graph.nodes)[-1])
