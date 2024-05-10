import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.fx import GraphModule, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

logger = logging.getLogger(__name__)


aten = torch.ops.aten

# GEMM operations
GEMM_OPS_MAPPING = {
    # convolution(Tensor input, Tensor weight, Tensor? bias,
    # SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool
    # transposed, SymInt[] output_padding, SymInt groups) -> Tensor
    aten.convolution.default: "conv",
    # addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    aten.addmm.default: "gemm",
    # mm(Tensor self, Tensor mat2) -> Tensor
    aten.mm.default:    "gemm",
    # bmm(Tensor self, Tensor mat2) -> Tensor
    aten.bmm.default:   "gemm",
}

# Vector operations that can be fused with GEMM
VECTOR_OPS_MAPPING = {
    # add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    aten.add.Tensor: "vec_add",
    # sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    aten.sub.Tensor: "vec_sub",
    # mul.Tensor(Tensor self, Tensor other) -> Tensor
    aten.mul.Tensor: "vec_mul",
    # div.Tensor(Tensor self, Tensor other) -> Tensor
    aten.div.Tensor: "vec_div",
    # exp(Tensor self) -> Tensor
    aten.exp.default: "vec_exp",
    # log(Tensor self) -> Tensor
    aten.log.default: "vec_log",
    # reciprocal(Tensor self) -> Tensor
    aten.reciprocal.default: "vec_recip",
    # sqrt(Tensor self) -> Tensor
    aten.sqrt.default: "vec_sqrt",
    # tanh(Tensor self) -> Tensor
    aten.tanh.default: "vec_tanh",
    # relu(Tensor self) -> Tensor
    aten.relu.default: "vec_relu",
    # gelu(Tensor self, *, str approximate=’none’) -> Tensor
    aten.gelu.default: "vec_gelu",
}

REDUCE_OPS_MAPPING = {
    # sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    aten.sum.dim_IntList: "vec_reduce_sum",
    # max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    aten.max.dim:         "vec_reduce_max",
    # mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    aten.mean.dim:        "vec_reduce_mean",
}

# Vector operations that can not be fused with GEMM
COMPLEX_VECTOR_OPS_MAPPING = {
    # _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    aten._softmax.default: "vec2_softmax",
    # native_layer_norm(Tensor input, SymInt[] normalized_shape,
    # Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
    aten.native_layer_norm.default: "vec2_layer_norm",
    # avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[],
    # int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True,
    # int? divisor_override=None) -> Tensor
    aten.avg_pool2d.default: "vec2_avg_pool2d",
    # max_pool2d_with_indices(Tensor self, int[2] kernel_size,
    # int[2] stride=[], int[2] padding=0, int[2] dilation=1,
    # bool ceil_mode=False) -> (Tensor, Tensor)
    aten.max_pool2d_with_indices.default: "vec2_max_pool2d",
}

def get_anchor_ops():
    return set(GEMM_OPS_MAPPING.keys() | COMPLEX_VECTOR_OPS_MAPPING.keys())


def map_operations(node, args):
    # TODO: iteratively replace all tensor arguments with their shapes and node name
    args = [tuple(arg.shape) if torch.is_tensor(arg) else arg for arg in args]
    if node.target == aten.convolution.default:
        return {
            "name": node.name,
            "type": "conv",
            "input": args[0],
            "weight": args[1],
            "bias": args[2],
            "stride": args[3],
            "padding": args[4],
            "dilation": args[5],
            "transposed": args[6],
        }
    elif node.target == aten.addmm.default:
        return {
            "name": node.name,
            "type": "gemm",
            "input": args[1],
            "weight": args[2],
            "bias": args[0],
        }
    elif node.target in [aten.mm.default, aten.bmm.default]:
        return {
            "name": node.name,
            "type": "gemm",
            "input": args[0],
            "weight": args[1],
            "bias": None,
        }
    elif node.target in VECTOR_OPS_MAPPING:
        return {
            "name": node.name,
            "type": VECTOR_OPS_MAPPING[node.target],
            "input": args[0],
            "other": args[1] if len(args) > 1 else None,
        }
    elif node.target in REDUCE_OPS_MAPPING:
        return {
            "name": node.name,
            "type": REDUCE_OPS_MAPPING[node.target],
            "input": args[0],
            "dim": args[1],
            "keepdim": args[2],
        }
    elif node.target == aten._softmax.default:
        return {
            "name": node.name,
            "type": COMPLEX_VECTOR_OPS_MAPPING[node.target],
            "input": args[0],
            "dim": args[1],
            "half_to_float": args[2],
        }
    elif node.target == aten.native_layer_norm.default:
        return {
            "name": node.name,
            "type": COMPLEX_VECTOR_OPS_MAPPING[node.target],
            "input": args[0],
            "normalized_shape": args[1],
            "weight": args[2],
            "bias": args[3],
            "eps": args[4],
        }
    elif node.target == aten.avg_pool2d.default:
        default_args = [None, None, [], 0, False, True, None]
        default_args[:len(args)] = args
        return {
            "name": node.name,
            "type": COMPLEX_VECTOR_OPS_MAPPING[node.target],
            "input": default_args[0],
            "kernel_size": default_args[1],
            "stride": default_args[2],
            "padding": default_args[3],
            "ceil_mode": default_args[4],
            "count_include_pad": default_args[5],
            "divisor_override": default_args[6],
        }
    elif node.target == aten.max_pool2d_with_indices.default:
        default_args = [None, None, [], 0, 1, False]
        default_args[:len(args)] = args
        return {
            "name": node.name,
            "type": COMPLEX_VECTOR_OPS_MAPPING[node.target],
            "input": default_args[0],
            "kernel_size": default_args[1],
            "stride": default_args[2],
            "padding": default_args[3],
            "dilation": default_args[4],
            "ceil_mode": default_args[5],
        }
    else:
        return {
            "name": node.name,
            "type": "unknown",
            "target": str(node.target),
            "args": args,
        }


class Attribute(nn.Module):
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def forward(self, args, kwargs):
        pass

    def __repr__(self):
        return f"attr: {' -> '.join([str(node) for node in self.nodes])}"


class FusedOperations(nn.Module):
    def __init__(self, nodes: List[Node]):
        super().__init__()
        self.nodes = nodes
        self.operations = []

    def forward(self, args_list, kwargs_list):
        result = None
        self.operations = []
        for i, node in enumerate(self.nodes):
            assert node.op == 'call_function', "Only call_function is supported"
            args = tuple(arg if arg != 'placeholder' else result for arg in args_list[i])
            result = node.target(*args, **kwargs_list[i])
            self.operations.append(map_operations(node, args))
        return result
    
    def get_operations(self):
        return self.operations

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
        anchor_ops = get_anchor_ops()
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

    def gen_code(self):
        all_ops = []
        for node in self.graph.nodes:
            if (
                node.op == 'call_module' and
                isinstance(self.modules[node.target], FusedOperations)
            ):
                all_ops.append(self.modules[node.target].get_operations())
            elif node.op == 'call_function':
                all_ops.append([map_operations(node, self.load_arg(node.args))])
        return all_ops
