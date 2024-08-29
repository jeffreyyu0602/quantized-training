import operator
import os
import struct
from typing import Callable, Dict

import torch
from torch.fx import Node

from .param_pb2 import (
    MatrixParam,
    VectorParam,
    PoolingParam,
    ReduceParam,
    ReshapeParam,
)


def _write_tensor_to_file(tensor, filename):
    tensor = tensor.float().flatten()
    packed_data = struct.pack(f'{tensor.numel()}f', *tensor.tolist())
    with open(filename, 'wb') as f:
        f.write(packed_data)
    print(f"Writing tensor to {filename}")


def _get_module_name(n: Node):
    # example: {
    #    'L__self___sub': ("L['self'].sub", <class '....Sub'>),
    #    'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
    # }
    # get_attr nodes doesn't have nn_module_stack?
    nn_module_stack = n.meta.get("nn_module_stack", {})

    def _normalize_path(n):
        prefix = 0
        # TODO This is non standard behavior and should be removed when we migrate off capture_pre_autograd_graph.
        if n.startswith("L['self']."):
            prefix = len("L['self'].")
        return n[prefix:]

    names = [_normalize_path(n) for n, _ in nn_module_stack.values()]
    return names


def _set_tensor_field(field, node, output_dir):
    assert isinstance(node, Node) and hasattr(node, 'value'), (
        f"Expected node {node} has value attribute. Make sure ShapeProp is called before mapping."
    )

    def _normalize_name(name):
        return name.replace("[", "_").replace("]", "").replace(".", "_")

    module_name = None
    if node.op == "get_attr":
        module_name = _normalize_name(node.target)
    elif node.op == "placeholder":
        if (
            (source_node := node.meta.get("source_node", None)) is not None
            and source_node.op == "get_attr"
        ):
            module_name = _normalize_name(source_node.target)
    elif node.op == "call_function":
        module_names = _get_module_name(node)
        if len(module_names) > 0:
            module_name = _normalize_name(module_names[-1])
    elif node.op == "call_module":
        if (gm := node.meta.get("source_module", None)) is not None:
            first_node = next(n for n in gm.graph.nodes if n.op == "call_function")
            module_names = _get_module_name(first_node)
            if len(module_names) > 0:
                module_name = _normalize_name(module_names[-1]) + "_fused"
    if module_name is not None:
        print(f"{node.name} -> {module_name}")

    tensor = node.value
    if output_dir is not None:
        _write_tensor_to_file(tensor, os.path.join(output_dir, f"{node.name}.bin"))

    field.node = node.name
    if (dtype := node.meta.get("dtype", None)) is not None:
        field.dtype = dtype
    elif (
        (source_node := node.meta.get("source_node", None))
        and (dtype := source_node.meta.get("dtype", None)) is not None
    ):
        field.dtype = dtype
    else:
        field.dtype = str(tensor.dtype).split(".")[1]

    if len(node.shape) > 0:
        field.shape.extend(list(node.shape))
    else:
        field.shape.append(1)

    if (memory := node.meta.get("memory", None)) is not None:
        field.memory.partition = memory.partition_id
        field.memory.offset = memory.start

    if (reshape := node.meta.get("reshape", None)) is not None:
        arg = reshape.args[0]
        if output_dir is not None:
            _write_tensor_to_file(arg.value, os.path.join(output_dir, f"{arg.name}.bin"))
        field.permutation.node = arg.name
        field.permutation.shape.extend(arg.shape)
        field.permutation.opcode = reshape.target.__name__.split(".")[0]
        field.permutation.dims.extend(
            reshape.args[1]
            if reshape.target == torch.ops.aten.permute.default
            else reshape.args[1:]
        )


def _set_repeated_field(field, value):
    if isinstance(value, (tuple, list)):
        field.extend(value)
    elif value is not None:
        field.append(value)


OP_TO_MAPPING_FUNC: Dict[str, Callable] = {}


def register_annotator(op: str):
    def decorator(mapping_func):
        OP_TO_MAPPING_FUNC[op] = mapping_func

    return decorator


@register_annotator("conv2d")
def map_conv2d(node, output_dir):
    """
    Schema: conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
    """
    if node.op != "call_function" or node.target not in [
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d_mx,
    ]:
        return None
    args = [None, None, None, 1, 0, 1, 1]
    args[:len(node.args)] = node.args
    param = MatrixParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    if node.target == torch.ops.aten.conv2d.default:
        _set_tensor_field(param.input, args[0], output_dir)
        _set_tensor_field(param.weight, args[1], output_dir)
    else:
        _set_tensor_field(param.mx_input.input, args[0], output_dir)
        _set_tensor_field(param.mx_input.scale, node.kwargs['scale_inp'], output_dir)
        _set_tensor_field(param.mx_weight.input, args[1], output_dir)
        _set_tensor_field(param.mx_weight.scale, node.kwargs['scale_wt'], output_dir)
    _set_tensor_field(param.bias, args[2], output_dir)
    _set_repeated_field(param.stride, args[3])
    _set_repeated_field(param.padding, args[4])
    _set_repeated_field(param.dilation, args[5])
    param.groups = args[6]
    return param


@register_annotator("linear")
def map_linear(node, output_dir):
    """
    Schema: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    """
    if node.op != "call_function" or node.target not in [
        torch.ops.aten.linear.default,
        torch.ops.quantized_ops.linear_mx,
    ]:
        return None
    param = MatrixParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    if node.target == torch.ops.aten.linear.default:
        _set_tensor_field(param.input, node.args[0], output_dir)
        _set_tensor_field(param.weight, node.args[1], output_dir)
    else:
        _set_tensor_field(param.mx_input.input, node.args[0], output_dir)
        _set_tensor_field(param.mx_input.scale, node.kwargs['scale_inp'], output_dir)
        _set_tensor_field(param.mx_weight.input, node.args[1], output_dir)
        _set_tensor_field(param.mx_weight.scale, node.kwargs['scale_wt'], output_dir)
    _set_tensor_field(param.bias, node.args[2], output_dir)
    return param


@register_annotator("matmul")
def map_matmul(node, output_dir):
    """
    Schema: matmul(Tensor self, Tensor other) -> Tensor
    """
    if node.op != "call_function" or node.target not in [
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.matmul_mx,
    ]:
        return None
    param = MatrixParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    if node.target == torch.ops.aten.matmul.default:
        _set_tensor_field(param.input, node.args[0], output_dir)
        _set_tensor_field(param.weight, node.args[1], output_dir)
    else:
        _set_tensor_field(param.mx_input.input, node.args[0], output_dir)
        _set_tensor_field(param.mx_input.scale, node.kwargs['scale_inp'], output_dir)
        _set_tensor_field(param.mx_weight.input, node.args[1], output_dir)
        _set_tensor_field(param.mx_weight.scale, node.kwargs['scale_wt'], output_dir)
    return param


@register_annotator("layer_norm")
def map_layer_norm(node, output_dir):
    """
    Schema: layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.layer_norm.default:
        return None
    param = MatrixParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_tensor_field(param.weight, node.args[2], output_dir)
    _set_tensor_field(param.bias, node.args[3], output_dir)
    return param


def _is_gemm_op(op: Callable) -> bool:
    return op in [
        torch.ops.aten.conv2d.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
    ]


def _is_elementwise_op(op: Callable) -> bool:
    return op in [
        torch.ops.aten.abs.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.amax.default,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div_.Tensor,
        torch.ops.aten.exp.default,
        torch.ops.aten.floor.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.gelu_.default,
        torch.ops.aten.log.default,
        torch.ops.aten.log2.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.pow.Scalar,
        torch.ops.aten.reciprocal.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub_.Tensor,
        torch.ops.aten.tanh.default,
        torch.ops.quantized_ops.dequantize_symmetric,
        torch.ops.quantized_ops.quantize_symmetric,

    ]


@register_annotator("elementwise")
def map_elementwise(node, output_dir):
    """
    Schema:
    add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    mul.Tensor(Tensor self, Tensor other) -> Tensor
    div.Tensor(Tensor self, Tensor other) -> Tensor
    exp(Tensor self) -> Tensor
    log(Tensor self) -> Tensor
    reciprocal(Tensor self) -> Tensor
    sqrt(Tensor self) -> Tensor
    tanh(Tensor self) -> Tensor
    relu(Tensor self) -> Tensor
    gelu(Tensor self, *, str approximate='none') -> Tensor
    """
    if node.op != "call_function" or not _is_elementwise_op(node.target):
        return None
    param = VectorParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    if isinstance(node.args[0], Node):
        _set_tensor_field(param.input, node.args[0], output_dir)
    else:
        param.input_scalar = node.args[0]

    if len(node.args) > 1:
        if isinstance(node.args[1], Node):
            _set_tensor_field(param.other, node.args[1], output_dir)
        elif node.target == torch.ops.aten.amax.default:
            param.dim.extend(node.args[1])
        else:
            param.other_scalar = node.args[1]

    # TODO: do not overload opcode. Add a new field called dtype for quantize/dequantize operations
    if node.target == torch.ops.quantized_ops.quantize_symmetric:
        param.opcode += "_to_" + node.args[2]
    elif node.target == torch.ops.quantized_ops.dequantize_symmetric:
        param.opcode += "_from_" + node.args[2]
    return param


@register_annotator("reduce")
def map_reduce(node, output_dir):
    if node.op != "call_function" or node.target not in [
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.max.dim,
        torch.ops.aten.mean.dim,
        torch.ops.aten._softmax.default,
        torch.ops.aten.softmax.int,
    ]:
        return None
    param = ReduceParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_repeated_field(param.dim, node.args[1])
    return param


@register_annotator("avg_pool2d")
def map_avg_pool2d(node, output_dir):
    """
    Schema: avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.avg_pool2d.default:
        return None
    args = [None, None, [], 0, False, True, None]
    args[:len(node.args)] = node.args
    param = PoolingParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, args[0], output_dir)
    _set_repeated_field(param.kernel_size, args[1])
    _set_repeated_field(param.stride, args[2])
    _set_repeated_field(param.padding, args[3])
    param.ceil_mode = args[4]
    param.count_include_pad = args[5]
    param.divisor_override = args[6]
    return param


@register_annotator("adaptive_avg_pool2d")
def map_adaptive_avg_pool2d(node, output_dir):
    """
    Schema: adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.adaptive_avg_pool2d.default:
        return None
    param = PoolingParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_repeated_field(param.output_size, node.args[1])
    return param


@register_annotator("max_pool2d")
def map_max_pool2d(node, output_dir):
    """
    Schema: max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.max_pool2d.default:
        return None
    args = [None, None, [], 0, 1, False]
    args[:len(node.args)] = node.args
    param = PoolingParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, args[0], output_dir)
    _set_repeated_field(param.kernel_size, args[1])
    _set_repeated_field(param.stride, args[2])
    _set_repeated_field(param.padding, args[3])
    _set_repeated_field(param.dilation, args[4])
    param.ceil_mode = args[5]
    return param


@register_annotator("permute")
def map_permute(node, output_dir):
    """
    Schema: permute(Tensor(a) self, int[] dims) -> Tensor(a)
    """
    if node.op != "call_function" or node.target != torch.ops.aten.permute.default:
        return None
    param = ReshapeParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_repeated_field(param.dims, node.args[1])
    return param


@register_annotator("transpose")
def map_transpose(node, output_dir):
    """
    Schema: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
    """
    if node.op != "call_function" or node.target != torch.ops.aten.transpose.int:
        return None
    param = ReshapeParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_repeated_field(param.dims, node.args[1:])
    return param


def _is_nop(op: Callable) -> bool:
    return op in [
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        # TODO: remove?
        torch.ops.aten.dropout.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        operator.getitem,
    ]
