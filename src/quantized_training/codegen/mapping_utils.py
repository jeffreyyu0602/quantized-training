import logging
import operator
import os
import random
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
    NopParam,
    Tensor,
)

logger = logging.getLogger(__name__)


def _save_tensor(tensor, filename):
    tensor = tensor.float().flatten()
    packed_data = struct.pack(f'{tensor.numel()}f', *tensor.tolist())
    with open(filename, 'wb') as f:
        f.write(packed_data)
    print(f"Writing tensor to {filename}")


def _set_tensor_field(field, node, output_dir=None, is_output=False):
    assert isinstance(node, Node) and hasattr(node, 'value'), (
        f"Expected node {node} has value attribute. Make sure ShapeProp is called before mapping."
    )

    # If there is a reshape operation, we take the input from the reshape args
    if node.op != "call_module" or is_output:
        if (reshape := node.meta.get("reshape", None)) is not None:
            field.permutation.node = reshape.name
            field.permutation.opcode = reshape.target.__name__.split(".")[0]
            field.permutation.dims.extend(
                reshape.args[1]
                if reshape.target == torch.ops.aten.permute.default
                else reshape.args[1:]
            )
            field.permutation.input_shape.extend(reshape.args[0].shape)
            field.permutation.output_shape.extend(node.shape)
            if not is_output:
                node = reshape.args[0]

    # The reshape op can be further fused with a dequantize op.
    if (dq_scale := node.meta.get("dq_scale", None)) is not None:
        field.scale = dq_scale
        node = node.args[0]

    if (source_node := node.meta.get("source_node", None)) is not None:
        node = source_node

    if output_dir is not None:
        _save_tensor(node.value, os.path.join(output_dir, f"{node.name}.bin"))

    field.node = node.name
    if (dtype := node.meta.get("dtype", None)) is not None:
        field.dtype = dtype
    else:
        field.dtype = str(node.value.dtype).split(".")[1]

    if len(node.shape) > 0:
        field.shape.extend(list(node.shape))
    else:
        field.shape.append(1)

    if (memory := node.meta.get("memory", None)) is not None:
        field.memory.partition = memory.partition_id
        field.memory.offset = memory.start


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
        torch.ops.quantized_ops.conv2d_mx.default,
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
        torch.ops.quantized_ops.linear_mx.default,
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
        torch.ops.quantized_ops.matmul_mx.default,
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


def _is_gemm_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.conv2d_mx.default,
        torch.ops.quantized_ops.linear_mx.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ]


def _is_elementwise_op(node: Node) -> bool:
    """The function test if a callable is elementwise by applying a random input
    tensor and modifying one element of the input tensor. If the output of the
    function changes only at the modified index, then the function is elementwise.

    Parameters:
    - node: The node to be tested.

    Returns:
    - True if the function is elementwise, False otherwise.
    """

    if node.target in [
        torch.ops.quantized_ops.dequantize_symmetric,
        torch.ops.quantized_ops.quantize_symmetric,
    ]:
        return True

    if len(node.all_input_nodes) > 2:
        return False

    # Get the shapes of all inputs and determine the broadcasted shape
    input_shapes = [n.meta['val'].shape for n in node.all_input_nodes]
    try:
        broadcasted_shape = torch.broadcast_shapes(*input_shapes)
    except RuntimeError:
        return False

    value_remap = {}
    def arg_transform(n):
        if n not in value_remap:
            value_remap[n] = torch.randn(broadcasted_shape)
        return value_remap[n].clone()

    args, kwargs = torch.fx.node.map_arg((node.args, node.kwargs), arg_transform)
    orig_output = node.target(*args, **kwargs)

    for arg in node.all_input_nodes:
        input_tensor = value_remap[arg]

        # Output shape should be the same as the input shape and the output must be different
        if input_tensor.shape != orig_output.shape or torch.all(input_tensor == orig_output):
            return False

        # Modify one element of the input tensor. TODO This might fail for some input values
        indices = tuple(random.randint(0, dim_size - 1) for dim_size in input_tensor.shape)
        new_input = input_tensor.clone()
        new_input[indices] *= -1.0

        new_args, new_kwargs = torch.fx.node.map_arg(
            (node.args, node.kwargs),
            lambda n: arg_transform(n) if n != arg else new_input,
        )
        new_output = node.target(*new_args, **new_kwargs)

        # Check if the output is different only at the modified index
        changed_indices = torch.nonzero(orig_output != new_output, as_tuple=False)

        if changed_indices.size(0) != 1 or tuple(changed_indices[0].tolist()) != indices:
            return False

    return True


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
    if node.op != "call_function" or not _is_elementwise_op(node):
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
        else:
            param.other_scalar = node.args[1]

    # TODO Add a new field called dtype for quantize/dequantize operations
    if len(node.args) > 2 and node.args[2] is not None:
        if node.target == torch.ops.quantized_ops.quantize_symmetric:
            param.opcode += "_to_" + node.args[2]
        elif node.target == torch.ops.quantized_ops.dequantize_symmetric:
            param.opcode += "_from_" + node.args[2]
    return param


@register_annotator("reduce")
def map_reduce(node, output_dir):
    if node.op != "call_function" or node.target not in [
        torch.ops.aten.amax.default,
        torch.ops.aten.max.dim,
        torch.ops.aten.mean.dim,
        torch.ops.aten.softmax.int,
        torch.ops.aten.sum.dim_IntList,
        torch.ops.quantized_ops.calculate_mx_qparam,
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


def _is_nop(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        # TODO: remove?
        torch.ops.aten.dropout.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.reshape.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        operator.getitem,
    ]


@register_annotator("nop")
def map_nop(node, output_dir):
    if node.op != "call_function" or not _is_nop(node) and node.target not in [
        torch.ops.aten.cat.default,
        torch.ops.aten.select.int,
        torch.ops.aten.stack.default,
    ]:
        logger.warning(f"Unsupported operation {node.name}: {node.target}")
    param = NopParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    for n in node.all_input_nodes:
        tensor = Tensor()
        _set_tensor_field(tensor, n)
        param.inputs.append(tensor)
    return param
