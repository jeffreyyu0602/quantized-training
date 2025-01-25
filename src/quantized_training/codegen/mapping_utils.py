import logging
import operator
import os
import struct
from typing import Callable, Dict

import torch
from torch.fx import Node

from .param_pb2 import (
    MatrixOp,
    Nop,
    PoolingOp,
    ReduceOp,
    ReshapeOp,
    SlicingOp,
    Tensor,
    VectorOp,
)
from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


# Global variable to store the custom save function
_custom_save_function = None

def register_save_function(custom_function):
    """
    Decorator to register a custom function for saving tensors.
    The custom function should accept two arguments: tensor and filename.
    """
    global _custom_save_function
    if not callable(custom_function):
        raise ValueError("The custom function must be callable.")
    _custom_save_function = custom_function
    print(f"Custom save function '{custom_function.__name__}' registered.")
    return custom_function


def _save_tensor(tensor, filename):
    tensor = tensor.float().flatten()
    packed_data = struct.pack(f'{tensor.numel()}f', *tensor.tolist())
    with open(filename, 'wb') as f:
        f.write(packed_data)
    print(f"Writing tensor to {filename}")


def save_tensor(tensor, filename):
    """
    Save the tensor to a file using the custom save function if defined,
    otherwise use the default _save_tensor function.
    """
    if _custom_save_function is not None:
        _custom_save_function(tensor, filename)
    else:
        _save_tensor(tensor, filename)


def _set_tensor_field(field, node, output_dir=None, is_output=False):
    assert isinstance(node, Node) and hasattr(node, 'value'), (
        f"Expected node {node} has value attribute."
    )

    # Reshape can be fused at the input or output of an operation/submodule.
    # If the node's op is call_module, then the reshape is fused at the output
    # of the last operation in the submodule.
    reshape_node = node.meta.get("reshape", None)
    if reshape_node is not None and (node.op != "call_module" or is_output):
        param = ReshapeOp()
        param.name = reshape_node.name
        param.opcode = reshape_node.target.__name__.split(".")[0]
        param.dims.extend(
            reshape_node.args[1]
            if reshape_node.target == torch.ops.aten.permute.default
            else reshape_node.args[1:]
        )
        param.input_sizes.extend(reshape_node.args[0].shape)
        param.output_sizes.extend(node.shape)
        field.reshape.CopyFrom(param)
        if not is_output:
            node = reshape_node.args[0]

    if (getitem_node := node.meta.get("indexing", None)) is not None:
        # TODO duplicate logic. Refactor to a function
        input_shape = getitem_node.args[0].shape
        default_args = [0, None, None, 1][len(getitem_node.args) - 1:]
        dim, start, end, step = list(getitem_node.args[1:]) + default_args
        end = min(end, input_shape[dim])

        param = SlicingOp()
        param.name = getitem_node.name
        param.opcode = getitem_node.target.__name__.split(".")[0]
        param.dim = dim
        param.start = start
        param.end = end
        param.step = step
        param.output_sizes.extend(node.shape)
        field.slicing.CopyFrom(param)
        node = getitem_node.args[0]

    if (dq_scale := node.meta.get("dq_scale", None)) is not None:
        field.scale = dq_scale
        node = node.args[0]

    if (source_node := node.meta.get("source_node", None)) is not None:
        node = source_node

    if output_dir is not None:
        save_tensor(node.value, os.path.join(output_dir, f"{node.name}.bin"))

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
    param = MatrixOp()
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
    if args[2] is not None:
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
    param = MatrixOp()
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
    if len(node.args) > 2:
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
    param = MatrixOp()
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
    param = MatrixOp()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    if len(node.args) > 2:
        _set_tensor_field(param.weight, node.args[2], output_dir)
    if len(node.args) > 3:
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
    return node.target in [
        # Core aten ops
        torch.ops.aten.abs.default,
        torch.ops.aten.acos.default,
        torch.ops.aten.acosh.default,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.asin.default,
        torch.ops.aten.asinh.default,
        torch.ops.aten.atan.default,
        torch.ops.aten.atan2.default,
        torch.ops.aten.atan2.out,
        torch.ops.aten.atanh.default,
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_not.default,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Scalar,
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.ceil.default,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp.Tensor,
        torch.ops.aten.cos.default,
        torch.ops.aten.cosh.default,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.div.Scalar_mode,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Tensor_mode,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.erf.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.expm1.default,
        torch.ops.aten.floor.default,
        torch.ops.aten.fmod.Scalar,
        torch.ops.aten.fmod.Tensor,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.gelu.default,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.isinf.default,
        torch.ops.aten.isnan.default,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.log.default,
        torch.ops.aten.log10.default,
        torch.ops.aten.log1p.default,
        torch.ops.aten.log2.default,
        torch.ops.aten.logical_and.default,
        torch.ops.aten.logical_not.default,
        torch.ops.aten.logical_or.default,
        torch.ops.aten.logical_xor.default,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.maximum.default,
        torch.ops.aten.minimum.default,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.nonzero.default,
        torch.ops.aten.pow.Scalar,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.reciprocal.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.remainder.Scalar,
        torch.ops.aten.remainder.Tensor,
        torch.ops.aten.round.default,
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.sign.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.sinh.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.tan.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.trunc.default,
        torch.ops.aten.where.self,
        # Inplace versions of the above operations
        torch.ops.aten.add_.Scalar,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.mul_.Scalar,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.gelu_.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.relu_.default,
        # Quantization operations
        torch.ops.quantized_ops.dequantize,
        torch.ops.quantized_ops.quantize,
        torch.ops.quantized_ops.vmap,
        # Not in the core aten operator set. Will be removed in the future.
        torch.ops.aten.silu.default,
    ]


@register_annotator("elementwise")
def map_elementwise(node, output_dir):
    if node.op != "call_function" or not _is_elementwise_op(node):
        return None
    param = VectorOp()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    if isinstance(node.args[0], Node):
        _set_tensor_field(param.input, node.args[0], output_dir)
    elif isinstance(node.args[0], (float, int)):
        param.input_scalar = node.args[0]
    else:
        logger.warning(f"Unsupported input type {type(node.args[0])} for {node}")

    if len(node.args) > 1:
        if isinstance(node.args[1], Node):
            _set_tensor_field(param.other, node.args[1], output_dir)
        elif isinstance(node.args[1], (float, int)):
            param.other_scalar = node.args[1]
        else:
            logger.warning(f"Unsupported input type {type(node.args[1])} for {node}")
    return param


@register_annotator("reduce")
def map_reduce(node, output_dir):
    if node.op != "call_function" or node.target not in [
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmin.default,
        torch.ops.aten.amax.default,
        torch.ops.aten.amin.default,
        torch.ops.aten.any.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmin.default,
        torch.ops.aten.cumsum.default,
        torch.ops.aten.max.dim,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.median.default,
        torch.ops.aten.min.dim,
        torch.ops.aten.prod.default,
        torch.ops.aten.prod.dim_int,
        torch.ops.aten.softmax.int,
        torch.ops.aten.sum.dim_IntList,
        torch.ops.quantized_ops.calculate_mx_qparam,
    ]:
        return None
    param = ReduceOp()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    if len(node.args) > 1:
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
    param = PoolingOp()
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
    param = PoolingOp()
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
    param = PoolingOp()
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
    param = ReshapeOp()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    param.dims.extend(node.args[1])
    param.input_sizes.extend(node.args[0].shape)
    param.output_sizes.extend(node.shape)
    return param


@register_annotator("transpose")
def map_transpose(node, output_dir):
    """
    Schema: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
    """
    if node.op != "call_function" or node.target != torch.ops.aten.transpose.int:
        return None
    param = ReshapeOp()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    param.dims.extend(node.args[1:])
    param.input_sizes.extend(node.args[0].shape)
    param.output_sizes.extend(node.shape)
    return param


def _is_nop(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.dropout.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
    ]


def _is_reshape_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ]


def _is_indexing_or_concatenation_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.stack.default,
        torch.ops.aten.cat.default,
    ]


@register_annotator("getitem")
def map_getitem(node, output_dir):
    if node.op != "call_function" or node.target != operator.getitem:
        return None
    from_node = node.args[0]
    assert all(isinstance(t, torch.Tensor) for t in from_node.value)
    param = Nop()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]

    partition = node.meta["memory"]
    offset = partition.start
    for i, tensor in enumerate(from_node.value):
        tensor_param = Tensor()
        tensor_param.node = f"{from_node.name}_{i}"
        tensor_param.dtype = str(tensor.dtype).split(".")[1]
        tensor_param.shape.extend(tuple(tensor.shape))
        tensor_param.memory.partition = partition.partition_id
        tensor_param.memory.offset = offset
        offset += tensor.numel() * dtype_byte_size(tensor.dtype)
        if output_dir is not None:
            save_tensor(tensor, os.path.join(output_dir, f"{from_node.name}_{i}.bin"))
        param.inputs.append(tensor_param)
    return param


def _is_slicing_nop(node: Node) -> bool:
    if node.target != torch.ops.aten.slice.Tensor:
        return False
    input_shape = node.args[0].shape
    default_args = [0, None, None, 1]
    dim, start, end, step = list(node.args[1:]) + default_args[len(node.args) - 1:]
    return start == 0 and end > input_shape[dim] and step == 1


@register_annotator("slice")
def map_slice(node, output_dir):
    if node.op != "call_function" or node.target != torch.ops.aten.slice.Tensor:
        return None
    if _is_slicing_nop(node):
        return None

    input_shape = node.args[0].shape
    default_args = [0, None, None, 1]
    dim, start, end, step = list(node.args[1:]) + default_args[len(node.args) - 1:]
    end = min(end, input_shape[dim])

    param = SlicingOp()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    param.dim = dim
    param.start = start
    param.end = end
    param.step = step
    param.output_sizes.extend(node.shape)
    return param


@register_annotator("select")
def map_select(node, output_dir):
    if node.op != "call_function" or node.target != torch.ops.aten.select.int:
        return None
    if node.args[1] == 0:
        return None
    param = SlicingOp()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    param.dim = node.args[1]
    param.start = node.args[2]
    param.end = node.args[2] + 1
    param.step = 1
    param.output_sizes.extend(node.shape)
    return param


def _can_be_handled_by_mp(node: Node) -> bool:
    """
    The following operations are handled by the memory placement and
    thus require no additional handling:
    """
    if node.target == torch.ops.aten.select.int:
        return node.args[1] == 0

    if node.target == torch.ops.aten.stack.default:
        return len(node.args) == 1 or node.args[1] == 0

    return False


@register_annotator("nop")
def map_nop(node, output_dir):
    if (
        not _is_nop(node) and
        not _is_slicing_nop(node) and
        not _can_be_handled_by_mp(node)
    ):
        logger.warning(f"Unsupported operation {node.name}: {node.target}")
    param = Nop()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    for n in node.all_input_nodes:
        if not isinstance(getattr(n, 'value', None), torch.Tensor):
            continue
        tensor = Tensor()
        _set_tensor_field(tensor, n, output_dir)
        param.inputs.append(tensor)
    return param
