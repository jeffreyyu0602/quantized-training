import copy
import logging
import math
import numpy as np
import operator
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.fx import Node
from torch.fx.operator_schemas import normalize_function

from .param_pb2 import Argument, OpOverload, Tensor

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


_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


def _write_numpy(np_array: np.ndarray, filename: str, shape: tuple):
    """Worker function: write the tensor and log when done."""
    try:
        np_array.tofile(filename)
        logger.info(f"✅ Saved tensor {shape} -> {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save tensor {shape} -> {filename}: {e}")


def _save_tensor(tensor: torch.Tensor, filename: str):
    """Asynchronously save tensor to a binary file."""
    t = tensor.detach().cpu().contiguous().to(torch.float32)
    np_array = t.numpy()
    _executor.submit(_write_numpy, np_array, filename, tuple(t.shape))


def save_tensor(tensor, filename):
    """
    Save the tensor to a file using the custom save function if defined,
    otherwise use the default _save_tensor function.
    """
    if _custom_save_function is not None:
        _custom_save_function(tensor, filename)
    else:
        _save_tensor(tensor, filename)


def _apply_transform(node, key, field, is_output=False):
    if (fused_op := node.meta.get(key)) is not None:
        fused_op.meta["tiled_shapes"] = node.meta.get("_tiled_shapes")
        fused_op.meta["scratchpad_map"] = node.meta.get("_scratchpad_map")
        field.CopyFrom(map_node(fused_op))
        if key == "reshape":
            field.kwargs["output_shape"].int_list.values.extend(node.shape)
        return fused_op.args[0] if not is_output else node
    return node


def _set_meminfo(field, segment):
    field.partition = int(segment.memory_space)
    field.address = int(segment.start)


def set_tensor_field(field, node, output_dir=None, is_output=False):
    if not isinstance(node, Node) or not hasattr(node, 'value'):
        raise TypeError(f"Expected node with value attribute, got {node!r}")

    tiled_shapes = node.meta.get("_tiled_shapes")
    scratchpad_map = node.meta.get("_scratchpad_map")

    # Apply transformations
    if node.op != "call_module" or is_output:
        node = _apply_transform(node, "reshape", field.reshape, is_output)
    node = _apply_transform(node, "dequantize", field.dequant)

    if (source_node := node.meta.get("source_node")) is not None:
        node = source_node

    if tiled_shapes is not None and node in tiled_shapes:
        field.tiled_shape.extend(tiled_shapes[node])
    if scratchpad_map is not None and node in scratchpad_map:
        _set_meminfo(field.scratchpad, scratchpad_map[node])

    # Tensor properties
    field.node = node.name
    field.shape.extend(node.shape or [1])

    if (dtype := node.meta.get("dtype")) is not None:
        field.dtype = dtype
    else:
        field.dtype = str(node.value.dtype).split(".")[1]

    if (memory := node.meta.get("memory")) is not None:
        _set_meminfo(field.memory, memory)

    if output_dir is not None:
        save_tensor(node.value, os.path.join(output_dir, f"{node.name}.bin"))


def set_output_field(param, node, output_dir):
    if isinstance(node.value, torch.Tensor):
        node.meta["_tiled_shapes"] = node.meta.get("tiled_shapes")
        node.meta["_scratchpad_map"] = node.meta.get("scratchpad_map")
        set_tensor_field(param.output, node, output_dir, True)
        node.meta.pop("_tiled_shapes", None)
        node.meta.pop("_scratchpad_map", None)
    elif isinstance(node.value, (tuple, list)):
        if (memory := node.meta.get("memory")) is not None:
            memory_copy = copy.copy(memory)
            output_sizes = node.meta["output_sizes"]

        if (scratchpad_map := node.meta.get("scratchpad_map")) is not None:
            scratchpad_copy = copy.copy(scratchpad_map[node])
            tiled_output_sizes = node.meta["tiled_output_sizes"]

        if (tiled_shape := node.meta.get("tiled_shapes")) is not None:
            shapes = tiled_shape[node]

        dtypes = node.meta.get("dtype", [None] * len(node.value))

        for i, t in enumerate(node.value):
            tensor = Tensor(
                node=f"{node.name}_{i}",
                shape=list(t.shape),
                dtype=dtypes[i] or str(t.dtype).split(".")[1],
            )

            if memory is not None:
                _set_meminfo(tensor.memory, memory_copy)
                memory_copy.start += output_sizes[i]

            if scratchpad_map is not None:
                _set_meminfo(tensor.scratchpad, scratchpad_copy)
                scratchpad_copy.start += tiled_output_sizes[i]

            if tiled_shape is not None:
                tensor.tiled_shape.extend(shapes[i])

            if output_dir is not None:
                save_tensor(t, os.path.join(output_dir, f"{node.name}_{i}.bin"))

            param.outputs.tensors.append(tensor)
    else:
        logger.warning(f"Unsupported output type: {type(node.value)}")


def convert_arg(value, output_dir=None) -> Argument:
    """
    Converts an argument (which could be a Tensor, list, int, float, etc.)
    into an Argument protobuf.
    """
    arg = Argument()

    if isinstance(value, torch.fx.Node):
        if isinstance(value.value, torch.Tensor):
            set_tensor_field(arg.tensor, value, output_dir)
        elif isinstance(value.value, (tuple, list)):
            arg.tensor_list.tensors.extend([
                Tensor(node=f"{value.name}_{i}") for i in range(len(value.value))
            ])
    elif isinstance(value, int):
        arg.int_value = value
    elif isinstance(value, float):
        arg.float_value = value
    elif isinstance(value, bool):
        arg.bool_value = value
    elif isinstance(value, str):
        arg.str_value = value
    elif isinstance(value, (
        torch.dtype, torch.layout, torch.device, torch.memory_format
    )):
        arg.str_value = str(value).split(".")[-1]
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, torch.fx.Node) or x is None for x in value):
            arg.tensor_list.tensors.extend([
                convert_arg(x).tensor if x is not None else Tensor(is_none=True)
                for x in value
            ])
        elif all(isinstance(x, int) for x in value):
            arg.int_list.values.extend(value)
        elif all(isinstance(x, bool) for x in value):
            arg.bool_list.values.extend(value)
        elif all(isinstance(x, (int, float, bool)) for x in value):
            arg.scalar_list.values.extend(value)
        else:
            raise TypeError(f"Unsupported list value: {value}")
    else:
        raise TypeError(f"Unsupported arg type: {type(value)}")

    return arg


def map_node(node: torch.fx.Node, output_dir=None) -> OpOverload:
    """
    Converts a torch.fx.Node into an OpOverload protobuf message.
    """
    if hasattr(node.target, "_schema"):
        target = node.target._schema.name.split('::')[1]
    else:
        target = str(node.target)

    op_overload = OpOverload(
        name=node.name,
        op=node.op,
        target=target,
    )

    if is_nop(node) or is_addressing_op(node) or node.target == operator.getitem:
        op_overload.op = "nop"

    if node.target == torch.ops.aten.pad.default:
        op_overload.op = "cpu"

    new_args_and_kwargs = normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )

    if new_args_and_kwargs is not None:
        args, kwargs = new_args_and_kwargs.args, new_args_and_kwargs.kwargs
    else:
        args, kwargs = node.args, node.kwargs

    # Pass L2 tiling metadata to input nodes
    for n in node.all_input_nodes:
        n.meta["_tiled_shapes"] = node.meta.get("tiled_shapes")
        n.meta["_scratchpad_map"] = node.meta.get("scratchpad_map")

    # Convert positional arguments
    for arg in args:
        op_overload.args.append(convert_arg(arg))

    # Convert keyword arguments
    for key, value in kwargs.items():
        if not "qmap" in key and value is not None:
            op_overload.kwargs[key].CopyFrom(convert_arg(value, output_dir))

    for n in node.all_input_nodes:
        n.meta.pop("_tiled_shapes", None)
        n.meta.pop("_scratchpad_map", None)

    if "l2_tiling" in node.meta:
        op_overload.kwargs["l2_tiling"].int_list.values.extend(node.meta["l2_tiling"])

    return op_overload


aten = torch.ops.aten


def is_gemm_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.conv2d.default,
        torch.ops.quantized_ops.linear.default,
        torch.ops.quantized_ops.conv2d_mx.default,
        torch.ops.quantized_ops.linear_mx.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ]


def is_conv2d(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d.default,
        torch.ops.quantized_ops.conv2d_mx.default
    ]


def is_depthwise_conv(node: Node) -> bool:
    return (
        node.target in [
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_ops.conv2d_mx.default
        ] and
        len(node.args) == 7 and
        node.args[6] != 1
    )


def is_linear(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.linear.default,
        torch.ops.quantized_ops.linear.default,
        torch.ops.quantized_ops.linear_mx.default,
    ]


def is_fc(node: Node) -> bool:
    return (
        (is_linear(node) or is_matmul(node))
        and math.prod(node.args[0].shape[:-1]) == 1
    )


def is_matmul(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ]


def is_pooling(node: Node) -> bool:
    return node.target in [
        # Core Aten IR
        aten._adaptive_avg_pool2d,
        aten._adaptive_avg_pool3d,
        aten.adaptive_avg_pool1d,
        aten.avg_pool1d,
        aten.avg_pool2d,
        aten.avg_pool3d,
        aten.max_pool2d_with_indices,
        aten.max_pool3d_with_indices,
        # export_for_training IR
        aten.adaptive_avg_pool2d.default,
        aten.avg_pool2d.default,
        aten.max_pool2d.default,
    ]


def is_elementwise_op(node: Node) -> bool:
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
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
        torch.ops.quantized_ops.vmap.default,
        # Not in the core aten operator set. Will be removed in the future.
        torch.ops.aten.silu.default,
        torch.ops.aten.silu_.default,
    ]


def is_reshape_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ]


def is_indexing_or_concatenation_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.stack.default,
        torch.ops.aten.cat.default,
    ]


def is_prunable_op(node: Node) -> bool:
    """Operations that can be safely deleted from fx.Graph."""
    # A slice from 0 to the end of the input tensor
    if node.target == torch.ops.aten.slice.Tensor:
        default_args = [0, None, None, 1]
        dim, start, end, step = list(node.args[1:]) + default_args[len(node.args) - 1:]
        if start != 0 or step != 1:
            return False
        if hasattr(node.args[0], "shape"):
            return end > node.args[0].shape[dim]
        return end == 0x7fffffffffffffff

    # A select operation that selects the entire tensor
    if node.target == torch.ops.aten.select.int:
        return node.args[0].shape[node.args[1]] == 1

    if node.target == torch.ops.aten.expand.default:
        return all(x == 1 or x == -1 for x in node.args[1])

    return False


def is_nop(node: Node) -> bool:
    """
    The following operations do not require any computation nor handling
    on the memory placement side. Generate a NOP instruction for these ops
    to keep the compute graph intact.
    """
    if is_prunable_op(node):
        return True

    return node.target in [
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.copy_.default,
        torch.ops.aten.dropout.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.to.dtype,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
    ]


def is_addressing_op(node: Node) -> bool:
    """
    The following operations are handled by the memory placement and
    thus require no additional handling:
    """
    if node.target == torch.ops.aten.select.int:
        return all(d == 1 for d in node.args[0].value.shape[:node.args[1]])

    if node.target in [torch.ops.aten.stack.default, torch.ops.aten.cat.default]:
        return len(node.args) == 1 or node.args[1] == 0

    return False
