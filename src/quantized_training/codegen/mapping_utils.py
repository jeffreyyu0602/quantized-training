import logging
import operator
import os
import struct

import torch
from torch.fx import Node
from torch.fx.operator_schemas import normalize_function

from .param_pb2 import Argument, Memory, OpOverload, Tensor

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
    t = tensor.float().flatten()
    packed_data = struct.pack(f'{t.numel()}f', *t.tolist())
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


def set_tensor_field(field, node, output_dir=None, is_output=False):
    assert isinstance(node, Node) and hasattr(node, 'value'), (
        f"Expected node {node} has value attribute."
    )

    original_node = node

    if (
        (reshape_node := node.meta.get("reshape")) is not None and
        not (node.op == "call_module" and not is_output)
    ):
        field.reshape.CopyFrom(map_node(reshape_node))
        field.reshape.kwargs["output_shape"].int_list.values.extend(node.shape)
        node = reshape_node.args[0] if not is_output else node

    if (getitem_node := node.meta.get("slicing")) is not None:
        field.reshape.CopyFrom(map_node(getitem_node))
        field.reshape.kwargs["output_shape"].int_list.values.extend(node.shape)
        node = getitem_node.args[0]

    if (dequantize_scale := node.meta.get("dq_scale")) is not None:
        field.scale = dequantize_scale
        node = node.args[0]

    if (source_node := node.meta.get("source_node")) is not None:
        node = source_node

    if (tiled_shape := original_node.meta.get("shape")) is not None:
        field.node = node.name + "_tiled"
        field.shape.extend(list(tiled_shape))
    else:
        field.node = node.name
        field.shape.extend(list(node.shape) or [1])

    if (dtype := node.meta.get("dtype")) is not None:
        field.dtype = dtype
    else:
        field.dtype = str(node.value.dtype).split(".")[1]

    if (memory := node.meta.get("memory")) is not None:
        field.memory.partition = memory.partition_id
        field.memory.address = memory.start
    elif is_output:
        print(f"Warning: Node {node.name} has no memory allocation.")
        print(node)

    if (scratchpad := original_node.meta.get("scratchpad")) is not None:
        field.scratchpad.offset = scratchpad.start

    if tiled_shape is not None and output_dir is not None:
        n = len(tiled_shape)
        slices = tuple(
            [slice(None)] * (node.value.ndim - n) + [slice(0, s) for s in tiled_shape]
        )
        save_tensor(
            node.value[slices], os.path.join(output_dir, f"{node.name}_tiled.bin")
        )
    elif output_dir is not None:
        save_tensor(node.value, os.path.join(output_dir, f"{node.name}.bin"))


def set_output_field(param, node, output_dir):
    if isinstance(node.value, torch.Tensor):
        if "tiled_shapes" in node.meta:
            node.meta["shape"] = node.meta["tiled_shapes"].get(node)

        if "scratchpad_mem" in node.meta:
            node.meta["scratchpad"] = node.meta["scratchpad_mem"].get(node)

        set_tensor_field(param.output, node, output_dir, True)
    
        node.meta.pop("shape", None)
        node.meta.pop("scratchpad", None)
    elif isinstance(node.value, (tuple, list)):
        if (memory := node.meta.get("memory")) is not None:
            partition = memory.partition_id
            address = memory.start

        if (scratchpad_mem := node.meta.get("scratchpad_mem")) is not None:
            offset = scratchpad_mem[node].start

        if (tiled_shape := node.meta.get("tiled_shapes")) is not None:
            shapes = tiled_shape[node]

        for i, t in enumerate(node.value):
            tensor = Tensor(
                node=f"{node.name}_{i}_tiled" if tiled_shape else f"{node.name}_{i}",
                shape=shapes[i] if tiled_shape else list(t.shape),
                dtype=node.meta["dtype"][i] or str(t.dtype).split(".")[1],
            )

            if memory is not None:
                tensor.memory.partition = partition
                tensor.memory.address = int(address)
                address += node.meta.get("output_sizes")[i]

            if scratchpad_mem is not None:
                tensor.scratchpad.offset = int(offset)
                offset += node.meta.get("output_sizes")[i]

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

    if _is_nop(node) or is_addressing_op(node) or node.target == operator.getitem:
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

    # Convert positional arguments
    for arg in args:
        op_overload.args.append(convert_arg(arg))

    tiled_shapes = node.meta.get("tiled_shapes")
    scratchpad_mem = node.meta.get("scratchpad_mem")

    # Convert keyword arguments
    for key, value in kwargs.items():
        if key in ["code", "scale_code"] or value is None:
            continue

        if isinstance(value, torch.fx.Node):
            n = value.meta.get("input_node", value)
            n = n.meta.get("source_node", n)
            if tiled_shapes is not None:
                value.meta["shape"] = tiled_shapes.get(n)

            if scratchpad_mem is not None:
                value.meta["scratchpad"] = scratchpad_mem.get(n)

        op_overload.kwargs[key].CopyFrom(convert_arg(value, output_dir))

        if isinstance(value, torch.fx.Node):
            value.meta.pop("shape", None)
            value.meta.pop("scratchpad", None)

    if "l2_tiling" in node.meta:
        op_overload.kwargs["l2_tiling"].int_list.values.extend(node.meta["l2_tiling"])

    return op_overload


def _is_gemm_op(node: Node) -> bool:
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
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
        torch.ops.quantized_ops.vmap.default,
        # Not in the core aten operator set. Will be removed in the future.
        torch.ops.aten.silu.default,
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


def _is_nop(node: Node) -> bool:
    """
    The following operations do not require any computation nor handling
    on the memory placement side. Generate a NOP instruction for these ops
    to keep the compute graph intact.
    """
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
