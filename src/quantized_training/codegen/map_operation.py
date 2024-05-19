import operator
import os
import struct

import torch
from torch.fx import Node


aten = torch.ops.aten

# GEMM and layer norm
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
    # native_layer_norm(Tensor input, SymInt[] normalized_shape,
    # Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
    aten.native_layer_norm.default: "layer_norm",
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
    aten.reciprocal.default: "vec_reciprocal",
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
    aten.sum.dim_IntList:  "reduce_sum",
    # max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    aten.max.dim:          "reduce_max",
    # mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    aten.mean.dim:         "reduce_mean",
    # _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    aten._softmax.default: "reduce_softmax",
}

POOLING_OPS_MAPPING = {
    # avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[],
    # int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True,
    # int? divisor_override=None) -> Tensor
    aten.avg_pool2d.default: "pooling_avg_pool2d",
    # max_pool2d_with_indices(Tensor self, int[2] kernel_size,
    # int[2] stride=[], int[2] padding=0, int[2] dilation=1,
    # bool ceil_mode=False) -> (Tensor, Tensor)
    aten.max_pool2d_with_indices.default: "pooling_max_pool2d",
}

SHAPE_OPS_MAPPING = {
    # permute(Tensor(a) self, int[] dims) -> Tensor(a)
    aten.permute.default: "shape_permute",
    # transpose(Tensor(a) a, int[] permutation) -> Tensor(a)
    aten.transpose.int: "shape_transpose",
}

NOP_MAPPING = {
    # TODO: cat operation should be handled by the accelerator
    # cat(Tensor[] tensors, int dim=0) -> Tensor
    aten.cat.default: "nop",
    # clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
    aten.clone.default: "nop",
    # expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    aten.expand.default: "nop",
    aten.t.default: "nop",
    # unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    aten.unsqueeze.default: "nop",
    # view(Tensor(a) self, SymInt[] size) -> Tensor(a)
    aten.view.default: "nop",
    operator.getitem: "nop",
}


def _write_tensor_to_file(tensor, filename):
    tensor = tensor.float().flatten()
    packed_data = struct.pack(f'{tensor.numel()}f', *tensor.tolist())
    with open(filename, 'wb') as f:
        f.write(packed_data)
    print(f"Writing tensor to {filename}")


def _set_repeated_field(obj, attr_path, value, output_dir=None):
    attributes = attr_path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)

    if isinstance(value, Node):
        obj.node = value.name
        obj.shape.extend(list(value.tensor_value.shape))
        if output_dir is not None:
            _write_tensor_to_file(
                value.tensor_value, os.path.join(output_dir, f"{value.name}.bin")
            )
    elif isinstance(value, (tuple, list)):
        obj.extend(value)
    elif value is not None:
        obj.append(value)


def generate_param(node, args, output_dir):
    from .build import param_pb2
    aten = torch.ops.aten
    param = None
    if node.target == aten.convolution.default:
        param = param_pb2.MatrixParam()
        param.name = node.name
        param.opcode = GEMM_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", args[0], output_dir)
        _set_repeated_field(param, "weight", args[1], output_dir)
        _set_repeated_field(param, "bias", args[2], output_dir)
        _set_repeated_field(param, "stride", args[3])
        _set_repeated_field(param, "padding", args[4])
        _set_repeated_field(param, "dilation", args[5])
        param.transposed = args[6]
    elif node.target == aten.addmm.default:
        param = param_pb2.MatrixParam()
        param.name = node.name
        param.opcode = GEMM_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", args[1], output_dir)
        _set_repeated_field(param, "weight", args[2], output_dir)
        _set_repeated_field(param, "bias", args[0], output_dir)
    elif node.target in [aten.mm.default, aten.bmm.default]:
        param = param_pb2.MatrixParam()
        param.name = node.name
        param.opcode = GEMM_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", args[0], output_dir)
        _set_repeated_field(param, "weight", args[1], output_dir)
    elif node.target in VECTOR_OPS_MAPPING:
        param = param_pb2.VectorParam()
        param.name = node.name
        param.opcode = VECTOR_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", args[0])
        if len(args) > 1:
            if isinstance(args[1], Node):
                _set_repeated_field(param, "other", args[1], output_dir)
            else:
                param.scalar = args[1]  # second argument is a scalar
    elif node.target == aten.native_layer_norm.default:
        param = param_pb2.MatrixParam()
        param.name = node.name
        param.opcode = "layer_norm"
        _set_repeated_field(param, "input", args[0], output_dir)
        _set_repeated_field(param, "weight", args[2], output_dir)
        _set_repeated_field(param, "bias", args[3], output_dir)
    elif node.target in REDUCE_OPS_MAPPING:
        param = param_pb2.ReduceParam()
        param.name = node.name
        param.opcode = REDUCE_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", args[0], output_dir)
        _set_repeated_field(param, "dim", args[1])
    elif node.target == aten.avg_pool2d.default:
        default_args = [None, None, [], 0, False, True, None]
        default_args[:len(args)] = args
        param = param_pb2.PoolingParam()
        param.name = node.name
        param.opcode = POOLING_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", default_args[0], output_dir)
        _set_repeated_field(param, "kernel_size", default_args[1])
        _set_repeated_field(param, "stride", default_args[2])
        _set_repeated_field(param, "padding", default_args[3])
        param.ceil_mode = default_args[4]
        param.count_include_pad = default_args[5]
        param.divisor_override = default_args[6]
    elif node.target == aten.max_pool2d_with_indices.default:
        default_args = [None, None, [], 0, 1, False]
        default_args[:len(args)] = args
        param = param_pb2.PoolingParam()
        param.name = node.name
        param.opcode = POOLING_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", default_args[0], output_dir)
        _set_repeated_field(param, "kernel_size", default_args[1])
        _set_repeated_field(param, "stride", default_args[2])
        _set_repeated_field(param, "padding", default_args[3])
        _set_repeated_field(param, "dilation", default_args[4])
        param.ceil_mode = default_args[5]
    elif node.target in SHAPE_OPS_MAPPING:
        param = param_pb2.ShapeParam()
        param.name = node.name
        param.opcode = SHAPE_OPS_MAPPING[node.target]
        _set_repeated_field(param, "input", args[0], output_dir)
        if node.target == aten.permute.default:
            _set_repeated_field(param, "dims", args[1])
        elif node.target == aten.transpose.int:
            _set_repeated_field(param, "dims", args[1:3])
    elif node.target not in NOP_MAPPING:
        print('-' * 40)
        print("Unsupported operation")
        print(node.name)
        print(node.target)
    return param


def map_operation(op, name, output_dir):
    params = [generate_param(*args, output_dir) for args in zip(op.nodes, op.all_input_nodes)]
    params = [param for param in params if param is not None]

    if len(params) == 0:
        return None

    from .build import param_pb2
    param = param_pb2.Param()
    param.name = name
    if params[0].opcode in ["conv", "gemm"]:
        param.matrix_param.CopyFrom(params[0])
        if len(params) > 1:
            param.vector_params.extend(params[1:])
            param.fused = True
        return param

    if params[0].opcode.startswith("vec"):
        param.vector_params.extend(params)
        return param

    assert len(params) == 1, (
        "Reduction, pooling, and shape permutation operation cannot be fused with other operations"
    )

    if params[0].opcode == "layer_norm":
        param.matrix_param.CopyFrom(params[0])
    if params[0].opcode.startswith("reduce"):
        param.reduce_param.CopyFrom(params[0])
    elif params[0].opcode.startswith("shape"):
        param.shape_param.CopyFrom(params[0])
    elif params[0].opcode.startswith("pooling"):
        param.pooling_param.CopyFrom(params[0])
    return param
