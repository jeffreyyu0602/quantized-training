import torch
from google.protobuf.descriptor import FieldDescriptor

from .build.src.quantized_training.accelerator import param_pb2


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
    aten.sum.dim_IntList: "vec_reduce_sum",
    # max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    aten.max.dim:         "vec_reduce_max",
    # mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    aten.mean.dim:        "vec_reduce_mean",
    # _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    aten._softmax.default: "vec_softmax",
}

# Vector operations that can not be fused with GEMM
COMPLEX_VECTOR_OPS_MAPPING = {
    # native_layer_norm(Tensor input, SymInt[] normalized_shape,
    # Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
    aten.native_layer_norm.default: "vec_layer_norm",
    # avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[],
    # int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True,
    # int? divisor_override=None) -> Tensor
    aten.avg_pool2d.default: "vec_avg_pool2d",
    # max_pool2d_with_indices(Tensor self, int[2] kernel_size,
    # int[2] stride=[], int[2] padding=0, int[2] dilation=1,
    # bool ceil_mode=False) -> (Tensor, Tensor)
    aten.max_pool2d_with_indices.default: "vec_max_pool2d",
}

SHAPE_OPS_MAPPING = {
    # view(Tensor(a) self, SymInt[] size) -> Tensor(a)
    aten.view.default: "shape_view",
    # permute(Tensor(a) self, int[] dims) -> Tensor(a)
    aten.permute.default: "shape_permute",
    aten.t.default: "shape_transpose"
}


def _convert_arg(arg):
    if isinstance(arg, torch.Tensor):
        return list(arg.shape)
    elif isinstance(arg, int) and not isinstance(arg, bool):
        return [arg]
    elif isinstance(arg, tuple):
        return tuple(_convert_arg(item) for item in arg)
    else:
        return arg


def generate_param(node, args):
    aten = torch.ops.aten
    args = _convert_arg(args)
    if node.target == aten.convolution.default:
        param = param_pb2.MatrixParam()
        param.opcode = GEMM_OPS_MAPPING[node.target]
        param.input.extend(args[0])
        param.weight.extend(args[1])
        param.bias.extend(args[2])
        param.stride.extend(args[3])
        param.padding.extend(args[4])
        param.dilation.extend(args[5])
        param.transposed = args[6]
        # param.output_padding.extend(args[7])
        # param.groups = args[8]
    elif node.target == aten.addmm.default:
        param = param_pb2.MatrixParam()
        param.opcode = GEMM_OPS_MAPPING[node.target]
        param.input.extend(args[1])
        param.weight.extend(args[2])
        param.bias.extend(args[0])
    elif node.target in [aten.mm.default, aten.bmm.default]:
        param = param_pb2.MatrixParam()
        param.opcode = GEMM_OPS_MAPPING[node.target]
        param.input.extend(args[0])
        param.weight.extend(args[1])
    elif node.target in VECTOR_OPS_MAPPING:
        param = param_pb2.VectorParam()
        param.opcode = VECTOR_OPS_MAPPING[node.target]
        param.input.extend(args[0])
        if len(args) > 1:
            param.other.extend(args[1])
    elif node.target in REDUCE_OPS_MAPPING:
        param = param_pb2.ReductionParam()
        param.opcode = REDUCE_OPS_MAPPING[node.target]
        param.input.extend(args[0])
        param.dim.extend(args[1])
    elif node.target == aten.native_layer_norm.default:
        param = param_pb2.MatrixParam()
        param.opcode = COMPLEX_VECTOR_OPS_MAPPING[node.target]
        param.input.extend(args[0])
        param.weight.extend(args[1])
        param.bias.extend(args[2])
        # param.normalized_shape = args[3]
        # param.eps = args[4]
    elif node.target == aten.avg_pool2d.default:
        default_args = [None, None, [], [0], False, True, None]
        default_args[:len(args)] = args
        param = param_pb2.MatrixParam()
        param.opcode = COMPLEX_VECTOR_OPS_MAPPING[node.target]
        param.input.extend(default_args[0])
        param.weight.extend(default_args[1])
        param.stride.extend(default_args[2])
        param.padding.extend(default_args[3])
        param.ceil_mode = default_args[4]
        # param.count_include_pad = default_args[5]
        # param.divisor_override = default_args[6]
    elif node.target == aten.max_pool2d_with_indices.default:
        default_args = [None, None, [], [0], [1], False]
        default_args[:len(args)] = args
        param = param_pb2.MatrixParam()
        param.opcode = COMPLEX_VECTOR_OPS_MAPPING[node.target]
        param.input.extend(default_args[0])
        param.weight.extend(default_args[1])
        param.stride.extend(default_args[2])
        param.padding.extend(default_args[3])
        param.dilation.extend(default_args[4])
        param.ceil_mode = default_args[5]
    else:
        print("Unsupported operation")
        print(node.name)
        print(node.target)
        param = None
    return param


def map_operation(op):
    params = [generate_param(*args) for args in zip(op.nodes, op.args)]
    params = [param for param in params if param is not None]
    if len(params) == 0:
        return None
    vec_params = [param for param in params if isinstance(param, param_pb2.VectorParam)]
    reduce_params = [param for param in params if isinstance(param, param_pb2.ReductionParam)]
    param = params[0]
    param.vector_ops.extend(vec_params)
    return param


def format_protobuf_message(message):
    output = []
    for field, value in message.ListFields():
        if field.label == FieldDescriptor.LABEL_REPEATED:
            formatted_value = list(value)  # Convert repeated fields to a list
        else:
            formatted_value = value
        
        # Handle nested messages
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            if field.label == FieldDescriptor.LABEL_REPEATED:
                formatted_value = [format_protobuf_message(v) for v in value]
            else:
                formatted_value = format_protobuf_message(value)
        
        output.append(f"{field.name}: {formatted_value}")
    
    return "{" + ", ".join(output) + "}"
