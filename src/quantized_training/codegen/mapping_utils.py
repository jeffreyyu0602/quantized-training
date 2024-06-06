import operator
import os
import struct
from typing import Callable, Dict

import torch
from torch.ao.quantization import FakeQuantizeBase
from torch.fx import Node


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


def _get_obs_or_fq(gm, node):
    named_modules = dict(gm.named_modules(remove_duplicate=False))
    if node.op == "call_module" and str(node.target) in named_modules:
        obs_or_fq = named_modules[str(node.target)]
        if isinstance(obs_or_fq, FakeQuantizeBase):
            return obs_or_fq
    return None


def _set_tensor_field(field, node, output_dir):
    assert isinstance(node, Node), "tensor field must be a Node"
    field.node = node.name
    field.dtype = str(node.dtype).split(".")[1]
    field.shape.extend(list(node.shape))

    gm = node.graph.owning_module
    named_modules = dict(gm.named_modules(remove_duplicate=False))
    scale = None
    if node.op == "call_module" and str(node.target) in named_modules:
        obs_or_fq = named_modules[str(node.target)]
        if isinstance(obs_or_fq, FakeQuantizeBase):
            field.dtype = obs_or_fq.dtype
            scale = obs_or_fq.scale

    if output_dir is not None:
        _write_tensor_to_file(
            node.value, os.path.join(output_dir, f"{node.name}.bin")
        )

    return scale


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
    if node.op != "call_function" or node.target != torch.ops.aten.conv2d.default:
        return None
    default_args = [None, None, None, 1, 0, 1, 1]
    default_args[:len(node.args)] = node.args

    from .param_pb2 import MatrixParam, VectorParam
    param = MatrixParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    scale_act = _set_tensor_field(param.input, default_args[0], output_dir)
    scale_wt = _set_tensor_field(param.weight, default_args[1], output_dir)
    _set_tensor_field(param.bias, default_args[2], output_dir)
    _set_repeated_field(param.stride, default_args[3])
    _set_repeated_field(param.padding, default_args[4])
    _set_repeated_field(param.dilation, default_args[5])
    # param.groups = default_args[6]

    scale = scale_act if scale_act is not None else torch.tensor(1.0)
    scale = scale * (scale_wt if scale_wt is not None else torch.tensor(1.0))
    gm = node.graph.owning_module
    obs_or_fq = _get_obs_or_fq(gm, next(iter(node.users)))
    if obs_or_fq is not None:
        scale = scale / obs_or_fq.scale

    quantization_param = VectorParam()
    quantization_param.name = node.name + "_scale"
    quantization_param.opcode = "quantize" if obs_or_fq is not None else "dequantize"
    _set_tensor_field(quantization_param.input, node, output_dir)
    quantization_param.other.node = node.name + "_scale"
    quantization_param.other.dtype = "bfloat16"
    quantization_param.other.shape.extend(list(scale.shape))
    # _write_tensor_to_file(scale, os.path.join(output_dir, f"{node.name}_scale.bin"))
    # import json
    # from google.protobuf.json_format import MessageToDict
    # print(json.dumps(MessageToDict(quantization_param), indent=4))
    return param


@register_annotator("linear")
def map_linear(node, output_dir):
    """
    Schema: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
        return None
    from .param_pb2 import MatrixParam
    param = MatrixParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_tensor_field(param.weight, node.args[1], output_dir)
    _set_tensor_field(param.bias, node.args[2], output_dir)
    return param


@register_annotator("matmul")
def map_matmul(node, output_dir):
    """
    Schema: matmul(Tensor self, Tensor other) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.matmul.default:
        return None
    from .param_pb2 import MatrixParam
    param = MatrixParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_tensor_field(param.weight, node.args[1], output_dir)
    return param


@register_annotator("layer_norm")
def map_layer_norm(node, output_dir):
    """
    Schema: layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.layer_norm.default:
        return None
    from .param_pb2 import MatrixParam
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
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.exp.default,
        torch.ops.aten.log.default,
        torch.ops.aten.reciprocal.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.gelu_.default
    ]


@register_annotator("elwise")
def map_elwise(node, output_dir):
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
    gelu(Tensor self, *, str approximate=’none’) -> Tensor
    """
    if node.op != "call_function" or not _is_elementwise_op(node.target):
        return None
    from .param_pb2 import VectorParam
    param = VectorParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    if len(node.args) > 1:
        _set_tensor_field(param.other, node.args[1], output_dir)
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
    from .param_pb2 import ReduceParam
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
    default_args = [None, None, [], 0, False, True, None]
    default_args[:len(node.args)] = node.args
    from .param_pb2 import PoolingParam
    param = PoolingParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, default_args[0], output_dir)
    _set_repeated_field(param.kernel_size, default_args[1])
    _set_repeated_field(param.stride, default_args[2])
    _set_repeated_field(param.padding, default_args[3])
    param.ceil_mode = default_args[4]
    param.count_include_pad = default_args[5]
    param.divisor_override = default_args[6]
    return param


@register_annotator("adaptive_avg_pool2d")
def map_adaptive_avg_pool2d(node, output_dir):
    """
    Schema: adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
    """
    if node.op != "call_function" or node.target != torch.ops.aten.adaptive_avg_pool2d.default:
        return None
    from .param_pb2 import PoolingParam
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
    default_args = [None, None, [], 0, 1, False]
    default_args[:len(node.args)] = node.args
    from .param_pb2 import PoolingParam
    param = PoolingParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, default_args[0], output_dir)
    _set_repeated_field(param.kernel_size, default_args[1])
    _set_repeated_field(param.stride, default_args[2])
    _set_repeated_field(param.padding, default_args[3])
    _set_repeated_field(param.dilation, default_args[4])
    param.ceil_mode = default_args[5]
    return param


@register_annotator("permute")
def map_permute(node, output_dir):
    """
    Schema: permute(Tensor(a) self, int[] dims) -> Tensor(a)
    """
    if node.op != "call_function" or node.target != torch.ops.aten.permute.default:
        return None
    from .param_pb2 import ShapeParam
    param = ShapeParam()
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
    from .param_pb2 import ShapeParam
    param = ShapeParam()
    param.name = node.name
    param.opcode = node.target.__name__.split(".")[0]
    _set_tensor_field(param.input, node.args[0], output_dir)
    _set_repeated_field(param.dims, node.args[1:])
    return param


def _is_nop(op: Callable) -> bool:
    return op in [
        torch.ops.aten.cat.default,
        torch.ops.aten.clone.default,
        # TODO: remove?
        torch.ops.aten.dropout.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.t.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        operator.getitem,
    ]
