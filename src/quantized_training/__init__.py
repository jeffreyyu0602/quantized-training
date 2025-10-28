from .codegen import *
from .decomposed import *
from .fake_quantize import *
from .fp8 import *
from .llm_utils import *
from .normal_float import *
from .posit import *
from .pt2e_utils import *
from .qconfig import *
from .quantize import *
from .quantize_pt2e import *
from .quantizer import *
from .training_args import *
from .utils import *
from .histogram import *
from .quantize_pt2e import fuse_quantize_dequantize_with_previous_op
from .codegen.mapping import rename_nodes_with_param_names
from google.protobuf import text_format
import operator
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten


__all__ = [
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QuantizationSpec",
    "TorchExportableModuleWithStaticCache",
    "add_qspec_args",
    "convert_and_export_with_split_cache",
    "convert",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "dispatch_model",
    "dtype_byte_size",
    "export_model",
    "generate",
    "get_aten_graph_module",
    "get_device_map",
    "get_node_name_to_scope",
    "get_qconfig",
    "get_quantized_model",
    "get_default_quantizer",
    "insert_align_device_nodes",
    "plot_histogram",
    "plot_layer_range",
    "prepare",
    "prepare_pt2e",
    "print_node_scope_tabular",
    "propagate_config",
    "quantize",
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "quantize_to_nf",
    "quantize_to_posit",
    "replace_softmax",
    "setup_logging",
    "sink_obs_or_fq",
    "swap_llama_attention",
]

class qscheme: ...

# Defined in quantized_training/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING
group_wise_affine: qscheme = QScheme.GROUP_WISE_AFFINE

aten = torch.ops.aten
quantized_ops = torch.ops.quantized_ops

OPERATOR_MAPPINGS = {
    "gemm": [nn.Conv2d, nn.Linear, F.conv2d, F.linear, torch.matmul, operator.matmul],
    "add": ["add", "add_", operator.add, torch.add, operator.iadd, aten.add.Tensor],
    "sub": ["sub", "sub_", operator.sub, torch.sub, operator.isub],
    "mul": ["mul", "mul_", operator.mul, torch.mul, operator.imul],
    "div": ["div", "div_", operator.truediv, torch.div, operator.itruediv],
    "relu": [nn.ReLU, F.relu, F.relu_],
    "gelu": [nn.GELU, F.gelu],
    "silu": [nn.SiLU, F.silu],
    "hardtanh" : [nn.ReLU6, F.relu6],
    "maxpool2d": [nn.MaxPool2d, F.max_pool2d],
    "avgpool2d": [nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d],
    "layer_norm": ["layer_norm", nn.LayerNorm, F.layer_norm],
    "softmax": ["softmax", nn.Softmax, F.softmax],
    "quantize": [quantized_ops.quantize.default, quantized_ops.quantize_mx.default],
    "dequantize": [quantized_ops.dequantize.default],
}

def fuse(model, patterns, example_args, example_kwargs=None, fuse_reshape=True):
    if example_kwargs is None:
        example_kwargs = {}

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*flatten_args)

    vector_stages = []
    for pattern in patterns:
        # If there is no corresponding mapping, we directly append the op itself
        vector_stages.append([
            [item for op in ops for item in OPERATOR_MAPPINGS.get(op, [op])]
            for ops in pattern
        ])

    fuse_operator(model, vector_stages, fuse_reshape)
    return model


def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    patterns=None,
    fuse_operator=True,
    transpose_weight=False,
    transpose_fc=False,
    cache_size=None,
    num_banks=None,
    unroll_dims=None,
    conv2d_im2col=False,
    fuse_reshape=True,
):
    if example_kwargs is None:
        example_kwargs = {}

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*flatten_args)

    # Turn batched matmul into multiple matmuls
    split_multi_head_attention(model)

    # TODO Turned off for large model now. This will be removed in the future
    # once we can handle them on the C compiler side.
    if len(model.graph.nodes) < 10000:
        # Convert torch.expand to memory copy
        convert_expand_to_memory_copy(model)

        # Perform transformations to the model
        convert_cat_and_stack_as_stack_on_dim0(model)
        convert_cat_with_mismatched_shapes_to_stack(model)

    # Move quantize and dequantize ops to the end of last compute op
    fuse_quantize_dequantize_with_previous_op(model)

    if conv2d_im2col:
        replace_conv2d_with_im2col(model)

    if unroll_dims is not None:
        pad_gemm_inputs_to_hardware_unroll_size(model, *unroll_dims)
        pad_vector_ops_to_hardware_unroll_size(model, unroll_dims[1])

    if cache_size is not None:
        run_matrix_op_l2_tiling(model, unroll_dims, cache_size, num_banks)
        run_vector_op_l2_tiling(model, unroll_dims, cache_size, num_banks)

    transpose_linear_weights(model, transpose_weight, transpose_fc)
    if transpose_weight:
        transpose_conv2d_inputs_and_weights(model)
    ShapeProp(model).propagate(*flatten_args)
    eliminate_reshape_with_no_effect(model)

    if fuse_operator:
        fuse(model, patterns, flatten_args, fuse_reshape=fuse_reshape)

    rename_nodes_with_param_names(model)


def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    total_memory=None,
    cache_size=None,
    num_banks=None,
    bank_width=None,
    unroll_dims=None,
    output_dir=None,
    output_file="compute_graph",
    dump_snapshot=False,
    dump_verification_file=True,
):
    flatten_args, spec = tree_flatten((example_args, example_kwargs))

    ShapeProp(model).propagate(*flatten_args)

    allocator = MemoryAllocator(total_memory)
    run_memory_mapping(
        model, allocator, cache_size, num_banks, bank_width, unroll_dims
    )

    os.makedirs(output_dir, exist_ok=True)

    if dump_snapshot:
        allocator.dump_snapshots(os.path.join(output_dir, "memory_snapshots.png"))

    path = (
        os.path.join(output_dir, "tensor_files")
        if dump_verification_file else None
    )
    params = gen_code(model, flatten_args, path)

    with open(os.path.join(output_dir, 'model.txt'), "w") as f:
        f.write(text_format.MessageToString(params))

    operations = [
        op.op.name if op.WhichOneof('op_type') == 'op' else op.fused_op.name
        for op in params.ops if op.op.op != 'nop'
    ]

    with open(os.path.join(output_dir, 'layers.txt'), 'w') as f:
        f.write('\n'.join(operations))

    if len(model.graph.nodes) < 10000:
        gen_compute_graph(model, os.path.join(output_dir, output_file))
