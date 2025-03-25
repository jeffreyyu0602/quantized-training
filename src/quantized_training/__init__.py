from .codegen import *
from .decomposed import *
from .fake_quantize import *
from .fp8 import *
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
from google.protobuf import text_format
import operator


__all__ = [
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QuantizationSpec",
    "add_qspec_args",
    "convert",
    "derive_bias_qparams_fn",
    "dispatch_model",
    "dtype_byte_size",
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
]

class qscheme: ...

# Defined in quantized_training/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING

OPERATOR_MAPPINGS = {
    "gemm": [torch.nn.Conv2d, torch.nn.Linear, torch.matmul, operator.matmul],
    "add": ["add", "add_", operator.add, torch.add, operator.iadd],
    "sub": ["sub", "sub_", operator.sub, torch.sub, operator.isub],
    "mul": ["mul", "mul_", operator.mul, torch.mul, operator.imul],
    "div": ["div", "div_", operator.truediv, torch.div, operator.itruediv],
    "relu": [torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_],
    "gelu": [torch.nn.GELU, torch.nn.functional.gelu],
    "silu": [torch.nn.SiLU, torch.nn.functional.silu],
    "maxpool2d": [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d],
    "avgpool2d": [torch.nn.AdaptiveAvgPool2d, torch.nn.functional.adaptive_avg_pool2d],
    "quantize": [torch.ops.quantized_ops.quantize.default, torch.ops.quantized_ops.quantize_mx.default],
    "dequantize": [torch.ops.quantized_ops.dequantize.default],
}


def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    patterns=None,
):
    if example_kwargs is None:
        example_kwargs = {}

    # Run the model once to get a reference output
    orig_out = model(*example_args, **example_kwargs)

    from torch.utils._pytree import tree_flatten
    flatten_args, spec = tree_flatten((example_args, example_kwargs))

    # Turn batched matmul into multiple matmuls
    ShapeProp(model).propagate(*flatten_args)
    split_multi_head_attention(model)

    # Convert torch.expand in Grouped Query Attention to memory copy
    convert_expand_to_memory_copy(model)

    # Perform transformations to the model
    ShapeProp(model).propagate(*flatten_args)
    convert_cat_and_stack_as_stack_on_dim0(model)
    convert_cat_with_mismatched_shapes_to_stack(model)

    # Move quantize and dequantize ops to the end of last compute op
    ShapeProp(model).propagate(*flatten_args)
    fuse_quantize_dequantize_with_previous_op(model)

    for pattern in patterns:
        # If there is no corresponding mapping, we directly append the op itself
        vector_stages = [
            [item for op in ops for item in OPERATOR_MAPPINGS.get(op, [op])]
            for ops in pattern
        ]

        ShapeProp(model).propagate(*flatten_args)
        fuse_operator(model, vector_stages)

    model.graph.print_tabular()

    new_out = model(*example_args, *list(example_kwargs.values()))
    return orig_out, new_out


def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    total_memory=None,
    bank_width=None,
    output_dir=None,
    output_file="compute_graph",
):
    from torch.utils._pytree import tree_flatten
    flatten_args, spec = tree_flatten((example_args, example_kwargs))

    ShapeProp(model).propagate(*flatten_args)

    manager = MemoryManager(total_memory, bank_width=bank_width)
    allocate_weights(model, manager)
    allocate_activations(model, manager)

    model_params = gen_code(model, flatten_args, os.path.join(output_dir, "tensor_files"))

    with open(os.path.join(output_dir, 'model.txt'), "w") as f:
        f.write(text_format.MessageToString(model_params))

    with open(os.path.join(output_dir, 'layers.txt'), 'w') as f:
        for op in model_params.ops:
            if op.WhichOneof('op_type') == 'fused_op':
                f.write(op.fused_op.name + '\n')
            elif op.op.op != 'nop':
                f.write(op.op.name + '\n')

    gen_compute_graph(model, os.path.join(output_dir, output_file))
