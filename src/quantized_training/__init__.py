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
from .codegen.mapping import replace_elementwise_with_vmap
from .quantize_pt2e import _fuse_quantize_dequantize_with_previous_op
from google.protobuf import text_format
import operator


__all__ = [
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QuantizationSpec",
    "add_qspec_args",
    "convert",
    "dispatch_model",
    "dtype_byte_size",
    "get_aten_graph_module",
    "get_device_map",
    "get_node_name_to_scope",
    "get_qconfig",
    "get_quantized_model",
    "get_default_quantizer",
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
    "maxpool2d": [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d],
    "avgpool2d": [torch.nn.AdaptiveAvgPool2d, torch.nn.functional.adaptive_avg_pool2d],
    "quantize": [torch.ops.quantized_ops.quantize],
    "dequantize": [torch.ops.quantized_ops.dequantize],
}

def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    *,
    output_file="compute_graph",
    output_dir=None
):
    if example_kwargs is None:
        example_kwargs = {}

    orig_model = copy.deepcopy(model)

    from torch.utils._pytree import tree_flatten
    flatten_args, spec = tree_flatten((example_args, example_kwargs))

    ShapeProp(model).propagate(*flatten_args)
    split_multi_head_attention(model)

    # There are additional dequantize ops produced by head splitting. Fuse them
    # with the previous op.
    ShapeProp(model).propagate(*flatten_args)
    _fuse_quantize_dequantize_with_previous_op(model)

    vector_stages = {
        0: ["gemm"],
        1: ["dequantize"],
        2: ["add", "sub", "mul", "div"],
        3: ["exp"],
        4: ["add", "mul", "div"],
        5: ["relu"],
        7: ["quantize"],
    }

    # If there is no corresponding mapping, we directly append the op string
    vector_stages = {
        stage: [item for op in ops for item in OPERATOR_MAPPINGS.get(op, [op])]
        for stage, ops in vector_stages.items()
    }

    replace_elementwise_with_vmap(model, vector_stages)

    ShapeProp(model).propagate(*flatten_args)
    fuse_operator(model, vector_stages)

    model.graph.print_tabular()

    ShapeProp(model).propagate(*flatten_args)

    try:
        manager = MemoryManager(1024 ** 4)
        allocate_weights(model, manager)
        allocate_activations(model, manager)
    except RuntimeError:
        manager.print_partitions()
        filename = os.path.join(output_dir, "memory_layout.png")
        visualize_memory_layout(manager.snapshots, filename)
        logger.error(f"Memory allocation failed. Memory layout saved to {filename}")

    manager.print_partitions()
    print("\nMemory allocated to tensors:")
    for node in model.graph.nodes:
        if (partition := node.meta.get('memory', None)) is None:
            print(f"Node {node.name} does not have memory allocated")
            continue
        print(f"{node.name}: {partition.start}, {partition.end}")

    params = gen_code(model, flatten_args, os.path.join(output_dir, "tensor_files"))
    with open(os.path.join(output_dir, 'params.txt'), "w") as f:
        f.write(text_format.MessageToString(params))

    layers = [p.name for p in params.params if p.WhichOneof("param") != "nop_param"]
    with open(os.path.join(output_dir, 'layers.txt'), 'w') as f:
        f.write('\n'.join(layers))

    gen_compute_graph(model, os.path.join(output_dir, output_file))

    orig_out = orig_model(*example_args, **example_kwargs)
    new_out = model(*example_args, *list(example_kwargs.values()))
    return orig_out, new_out
