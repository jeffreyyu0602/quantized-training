import torch
from quantized_training import QuantizationSpec, QuantizationConfig


QUANTIZATION_CONFIG = {
    "q_0": {
        r"self_attn\.[qkvo]_proj$": "nf4_5,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        r"mlp.(?:gate|up|down)_proj$": [
            "nf4_5,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_5,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3"
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "int6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3"
        ],
    }
}

def set_qscheme(quantizer, qscheme):
    for module_name_or_op_type, qspec in qscheme.items():
        if qspec is None:
            qconfig = None
        elif isinstance(qspec, str):
            qspec = QuantizationSpec.from_str(qspec)
            qconfig = QuantizationConfig(qspec, None, qspec, None)
        elif len(qspec) == 2:
            activation = QuantizationSpec.from_str(qspec[0])
            weight = QuantizationSpec.from_str(qspec[1])
            qconfig = QuantizationConfig(activation, None, weight, None)
        elif len(qspec) == 3:
            activation = QuantizationSpec.from_str(qspec[0])
            weight = QuantizationSpec.from_str(qspec[1])
            bias = QuantizationSpec.from_str(qspec[2])
            qconfig = QuantizationConfig(activation, None, weight, bias)
        else:
            raise ValueError(f"Invalid qspec: {qspec}")

        if isinstance(module_name_or_op_type, tuple):
            print("Setting module name and object type for", module_name_or_op_type)
            quantizer.set_module_name_object_type_order(*module_name_or_op_type, qconfig)
        if isinstance(module_name_or_op_type, str):
            print("Setting module name for", module_name_or_op_type)
            quantizer.set_module_name(module_name_or_op_type, qconfig)
        elif isinstance(module_name_or_op_type, torch._ops.OpOverload):
            print("Setting op overload for", module_name_or_op_type)
            quantizer.set_object_type(module_name_or_op_type, qconfig)
        else:
            print("Setting module type for", module_name_or_op_type)
            quantizer.set_module_type(module_name_or_op_type, qconfig)

    return quantizer
