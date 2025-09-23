import logging
import torch
from quantized_training import QuantizationSpec, QuantizationConfig


logger = logging.getLogger(__name__)


QUANTIZATION_CONFIGS = {
    "linear4": {
        torch.nn.Linear: [
            "nf4,qs=microscaling,bs=64,ax=-1",
            "nf4,qs=microscaling,bs=64,ax=-1",
        ],
    },
    "matmul4": {
        torch.ops.aten.matmul.default: [
            "nf4,qs=microscaling,bs=64,ax=-1",
            "nf4,qs=microscaling,bs=64,ax=-2",
        ],
    },
    "linear4_matmul6": {
        torch.nn.Linear: [
            "nf4,qs=microscaling,bs=64,ax=-1",
            "nf4,qs=microscaling,bs=64,ax=-1",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1",
            "int6,qs=microscaling,bs=64,ax=-2",
        ],
    },
    "linear4_matmul6_fp8": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "int6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3",
        ],
    },
    "linear4_matmul6_fp8_mixhead": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "int6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3",
        ],
        (r"lm_head", torch.ops.aten.linear.default, 0): [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
    },
    "linear4_matmul6_fp8_outlier": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3,outlier=4.0",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "int6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3",
        ],
        (r"lm_head", torch.ops.aten.linear.default, 0): [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
    },
}

def set_qconfig(quantizer, qconfigs):
    for key, qspec in qconfigs.items():
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

        if isinstance(key, tuple):
            logger.info(f"Setting qconfig for module name, object type and order: {key}")
            quantizer.set_module_name_object_type_order(*key, qconfig)
        elif isinstance(key, str):
            logger.info(f"Setting qconfig for module name: {key}")
            quantizer.set_module_name(key, qconfig)
        elif isinstance(key, type) and issubclass(key, torch.nn.Module):
            logger.info(f"Setting qconfig for module type: {key}")
            quantizer.set_module_type(key, qconfig)
        elif isinstance(key, torch._ops.OpOverload):
            logger.info(f"Setting qconfig for op overload: {key}")
            quantizer.set_object_type(key, qconfig)
        else:
            raise ValueError(f"Invalid module name or type: {key}")

    return quantizer
