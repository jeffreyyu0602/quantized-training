import torch
import torch.nn.functional as F
from torch.nn.utils.parametrize import type_before_parametrizations

from peft.tuners import lora
from peft.utils.other import transpose

__all__ = [
    "Linear"
]

class Linear(lora.Linear):
    _FLOAT_MODULE = lora.Linear

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        qconfig=None,
        **kwargs,
    ):
        super().__init__(adapter_name, in_features, out_features, r, lora_alpha,
                         lora_dropout, fan_in_fan_out, is_target_conv_1d_layer, **kwargs)
        assert qconfig, 'quantizer must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(**kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            orig_weights = self.weight.data.clone()
            for active_adapter in self.active_adapters:
                if active_adapter in self.lora_A.keys():
                    weight_A = self.weight_fake_quant(self.lora_A[active_adapter].weight)
                    weight_B = self.weight_fake_quant(self.lora_B[active_adapter].weight)
                    scaling = self.scaling[active_adapter]
                    orig_weights += transpose(weight_B @ weight_A, self.fan_in_fan_out) * scaling
            orig_weights = self.weight_fake_quant(orig_weights)
            result = F.linear(x, transpose(orig_weights, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result

    @classmethod
    def from_float(cls, mod):
        assert type_before_parametrizations(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        if mod.merged:
            mod.unmerge()

        adaptor = mod.active_adapter[0]
        qconfig = mod.qconfig
        qat_linear = cls(adaptor, mod.in_features, mod.out_features, bias=mod.bias is not None,
                         r=mod.r[adaptor], lora_alpha=mod.lora_alpha[adaptor],
                         fan_in_fan_out=mod.fan_in_fan_out, init_lora_weights=False, qconfig=qconfig)
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        qat_linear.r = mod.r
        qat_linear.lora_alpha = mod.lora_alpha
        qat_linear.scaling = mod.scaling
        qat_linear.lora_dropout = mod.lora_dropout
        qat_linear.lora_A = mod.lora_A
        qat_linear.lora_B = mod.lora_B
        return qat_linear

    @classmethod
    def to_float():
        raise NotImplementedError
