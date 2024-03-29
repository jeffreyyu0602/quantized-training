import logging
import re

import torch
from torch.ao.quantization import FakeQuantizeBase

from .fp8 import quantize_to_fp8_e4m3, quantize_to_fp8_e5m2
from .posit import quantize_to_posit

__all__ = [
    "FusedAmaxObsFakeQuantize",
]

logger = logging.getLogger(__name__)

def _round_to_mbits(input, mbits):
    lb_mask = 1 << (23 - mbits)
    gb_mask = lb_mask >> 1
    sb_mask = gb_mask - 1

    raw_bits = input.float().view(torch.int32)
    lb = (raw_bits & lb_mask) != 0
    gb = (raw_bits & gb_mask) != 0
    sb = (raw_bits & sb_mask) != 0
    rb = (lb & gb) | (gb & sb)

    raw_bits[rb] += lb_mask
    raw_bits &= ~(lb_mask - 1)
    return raw_bits.view(torch.float32).to(input.dtype)

def dtype_to_fq_fn(dtype: str):
    """Return the quantization function for the given dtype."""
    if (match := re.fullmatch(r'posit(\d+)_(\d+)', dtype)):
        nbits, es = match.groups()
        return lambda x: quantize_to_posit(x, int(nbits), int(es), round_to_even=True)
    elif (match := re.fullmatch(r'(?:FP8\.)?(E4M3|E5M2)', dtype, re.IGNORECASE)):
        fp8_format = match.group(1).lower()
        return quantize_to_fp8_e4m3 if fp8_format == 'e4m3' else quantize_to_fp8_e5m2
    elif (match := re.fullmatch(r'clamp_max_(\d+)', dtype)):
        max_value = float(match.group(1))
        return lambda x: torch.clamp(x, -max_value, max_value)
    elif (match := re.fullmatch(r'clamp_min_exp_(-?\d+)', dtype)):
        min_exp = int(match.group(1))
        return lambda x: torch.where(torch.log2(torch.abs(x)) < min_exp, 0, x)
    elif (match := re.fullmatch(r'E8M(\d+)', dtype, re.IGNORECASE)):
        mbits = int(match.group(1))
        assert mbits >= 0 and mbits <= 23, "mbits must be between 0 and 23"
        return lambda x: _round_to_mbits(x, mbits)
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")

class FakeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, values, scale=None):
        if scale is not None:
            input = input * scale

        if input.dtype == torch.bfloat16:
            indices = input.view(torch.int16).int() & 0xffff
        else:
            raw_bits = input.float().view(torch.int32)
            indices = ((raw_bits >> 16) & 0xffff) | ((raw_bits & 0xffff) != 0).int()
        input = values[indices].to(input.dtype)

        return input / scale if scale is not None else input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class FusedAmaxObsFakeQuantize(FakeQuantizeBase):
    r"""Observer module for computing the quantization parameters based on the
    historical amax values.

    This observer uses the tensor amax statistics to compute the quantization
    parameters. The module records the maximums of absolute values of output
    tensors, and uses this statistic to compute the quantization parameters.
    """
    value_map: torch.Tensor
    amax_history: torch.Tensor
    scale: torch.Tensor
    histogram: torch.Tensor

    def __init__(
        self,
        dtype: str,
        is_per_tensor: bool = False,
        quant_max: float = None,
        amax_history_len: int = 50,
        observer_enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.is_per_tensor = is_per_tensor
        self.quant_max = quant_max
        self.amax_history_len = amax_history_len
        self.name = kwargs.get("name", None)
        device = kwargs.get("device", None)
        factory_kwargs = {'device': device, 'dtype': torch.float}
        self.register_buffer("amax_history", torch.zeros(amax_history_len, **factory_kwargs))
        self.register_buffer('scale', torch.tensor([1.0], **factory_kwargs))
        # Generate all possible bfloat16 values and quantize them to the given dtype.
        input = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)
        self.fake_quant_fn = dtype_to_fq_fn(dtype)
        self.register_buffer("value_map", self.fake_quant_fn(input), persistent=False)
        # Records the histogram of the input tensor
        self.enable_observer(observer_enabled)
        self.register_buffer("histogram", torch.zeros(254, **factory_kwargs))

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, dtype={}, is_per_tensor={}, "
            "quant_max={}, amax_history_len={}, scale={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.dtype,
                self.is_per_tensor,
                self.quant_max,
                self.amax_history_len,
                self.scale,
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_per_tensor:
            # Remove attention mask when computing scaling factor
            x_detached = X.detach()
            min_val = torch.finfo(X.dtype).min
            attention_mask = torch.where(x_detached == min_val, min_val, 0.0)
            X -= attention_mask

            amax = torch.amax(self.amax_history)
            self.amax_history = torch.roll(self.amax_history, -1, 0)
            self.amax_history[0] = torch.amax(torch.abs(X.detach())).float()

            sf = self.quant_max / amax
            sf = torch.pow(2, torch.floor(torch.log2(sf)))
            sf = torch.where(amax > 0.0, sf, self.scale)
            sf = torch.where(torch.isfinite(amax), sf, self.scale)
            self.scale = sf

        if self.observer_enabled[0] == 1:
            self._combine_histograms(self.histogram, X)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_tensor:
                X = FakeQuantFunction.apply(X, self.value_map, self.scale.to(X.dtype))
                X += attention_mask
            else:
                X = FakeQuantFunction.apply(X, self.value_map)

        return X

    def _combine_histograms(self, histogram: torch.Tensor, X: torch.Tensor) -> None:
        x_detached = X.detach().float()
        exponent = torch.floor(torch.log2(x_detached[x_detached != 0].abs()))
        combined_histogram = torch.histc(exponent, 254, min=-126, max=127)
        histogram += combined_histogram