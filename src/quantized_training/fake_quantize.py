import logging
import re
from typing import Any, Union, Callable, Literal, Tuple

import torch
import matplotlib.pyplot as plt
from torch.ao.quantization import FakeQuantizeBase

from .fp8 import quantize_to_fp8_e4m3, quantize_to_fp8_e5m2
from .posit import quantize_to_posit

__all__ = [
    "FusedAmaxObsFakeQuantize",
    "quantization_dtype_to_function",
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

def quantization_dtype_to_function(dtype: str) -> Callable:
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

def _convert(input: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Index value tensor using the bit pattern of input tensor."""
    if input.dtype == torch.bfloat16:
        indices = input.view(torch.int16).int() & 0xffff
    else:
        raw_bits = input.float().view(torch.int32)
        indices = ((raw_bits >> 16) & 0xffff) | ((raw_bits & 0xffff) != 0).int()
    return values[indices].to(input.dtype)

class FakeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, values, scale=None):
        if scale is None:
            return _convert(input, values)
        else:
            return _convert(input * scale, values) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def _update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        amax_history = torch.roll(amax_history, -1, 0)
    amax_history[0].fill_(0.0)
    return amax_history

def _default_get_amax(
    amax_history: torch.Tensor,
    amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max":
        amax = torch.max(amax_history, dim=0).values
    else:  # amax_compute_algo == "most_recent"
        amax = amax_history[0].clone()

    amax_history = _update_amax_history(amax_history)
    return amax_history, amax

def _default_sf_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    quant_max: float,
    margin: int,
) -> torch.Tensor:
    """Default function to convert amax to scaling factor."""
    exp = torch.floor(torch.log2(quant_max / amax)) - margin
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(exp < 0, 1 / sf, sf)

    return sf

class FusedAmaxObsFakeQuantize(FakeQuantizeBase):
    r"""Observer module for computing the quantization parameters based on the
    historical amax values.

    This observer uses the tensor amax statistics to compute the quantization
    parameters. The module records the maximums of absolute values of output
    tensors, and uses this statistic to compute the quantization parameters.
    """
    amax_history: torch.Tensor
    scale: torch.Tensor
    value_map: torch.Tensor
    histogram_pre_process: torch.Tensor
    histogram_post_process: torch.Tensor

    def __init__(
        self,
        dtype: str,
        quant_max: float = None,
        amax_history_len: int = 50,
        amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = "max",
        quantize_per_tensor: bool = False,
        histogram_observer_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.quant_max = quant_max
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.quantize_per_tensor = quantize_per_tensor
        if not quantize_per_tensor:
            self.disable_observer()
        self.layer_name = kwargs.get("layer_name", None)
        device = kwargs.get("device", None)
        factory_kwargs = {'device': device, 'dtype': torch.float}
        self.register_buffer("amax_history", torch.zeros(amax_history_len, **factory_kwargs))
        self.register_buffer('scale', torch.tensor([1.0], **factory_kwargs))
        # Generate all possible bfloat16 values and quantize them to the given dtype.
        try:
            input = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)
        except RuntimeError:
            logger.warn("Bfloat16 is not supported on this device.")
            input = (torch.arange(2 ** 16, dtype=torch.int32, device=device) << 16).view(torch.float)
        self.fake_quant_function = quantization_dtype_to_function(dtype)
        self.register_buffer("value_map", self.fake_quant_function(input), persistent=False)
        # Records the histogram of the input and quantized input values
        self.register_buffer('histogram_observer_enabled', torch.tensor([histogram_observer_enabled], dtype=torch.uint8))
        self.register_buffer("histogram_pre_process", torch.zeros(254, **factory_kwargs))
        self.register_buffer("histogram_post_process", torch.zeros(254, **factory_kwargs))

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, scale={}, quantize_per_tensor={}, "
            "dtype={}, quant_max={}, amax_history_len={}, amax_compute_algo={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.scale,
                self.quantize_per_tensor,
                self.dtype,
                self.quant_max,
                self.amax_history_len,
                self.amax_compute_algo,
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled[0] == 1:
            self.amax_history, amax = _default_get_amax(
                self.amax_history,
                self.amax_compute_algo,
            )
            self.scale = _default_sf_compute(
                amax,
                self.scale,
                self.quant_max,
                0,
            )
            self.amax_history[0] = torch.amax(torch.abs(X.detach())).float()

        if self.histogram_observer_enabled[0] == 1:
            self._combine_histograms(self.histogram_pre_process, X)

        if self.fake_quant_enabled[0] == 1:
            if self.quantize_per_tensor:
                X = FakeQuantFunction.apply(X, self.value_map, self.scale.to(X.dtype))
            else:
                X = FakeQuantFunction.apply(X, self.value_map)

        if self.histogram_observer_enabled[0] == 1:
            self._combine_histograms(self.histogram_post_process, X)

        return X

    def _combine_histograms(self, histogram: torch.Tensor, X: torch.Tensor) -> None:
        x = X.detach().float()
        x_exp = torch.floor(torch.log2(x[x != 0].abs()))
        combined_histogram = torch.histc(x_exp, 254, min=-126, max=127)
        histogram += combined_histogram

    def save_hist(self, filename):
        hist1 = self.histogram_pre_process.cpu()
        hist2 = self.histogram_post_process.cpu()

        non_empty_bins1 = torch.nonzero(hist1).flatten()
        non_empty_bins2 = torch.nonzero(hist2).flatten()

        if len(non_empty_bins1) == 0 or len(non_empty_bins2) == 0:
            logger.warn("One or both histograms are empty. Skipping plotting.")
            return

        first_non_zero = min(non_empty_bins1[0], non_empty_bins2[0])
        last_non_zero = max(non_empty_bins1[-1], non_empty_bins2[-1])

        hist1 = hist1[first_non_zero:last_non_zero + 1]
        hist2 = hist2[first_non_zero:last_non_zero + 1]

        bins = torch.linspace(-126, 127, 255)
        bins = bins[first_non_zero:last_non_zero + 2]
        bar_width = (bins[1] - bins[0]) * 0.4

        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1] - bar_width/2, hist1, width=bar_width, label='Before quantization')
        plt.bar(bins[:-1] + bar_width/2, hist2, width=bar_width, label='After quantization')

        plt.title('Activation Distribution')
        plt.xlabel('Exponent Value')
        plt.ylabel('Count')
        plt.legend()

        plt.savefig(filename)