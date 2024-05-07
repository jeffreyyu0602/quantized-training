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


def _round_mantissa(input, mbits):
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


def _clamp_exp(x, ebits):
    exp_bias = 2 ** (ebits - 1) - 1
    min_exp, max_exp = 1 - exp_bias, 2 ** ebits - 2 - exp_bias
    exponent = torch.floor(torch.log2(x))
    x = torch.where(exponent < min_exp, 0, x)
    x = torch.where(exponent > max_exp, 2 ** (max_exp + 1), x)
    return x


def _get_fq_fn(dtype: str):
    """Return the quantization function for the given dtype."""
    if (match := re.fullmatch(r'posit(\d+)_(\d+)', dtype)):
        nbits, es = match.groups()
        return lambda x: quantize_to_posit(x, int(nbits), int(es), round_to_even=True)
    elif (match := re.fullmatch(r'(?:FP8\.)?(E4M3|E5M2)', dtype, re.IGNORECASE)):
        fp8_format = match.group(1).lower()
        return quantize_to_fp8_e4m3 if fp8_format == 'e4m3' else quantize_to_fp8_e5m2
    elif (match := re.fullmatch(r'E8M(\d+)', dtype, re.IGNORECASE)):
        mbits = int(match.group(1))
        assert mbits >= 0 and mbits <= 23, "mbits must be between 0 and 23"
        return lambda x: _round_mantissa(x, mbits)
    elif (match := re.fullmatch(r'E(\d+)MY', dtype, re.IGNORECASE)):
        ebits = int(match.group(1))
        return lambda x: _clamp_exp(x, ebits)
    elif dtype == "binary":
        return lambda x: torch.sign(x - x.mean())
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")


def _get_max_val(x, qscheme, ch_axis=1, block_size=32):
    if qscheme == "per_tensor":
        max_val = torch.amax(torch.abs(x))
    elif qscheme == "per_channel":
        y = torch.flatten(x.transpose(0, ch_axis), start_dim=1)
        max_val = torch.amax(torch.abs(y), dim=1)
    elif qscheme == "per_vector":
        x_dim = x.size()
        sp_axis = len(x_dim) - 2
        new_x_shape = (
            x_dim[:sp_axis] +
            (block_size, x_dim[sp_axis] // block_size) +
            x_dim[sp_axis + 1:]
        )
        y = x.view(new_x_shape)
        max_val = torch.amax(torch.abs(y), dim=(0, sp_axis)).unsqueeze(0)
    else:
        raise ValueError(f"Unrecognized qscheme: {qscheme}")
    return max_val


class FakeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, encodings):
        if scale is not None:
            scale = scale.to(input.dtype)
            input = input * scale

        if input.dtype == torch.bfloat16:
            indices = input.view(torch.int16).int() & 0xffff
        else:
            raw_bits = input.float().view(torch.int32)
            indices = ((raw_bits >> 16) & 0xffff) | ((raw_bits & 0xffff) != 0).int()
        input = encodings[indices].to(input.dtype)

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
    encodings: torch.Tensor
    amax_history: torch.Tensor
    scale: torch.Tensor
    histogram: torch.Tensor

    def __init__(
        self,
        dtype: str,
        qscheme: str = None,
        quant_max: float = None,
        amax_history_len: int = 50,
        ch_axis: int = 1,
        block_size: int = 8,
        record_histogram: bool = False,
        name=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_max = quant_max
        self.amax_history_len = amax_history_len
        self.ch_axis = ch_axis
        self.block_size = block_size
        self.name = name
        device = kwargs.get("device", None)
        # Generates a mapping from bfloat16 to quantized values of the given dtype
        input = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)
        self.fake_quant_fn = _get_fq_fn(dtype)
        self.register_buffer("encodings", self.fake_quant_fn(input), persistent=False)
        # Create amax history and scale buffer
        factory_kwargs = {'device': device, 'dtype': torch.float}
        self.enable_observer(qscheme is not None)
        if qscheme is not None:
            self.register_buffer("amax_history", None)
            self.register_buffer('scale', torch.tensor([1.0], **factory_kwargs))
        # Create histogram buffer
        if record_histogram:
            self.register_buffer("histogram", torch.zeros(254, **factory_kwargs))
        self.record_histogram = record_histogram

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, dtype={}, qscheme={}, "
            "quant_max={}, amax_history_len={}, ch_axis={}, block_size={}, "
            "name={}, record_histogram={}, scale={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.dtype,
                self.qscheme,
                self.quant_max,
                self.amax_history_len,
                self.ch_axis,
                self.block_size,
                self.name,
                self.record_histogram,
                self.scale,
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.record_histogram:
            self._combine_histograms(X)

        if self.observer_enabled[0] == 1:
            x = X.detach()  # avoid keeping autograd tape

            # Remove attention mask when computing scaling factor
            # dtype_min = torch.finfo(X.dtype).min
            # attention_mask = torch.where(x == dtype_min, dtype_min, 0.0)
            # X -= attention_mask

            max_val = _get_max_val(x, self.qscheme, self.ch_axis, self.block_size)
            if self.amax_history is None:
                self.amax_history = torch.zeros(
                    (self.amax_history_len, *max_val.shape), device=max_val.device
                )

            amax = torch.amax(self.amax_history, dim=0)
            self.amax_history = torch.roll(self.amax_history, -1, 0)
            self.amax_history[0] = max_val.float()

            sf = torch.pow(2, torch.floor(torch.log2(self.quant_max / amax)))
            self.scale = torch.where(torch.isfinite(sf), sf, self.scale)

        if self.fake_quant_enabled[0] == 1:
            X = FakeQuantFunction.apply(
                X, self.scale if self.qscheme is not None else None, self.encodings
            )
            # X += attention_mask

        return X

    def _combine_histograms(self, X: torch.Tensor) -> None:
        x = X.detach().float()
        exponent = torch.floor(torch.log2(torch.abs(x)))
        hist = torch.histc(exponent, 254, min=-126, max=127)
        self.histogram += hist
