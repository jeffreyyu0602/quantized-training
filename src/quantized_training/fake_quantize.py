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


def _get_dtype_to_fq_fn(dtype: str):
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
    elif (match := re.fullmatch(r'int(\d+)', dtype, re.IGNORECASE)):
        nbits = int(match.group(1))
        quant_min, quant_max = -2 ** (nbits - 1), 2 ** (nbits - 1) - 1
        return lambda x: torch.clamp(torch.round(x), quant_min, quant_max)
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")


def  _get_amax(x, qscheme, axis=1, block_size=32):
    if qscheme.value == "per_tensor":
        amax = torch.amax(torch.abs(x))
    elif qscheme.value == "per_channel":
        dim = tuple(range(x.dim()))
        dim = dim[:axis] + dim[axis + 1:]
        amax = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    elif qscheme.value == "per_vector":
        # TODO: handle cases where the vector dimension is not divisible
        # by block_size
        x_shape = x.size()
        new_x_shape = (
            x_shape[:axis] +
            (block_size, x_shape[axis] // block_size) +
            x_shape[axis + 1:]
        )
        y = x.view(new_x_shape)
        amax = torch.amax(torch.abs(y), dim=(0, axis)).unsqueeze(0)
    return amax


# TODO: perform fake quantization directly for simple data types
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

        if scale is not None:
            input = input / scale
        return input

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
        block_size: int = 16,
        record_histogram: bool = False,
        force_scale_power_of_two = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_max = quant_max
        self.amax_history_len = amax_history_len
        self.ch_axis = ch_axis
        self.block_size = block_size
        self.force_scale_power_of_two = force_scale_power_of_two
        self.name = kwargs.get("name", None)
        device = kwargs.get("device", None)
        # Generates a mapping from bfloat16 to quantized values of the given dtype
        input = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)
        self.fake_quant_fn = _get_dtype_to_fq_fn(dtype)
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
            "fake_quant_enabled={}, observer_enabled={}, name={}, dtype={}, "
            "qscheme={}, quant_max={}, amax_history_len={}, ch_axis={}, "
            "block_size={}, record_histogram={}, scale={}, "
            "force_scale_power_of_two={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.name,
                self.dtype,
                self.qscheme,
                self.quant_max,
                self.amax_history_len,
                self.ch_axis,
                self.block_size,
                self.record_histogram,
                self.scale,
                self.force_scale_power_of_two,
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Remove attention mask when computing scaling factor
        # if self.qscheme is not None:
        #     dtype_min = torch.finfo(X.dtype).min
        #     attention_mask = torch.where(x == dtype_min, dtype_min, 0.0)
        #     X -= attention_mask

        if self.record_histogram:
            self._combine_histograms(X)

        if self.observer_enabled[0] == 1:
            x = X.detach()  # avoid keeping autograd tape
            curr_amax = _get_amax(x, self.qscheme, self.ch_axis, self.block_size)
            if self.amax_history is None:
                self.amax_history = torch.zeros(
                    (self.amax_history_len, *curr_amax.shape), device=curr_amax.device
                )

            amax = torch.amax(self.amax_history, dim=0)
            self.amax_history = torch.roll(self.amax_history, -1, 0)
            self.amax_history[0] = curr_amax.float()

            sf = self.quant_max / amax
            if self.force_scale_power_of_two:
                sf = torch.pow(2, torch.floor(torch.log2(sf)))
            sf = torch.where(torch.isfinite(sf), sf, self.scale)
            self.scale = sf

        if self.fake_quant_enabled[0] == 1:
            X = FakeQuantFunction.apply(
                X, self.scale if self.qscheme is not None else None, self.encodings
            )

        # if self.qscheme is not None:
        #     X += attention_mask

        return X

    def _combine_histograms(self, X: torch.Tensor) -> None:
        x = X.detach().float()
        exp = torch.floor(torch.log2(torch.abs(x)))
        self.histogram += torch.histc(exp, 254, min=-126, max=127)