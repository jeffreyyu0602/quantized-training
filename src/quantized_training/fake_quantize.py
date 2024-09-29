import logging
import math
import re
from typing import Optional, List, Callable, Tuple

import torch
from torch import Tensor
from torch.ao.quantization import FakeQuantizeBase, ObserverOrFakeQuantize

import quantized_training as qt
from quantized_training.fp8 import (
    _quantize_elemwise_core,
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
)
from quantized_training.mx_utils import (
    _reshape_to_blocks,
    _shared_exponents,
    _undo_reshape_to_blocks,
)
from quantized_training.normal_float import quantize_to_nf
from quantized_training.posit import quantize_to_posit


__all__ = [
    "FusedAmaxObsFakeQuantize",
    "_DerivedObserverOrFakeQuantize",
]


logger = logging.getLogger(__name__)


def get_fake_quant_fn(dtype: str):
    """Return the quantization function for the given dtype."""
    if (match := re.fullmatch(r'posit(\d+)_(\d+)', dtype)):
        nbits, es = match.groups()
        return lambda x: quantize_to_posit(x, int(nbits), int(es), round_to_even=True)

    if (match := re.fullmatch(r'(?:fp8\.)?(e4m3|e5m2)', dtype, re.IGNORECASE)):
        fp8_format = match.group(1).lower()
        return quantize_to_fp8_e4m3 if fp8_format == 'e4m3' else quantize_to_fp8_e5m2

    if (match := re.fullmatch(r"fp(\d+)_e(\d+)m(\d+)", dtype)):
        nbits, ebits, mbits = map(int, match.groups())
        assert nbits == ebits + mbits + 1
        mbits = mbits + 2
        emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)
        if dtype != "fp8_e4m3":
            max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
        else:
            max_norm = 2**emax * 1.75
        return lambda x: _quantize_elemwise_core(x, mbits, ebits, max_norm, "even", True)

    if (match := re.fullmatch(r'int(\d+)', dtype)):
        nbits = int(match.group(1))
        quant_min, quant_max = -2 ** (nbits - 1), 2 ** (nbits - 1) - 1
        return lambda x: torch.clamp(torch.round(x), quant_min, quant_max)

    if (match := re.fullmatch(r'nf(\d+)', dtype)):
        nbits = int(match.group(1))
        return lambda x: quantize_to_nf(x, nbits)

    raise ValueError(f"Unrecognized dtype: {dtype}")


def _get_quantization_map(dtype, device=None):
    values = torch.arange(2 ** 16, dtype=torch.int16).view(torch.bfloat16)
    if dtype is not None:
        fq_fn = get_fake_quant_fn(dtype)
        values = fq_fn(values)
    return values.to(device)


def _quantize(input, quant_map):
    if input.dtype == torch.bfloat16:
        indices = input.view(torch.int16).to(torch.int32) & 0xffff
    else:
        raw_bits = input.to(torch.float32).view(torch.int32)
        indices = (raw_bits >> 16) & 0xffff
        indices = indices | ((raw_bits & 0xffff) != 0).to(indices.dtype)
    return quant_map[indices].to(input.dtype)


class MXFakeQuantFunction(torch.autograd.Function):
    """This function performs MX quantization by calculating the scaling
    factor using absolute maximum values in the tensor.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        fake_quant_enabled,
        quant_map,
        quant_max,
        shared_exp_method="max",
        axes=None,
        block_size=0,
        force_scale_power_of_two=False,
    ):
        if fake_quant_enabled[0] == 0:
            return input

        axes = [axes] if type(axes) == int else axes
        axes = [x + input.ndim if x < 0 else x for x in axes]

        # Perform tiling to the hardware vector size
        if block_size > 0:
            input, axes, orig_shape, padded_shape = _reshape_to_blocks(
                input, axes, block_size
            )

        shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

        if force_scale_power_of_two:
            # Get shared exponents
            shared_exp = _shared_exponents(
                input, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
            )

            # Offset the max exponent by the largest representable exponent
            # in the element data format
            shared_exp = shared_exp - math.floor(math.log2(quant_max))

            scale = (2 ** shared_exp).to(input.dtype)
        else:
            # Use absolute maximum value for scaling
            amax = torch.amax(torch.abs(input), dim=shared_exp_axes, keepdim=True)
            scale = amax / quant_max
            scale = torch.where(amax > 0.0, scale, 1.0)
            scale = torch.where(torch.isfinite(amax), scale, 1.0)

        input = _quantize(input / scale, quant_map) * scale

        # Undo tile reshaping
        if block_size:
            input = _undo_reshape_to_blocks(input, padded_shape, orig_shape, axes)

        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None, None


class FusedAmaxObsFakeQuantFunction(torch.autograd.Function):
    """This function observes the amax statistics of inputs and
    quantize the inputs based on the observed amax values.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        observer_enabled: torch.Tensor,
        fake_quant_enabled: torch.Tensor,
        quant_map: torch.Tensor,
        amax_history: torch.Tensor,
        scale: torch.Tensor,
        amax_history_len: int,
        quant_max: float,
        ch_axis: Optional[int] = None,
        per_row_fake_quant=False,
        force_scale_power_of_two=False,
    ) -> torch.Tensor:
        if observer_enabled[0] == 1:
            if per_row_fake_quant:
                ch_axis = ch_axis + input.ndim if ch_axis < 0 else ch_axis
                dim = tuple(i for i in range(input.ndim) if i != ch_axis)
                amax_cur = torch.amax(torch.abs(input), dim=dim, keepdim=True)
            else:
                amax_cur = torch.amax(torch.abs(input))

            if amax_history.numel() == 0:
                size = amax_cur.shape
                amax_history.resize_((amax_history_len,) + size).fill_(0.0)
                scale.resize_(size).fill_(1.0)

            amax = torch.amax(amax_history, dim=0)

            if amax_history.shape[0] > 1:
                new_amax_history = torch.roll(amax_history, -1, 0)
                amax_history.copy_(new_amax_history)
            amax_history[0] = amax_cur

            sf = amax / quant_max
            sf = torch.where(amax > 0.0, sf, scale)
            sf = torch.where(torch.isfinite(amax), sf, scale)
            if force_scale_power_of_two:
                sf = torch.pow(2, torch.floor(torch.log2(sf)))
            scale.copy_(sf)

        if fake_quant_enabled[0] == 1:
            scale = scale.to(input.dtype)
            input = _quantize(input / scale, quant_map) * scale

        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None, None, None, None, None


class FusedAmaxObsFakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.
    
    Observer module for computing the quantization parameters based on the
    historical amax values.

    This observer uses the tensor amax statistics to compute the quantization
    parameters. The module records the maximums of absolute values of output
    tensors, and uses this statistic to compute the quantization parameters.
    """

    quant_map: torch.Tensor
    amax_history: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        dtype: str,
        qscheme: Optional[str] = None,
        quant_max: Optional[float] = None,
        amax_history_len: int = None,
        ch_axis: Optional[int] = None,
        block_size: Optional[int] = None,
        record_histogram: bool = False,
        force_scale_power_of_two: bool = False,
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
        self.shared_exp_method = kwargs.get("shared_exp_method", "max")
        self.outlier_threshold = kwargs.get("outlier_threshold", None)
        device = kwargs.get("device", None)
        # Generate a quantization map from bfloat16 to quantized values of the given dtype
        values = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)
        fake_quant_fn = get_fake_quant_fn(dtype)
        self.register_buffer("quant_map", fake_quant_fn(values), persistent=False)
        # Create amax history and scale buffers
        factory_kwargs = {'device': device, 'dtype': torch.float}
        self.register_buffer("amax_history", torch.tensor([], **factory_kwargs))
        self.register_buffer('scale', torch.tensor([1.0], **factory_kwargs))
        self.observer_enabled[0] = self.qscheme is not None
        self.is_per_channel = self.qscheme == qt.per_channel_symmetric
        # Create histogram buffer
        self.record_histogram = record_histogram
        self.register_buffer("histogram", torch.zeros(254, **factory_kwargs), persistent=False)

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, dtype={}, amax_history_len={}, "
            "quant_max={}, qscheme={}, ch_axis={}, block_size={}, force_scale_power_of_two={}, "
            "scale={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.dtype,
                self.amax_history_len,
                self.quant_max,
                self.qscheme,
                self.ch_axis,
                self.block_size,
                self.force_scale_power_of_two,
                self.scale,
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO this is a workaround when input is not on the same device as the module
        devices = {p.device for p in self.buffers()}
        if len(devices) != 1 or next(iter(devices)) != X.device:
            self.to(X.device)

        if self.record_histogram:
            exp = torch.floor(torch.log2(torch.abs(X.detach().float())))
            self.histogram += torch.histc(exp, 254, min=-126, max=127)

        # Remove outliers from the activation. This needs to be inside
        # torch.autograd.Function for training.
        if self.outlier_threshold is not None:
            X = X.clone()
            outliers = X.abs() > self.outlier_threshold
            outliers_magnitude = X[outliers]
            X[outliers] = 0.0

            outlier_pct = outliers.sum().item() / X.numel()
            self.max_outlier_pct = max(outlier_pct, getattr(self, "max_outlier_pct", 0.0))

        if self.qscheme == qt.microscaling:
            X = MXFakeQuantFunction.apply(
                X,
                self.fake_quant_enabled,
                self.quant_map,
                self.quant_max,
                self.shared_exp_method,
                self.ch_axis,
                self.block_size,
                self.force_scale_power_of_two,
            )
        else:
            X = FusedAmaxObsFakeQuantFunction.apply(
                X,
                self.observer_enabled,
                self.fake_quant_enabled,
                self.quant_map,
                self.amax_history,
                self.scale,
                self.amax_history_len,
                self.quant_max,
                self.ch_axis,
                self.is_per_channel,
                self.force_scale_power_of_two,
            )

        # Restore outliers
        if self.outlier_threshold is not None:
            X[outliers] = outliers_magnitude

        return X

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'amax_history']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and amax_history
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'amax_history'
                    self.amax_history.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'amax_history'
                        self.amax_history.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class _DerivedObserverOrFakeQuantize(FakeQuantizeBase):
    r"""This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """

    def __init__(
        self,
        dtype: torch.dtype,
        obs_or_fqs: List[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[[List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]],
    ):
        super().__init__()
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        self.register_buffer("quant_map", _get_quantization_map(dtype), persistent=False)
        self.observer_enabled[0] = 0
        self.dtype = dtype
        self.qscheme = obs_or_fqs[1].qscheme

    def forward(self, x: Tensor) -> Tensor:
        devices = {p.device for p in self.buffers()}
        if len(devices) != 1 or next(iter(devices)) != x.device:
            self.to(x.device)

        return FusedAmaxObsFakeQuantFunction.apply(
            x,
            self.observer_enabled,
            self.fake_quant_enabled,
            self.quant_map,
            None,
            self.calculate_qparams(),
            None,
            None,
        )

    def calculate_qparams(self):
        return self.derive_qparams_fn(self.obs_or_fqs)
