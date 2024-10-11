import math
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch.library import Library, impl

from .fake_quantize import _quantize
from .mx_utils import _reshape_to_blocks, _shared_exponents


def _broadcast_shapes(input, target, block_size=32):
    for dim in range(len(target.shape)):
        if dim >= len(input.shape):
            break
        if input.shape[dim] != target.shape[dim]:
            input = torch.repeat_interleave(input, block_size, dim)
    if list(input.shape) != list(target.shape):
        slices = [slice(0, x) for x in target.shape]
        input = input[slices]
    return input

# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_ops", "DEF")

quantized_decomposed_lib.define(
    "quantize_symmetric(Tensor input, Tensor scale, str dtype, Tensor quant_map, SymInt? block_size=None) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_symmetric", "CompositeExplicitAutograd")
def quantize_symmetric(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: str,
    quant_map: torch.Tensor,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       dtype (torch.dtype): requested dtype (e.g. int8) for output Tensor
       quant_map (torch.Tensor): quantization map for mapping from float to quantized values
       block_size (int): block size for microscaling, default is None

    Returns:
       Tensor with requested dtype (e.g. int8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    if block_size is not None:
        scale = _broadcast_shapes(scale, input, block_size)
    return _quantize(input / scale, quant_map)

quantized_decomposed_lib.define(
    "dequantize_symmetric(Tensor input, Tensor scale, str? dtype, Tensor quant_map) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_symmetric", "CompositeExplicitAutograd")
def dequantize_symmetric(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: Optional[str],
    quant_map: torch.Tensor,
) -> torch.Tensor:
    """ Dequantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       dtype (torch.dtype): requested dtype (e.g. int24) for input Tensor
       quant_map (torch.Tensor): quantization map for mapping from float to quantized values

    Returns:
       Tensor with floating point types, note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    if dtype is not None:
        return _quantize(input, quant_map) * scale
    return input * scale

quantized_decomposed_lib.define(
    "conv2d_mx(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1, Tensor? scale_inp=None, Tensor? scale_wt=None, SymInt? block_size=None) -> Tensor")

@impl(quantized_decomposed_lib, "conv2d_mx", "CompositeExplicitAutograd")
def conv2d_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
    scale_inp: Optional[torch.Tensor] = None,
    scale_wt: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    if scale_inp is not None:
        scale_inp = _broadcast_shapes(scale_inp, input, block_size)
        input = input * scale_inp
    if scale_wt is not None:
        scale_wt = _broadcast_shapes(scale_wt, weight, block_size)
        weight = weight * scale_wt
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

quantized_decomposed_lib.define(
    "linear_mx(Tensor input, Tensor weight, Tensor? bias=None, Tensor? scale_inp=None, Tensor? scale_wt=None, SymInt? block_size=None) -> Tensor")

@impl(quantized_decomposed_lib, "linear_mx", "CompositeExplicitAutograd")
def linear_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    scale_inp: Optional[torch.Tensor] = None,
    scale_wt: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    if scale_inp is not None:
        scale_inp = _broadcast_shapes(scale_inp, input, block_size)
        input = input * scale_inp
    if scale_wt is not None:
        scale_wt = _broadcast_shapes(scale_wt, weight, block_size)
        weight = weight * scale_wt
    return F.linear(input, weight, bias)

quantized_decomposed_lib.define(
    "matmul_mx(Tensor self, Tensor other, Tensor? scale_inp=None, Tensor? scale_wt=None, SymInt? block_size=None) -> Tensor")

@impl(quantized_decomposed_lib, "matmul_mx", "CompositeExplicitAutograd")
def matmul_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_inp: Optional[torch.Tensor] = None,
    scale_wt: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    if scale_inp is not None:
        scale_inp = _broadcast_shapes(scale_inp, input, block_size)
        input = input * scale_inp
    if scale_wt is not None:
        scale_wt = _broadcast_shapes(scale_wt, weight, block_size)
        weight = weight * scale_wt
    return torch.matmul(input, weight)

quantized_decomposed_lib.define(
    "calculate_mx_qparam(Tensor self, int axis, float qmax, int block_size, bool force_scale_power_of_two=False) -> Tensor")

@impl(quantized_decomposed_lib, "calculate_mx_qparam", "CompositeExplicitAutograd")
def calculate_mx_qparam(
    input: torch.Tensor,
    axes: int,
    qmax: float,
    block_size: int,
    force_scale_power_of_two: bool = False,
) -> torch.Tensor:
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
            input, method="max", axes=shared_exp_axes, ebits=0,
        )

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - math.floor(math.log2(qmax))

        for axis in reversed(axes):
            # Remove extra dimension
            shared_exp = torch.squeeze(shared_exp, dim=axis + 1)

        return 2 ** shared_exp

    # Use absolute maximum value for scaling
    amax = torch.amax(torch.abs(input), dim=shared_exp_axes)
    scale = amax / qmax
    scale = torch.where(amax > 0.0, scale, 1.0)
    scale = torch.where(torch.isfinite(amax), scale, 1.0)
    return scale
