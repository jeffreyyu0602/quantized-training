import math
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch.library import Library, impl

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

quantized_decomposed_lib.define("vmap(Tensor self, Tensor code) -> Tensor")

@impl(quantized_decomposed_lib, "vmap", "CompositeExplicitAutograd")
def vmap(input: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    if input.dtype == torch.bfloat16:
        indices = input.view(torch.int16).to(torch.int32) & 0xffff
    else:
        raw_bits = input.to(torch.float32).view(torch.int32)
        indices = (raw_bits >> 16) & 0xffff
        indices = indices | ((raw_bits & 0xffff) != 0).to(indices.dtype)
    return code[indices].to(input.dtype)

quantized_decomposed_lib.define(
    "quantize(Tensor input, Tensor scale, str? dtype, Tensor code, SymInt? block_size=None) -> Tensor")

@impl(quantized_decomposed_lib, "quantize", "CompositeExplicitAutograd")
def quantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: Optional[str],
    code: torch.Tensor,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       dtype (torch.dtype): requested dtype (e.g. int8) for output Tensor
       code (torch.Tensor): quantization map for mapping from float to quantized values
       block_size (int): block size for microscaling, default is None

    Returns:
       Tensor with requested dtype (e.g. int8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    if block_size is not None:
        scale = _broadcast_shapes(scale, input, block_size)
    return vmap(input / scale, code)

quantized_decomposed_lib.define(
    "dequantize(Tensor input, Tensor scale, str? dtype, Tensor code) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize", "CompositeExplicitAutograd")
def dequantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: Optional[str],
    code: torch.Tensor,
) -> torch.Tensor:
    """ Dequantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       dtype (torch.dtype): requested dtype (e.g. int24) for input Tensor
       code (torch.Tensor): quantization map for mapping from float to quantized values

    Returns:
       Tensor with floating point types, note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    if dtype is not None:
        return vmap(input, code) * scale
    return input * scale

quantized_decomposed_lib.define(
    "conv2d_mx(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, "
    "SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1, *, Tensor? input_scale=None, "
    "Tensor? weight_scale=None, SymInt? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None) -> Tensor")

@impl(quantized_decomposed_lib, "conv2d_mx", "CompositeExplicitAutograd")
def conv2d_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
    *,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    input_code: Optional[torch.Tensor] = None,
    weight_code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input_code is not None:
        input = input_code[input.to(torch.long)]
    if weight_code is not None:
        weight = weight_code[weight.to(torch.long)]

    if input_scale is not None:
        input_scale = _broadcast_shapes(input_scale, input, block_size)
        input = input * input_scale
    if weight_scale is not None:
        weight_scale = _broadcast_shapes(weight_scale, weight, block_size)
        weight = weight * weight_scale

    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

quantized_decomposed_lib.define(
    "linear_mx(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor? input_scale=None, "
    "Tensor? weight_scale=None, SymInt? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None) -> Tensor")

@impl(quantized_decomposed_lib, "linear_mx", "CompositeExplicitAutograd")
def linear_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    *,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    input_code: Optional[torch.Tensor] = None,
    weight_code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input_code is not None:
        input = input_code[input.to(torch.long)]
    if weight_code is not None:
        weight = weight_code[weight.to(torch.long)]

    if input_scale is not None:
        input_scale = _broadcast_shapes(input_scale, input, block_size)
        input = input * input_scale
    if weight_scale is not None:
        weight_scale = _broadcast_shapes(weight_scale, weight, block_size)
        weight = weight * weight_scale

    return F.linear(input, weight, bias)

quantized_decomposed_lib.define(
    "matmul_mx(Tensor self, Tensor other, *, Tensor? input_scale=None, "
    "Tensor? weight_scale=None, SymInt? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None) -> Tensor")

@impl(quantized_decomposed_lib, "matmul_mx", "CompositeExplicitAutograd")
def matmul_mx(
    self: torch.Tensor,
    mat2: torch.Tensor,
    *,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    input_code: Optional[torch.Tensor] = None,
    weight_code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input_code is not None:
        self = input_code[self.to(torch.long)]
    if weight_code is not None:
        mat2 = weight_code[mat2.to(torch.long)]

    if input_scale is not None:
        input_scale = _broadcast_shapes(input_scale, self, block_size)
        self = self * input_scale
    if weight_scale is not None:
        weight_scale = _broadcast_shapes(weight_scale, mat2, block_size)
        mat2 = mat2 * weight_scale

    return torch.matmul(self, mat2)

quantized_decomposed_lib.define(
    "calculate_mx_qparam(Tensor self, int axis, float qmax, int block_size, "
    "bool force_scale_power_of_two=False, Tensor code=None) -> Tensor")

@impl(quantized_decomposed_lib, "calculate_mx_qparam", "CompositeExplicitAutograd")
def calculate_mx_qparam(
    input: torch.Tensor,
    axes: int,
    qmax: float,
    block_size: int,
    force_scale_power_of_two: bool = False,
    code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert block_size > 0

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + input.ndim if x < 0 else x for x in axes]

    # Perform tiling to the hardware vector size
    input, axes, orig_shape, padded_shape = _reshape_to_blocks(
        input, axes, block_size
    )

    shared_exp_axes = [x + 1 for x in axes]

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
    else:
        # Use absolute maximum value to compute scaling factors
        amax = torch.amax(torch.abs(input), dim=shared_exp_axes)
        scale = amax / qmax
        scale = torch.where(amax > 0.0, scale, 1.0)
        scale = torch.where(torch.isfinite(amax), scale, 1.0)

        # Quantize the scale using the codebook
        if code is not None:
            scale = vmap(scale, code)
        scale = torch.where(scale > 0.0, scale, 1.0)
        return scale
