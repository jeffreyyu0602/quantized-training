import math
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch.library import Library, impl

from .mx_utils import _reshape_to_blocks, _shared_exponents


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_ops", "DEF")

def expand(input, shape, block_size):
    while input.ndim < len(shape):
        input = input.unsqueeze(0)

    for dim in range(len(shape)):
        if input.shape[dim] != shape[dim]:
            input = torch.repeat_interleave(input, block_size, dim)

    if list(input.shape) != list(shape):
        slices = [slice(0, x) for x in shape]
        input = input[slices]
    return input


quantized_decomposed_lib.define("vmap(Tensor self, Tensor other) -> Tensor")


@impl(quantized_decomposed_lib, "vmap", "CompositeExplicitAutograd")
def vmap(input: torch.Tensor, code: torch.Tensor, chunk_size=65536) -> torch.Tensor:
    if input.dtype == torch.bfloat16:
        indices = input.view(torch.int16).to(torch.int32) & 0xffff
    else:
        raw_bits = input.to(torch.float32).view(torch.int32)
        indices = (raw_bits >> 16) & 0xffff
        indices = indices | ((raw_bits & 0xffff) != 0).to(indices.dtype)

    output = torch.empty_like(input, memory_format=torch.contiguous_format)
    indices_flat = indices.reshape(-1)
    output_flat = output.view(-1)

    for start in range(0, indices_flat.numel(), chunk_size):
        end = min(start + chunk_size, indices_flat.numel())
        output_flat[start:end] = code[indices_flat[start:end]]

    return output


quantized_decomposed_lib.define(
    "quantize(Tensor input, Tensor scale, str? dtype, Tensor code, int? block_size=None, "
    "Tensor quant_code=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize", "CompositeExplicitAutograd")
def quantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: Optional[str],
    code: torch.Tensor,
    block_size: Optional[int] = None,
    quant_code: Optional[torch.Tensor] = None,
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
        scale = expand(scale, input.shape, block_size)
    return vmap(input / scale, code)


quantized_decomposed_lib.define(
    "dequantize(Tensor input, Tensor scale, str? dtype, Tensor code) -> Tensor"
)


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
    "Tensor? weight_scale=None, int? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None) -> Tensor"
)


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
    # For codebook quantization, decode input and weight into float values first
    if input_code is not None:
        input = input_code[input.to(torch.long)]
    if weight_code is not None:
        weight = weight_code[weight.to(torch.long)]

    # Replicate scales to match input and weight shapes
    if input_scale is not None:
        input = input * expand(input_scale, input.shape, block_size)
    if weight_scale is not None:
        weight = weight * expand(weight_scale, weight.shape, block_size)

    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


quantized_decomposed_lib.define(
    "linear_mx(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor? input_scale=None, "
    "Tensor? weight_scale=None, int? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None) -> Tensor"
)


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
        input = input * expand(input_scale, input.shape, block_size)
    if weight_scale is not None:
        weight = weight * expand(weight_scale, weight.shape, block_size)

    return F.linear(input, weight, bias)


quantized_decomposed_lib.define(
    "matmul_mx(Tensor self, Tensor other, *, Tensor? input_scale=None, Tensor? weight_scale=None, "
    "int? block_size=None, Tensor? input_code=None, Tensor? weight_code=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "matmul_mx", "CompositeExplicitAutograd")
def matmul_mx(
    self: torch.Tensor,
    other: torch.Tensor,
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
        other = weight_code[other.to(torch.long)]

    if input_scale is not None:
        self = self * expand(input_scale, self.shape, block_size)
    if weight_scale is not None:
        other = other * expand(weight_scale, other.shape, block_size)

    return torch.matmul(self, other)


quantized_decomposed_lib.define(
    "calculate_mx_qparam(Tensor self, int axis, float quant_max, int block_size, "
    "bool force_scale_power_of_two=False, Tensor code=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "calculate_mx_qparam", "CompositeExplicitAutograd")
def calculate_mx_qparam(
    input: torch.Tensor,
    axis: int,
    quant_max: float,
    block_size: int,
    force_scale_power_of_two: bool = False,
    code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert block_size > 0

    # Make sure axes is a list of non-negative numbers
    axes = [axis + input.ndim if axis < 0 else axis]

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
        shared_exp = shared_exp - math.floor(math.log2(quant_max))

        for axis in reversed(axes):
            # Remove extra dimension
            shared_exp = torch.squeeze(shared_exp, dim=axis + 1)

        scale = 2 ** shared_exp
    else:
        # Use absolute maximum value to compute scaling factors
        amax = torch.amax(torch.abs(input), dim=shared_exp_axes)
        scale = amax / quant_max

        # Quantize the scale using the codebook
        if code is not None:
            scale = vmap(scale, code)

    scale = torch.where(scale > 0.0, scale, 1.0)
    return scale


quantized_decomposed_lib.define(
    "quantize_mx(Tensor self, int axis, float quant_max, int block_size, str dtype, Tensor code, "
    "bool force_scale_power_of_two=False, Tensor scale_code=None, Tensor quant_code=None) -> (Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "quantize_mx", "CompositeExplicitAutograd")
def quantize_mx(
    input: torch.Tensor,
    axis: int,
    quant_max: float,
    block_size: int,
    dtype: str,
    code: torch.Tensor,
    force_scale_power_of_two: bool = False,
    scale_code: Optional[torch.Tensor] = None,
    quant_code: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    scale = calculate_mx_qparam(
        input,
        axis=axis,
        quant_max=quant_max,
        block_size=block_size,
        force_scale_power_of_two=force_scale_power_of_two,
        code=scale_code,
    )
    input = quantize(input, scale, dtype, code, block_size)
    return scale, input
