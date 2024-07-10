import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.library import Library, impl

from .fake_quantize import _quantize
from .mx_utils import _reshape_to_blocks, _shared_exponents, _undo_reshape_to_blocks


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_ops", "DEF")

quantized_decomposed_lib.define(
    "quantize_symmetric(Tensor input, Tensor scale, str dtype, Tensor qvalues) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_symmetric", "CompositeExplicitAutograd")
def quantize_symmetric(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: str,
    qvalues: torch.Tensor,
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       dtype (torch.dtype): requested dtype (e.g. int8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. int8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    output = _quantize(input / scale, qvalues)
    if not hasattr(output, 'meta'):
        output.meta = {}
    output.meta['dtype'] = dtype
    return output

quantized_decomposed_lib.define(
    "dequantize_symmetric(Tensor input, Tensor scale, str dtype, Tensor qvalues) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_symmetric", "CompositeExplicitAutograd")
def dequantize_symmetric(
        input: torch.Tensor,
        scale: torch.Tensor,
        dtype: str,
        qvalues: torch.Tensor,
) -> torch.Tensor:
    """ Dequantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       dtype (torch.dtype): requested dtype (e.g. int24) for input Tensor

    Returns:
       Tensor with floating point types, note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    input = _quantize(input, qvalues)
    # TODO: cannot annotate the input tensor, need to find a way to do this
    if not hasattr(input, 'meta'):
        input.meta = {}
    input.meta['dtype'] = dtype
    return input * scale

quantized_decomposed_lib.define(
    "quantize_mx_dynamic(Tensor input, Tensor qvalues, str dtype, float quant_max, SymInt axes, SymInt block_size) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_mx_dynamic", "CompositeExplicitAutograd")
def quantize_mx_dynamic(
    input: torch.Tensor,
    qvalues: torch.Tensor,
    dtype: str,
    quant_max: float,
    axes: Union[int, Tuple[int]],
    block_size: int = 32,
) -> torch.Tensor:
    axes = [axes] if type(axes) == int else axes
    axes = [x + input.ndim if x < 0 else x for x in axes]

    # Perform tiling to the hardware vector size
    if block_size > 0:
        input_reshaped, axes, orig_shape, padded_shape = _reshape_to_blocks(
            input, axes, block_size
        )

    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Get shared exponents
    shared_exp = _shared_exponents(
        input_reshaped, method="max", axes=shared_exp_axes, ebits=0,
    )

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = shared_exp - math.floor(math.log2(quant_max))

    scale = (2**shared_exp).to(input.dtype)
    scale = torch.repeat_interleave(scale, block_size, shared_exp_axes[0])

    # Undo reshape to scale
    if block_size:
        scale = _undo_reshape_to_blocks(scale, padded_shape, orig_shape, axes)

    assert input.shape == scale.shape, (
        f"input shape {input.shape} != scale shape {scale.shape}"
    )

    input_q = _quantize(input / scale, qvalues)
    if not hasattr(input_q, 'meta'):
        input_q.meta = {}
    input_q.meta['dtype'] = dtype
    input_q.meta['scale'] = scale
    return input_q

quantized_decomposed_lib.define(
    "conv2d_mx(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor")

@impl(quantized_decomposed_lib, "conv2d_mx", "CompositeExplicitAutograd")
def conv2d_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    if hasattr(input, 'meta') and 'scale' in input.meta:
        input = input * input.meta['scale']
    if hasattr(weight, 'meta') and 'scale' in weight.meta:
        weight = weight * weight.meta['scale']
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

quantized_decomposed_lib.define(
    "linear_mx(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor")

@impl(quantized_decomposed_lib, "linear_mx", "CompositeExplicitAutograd")
def linear_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    if hasattr(input, 'meta') and 'scale' in input.meta:
        input = input * input.meta['scale']
    if hasattr(weight, 'meta') and 'scale' in weight.meta:
        weight = weight * weight.meta['scale']
    return F.linear(input, weight, bias)

quantized_decomposed_lib.define(
    "matmul_mx(Tensor self, Tensor other) -> Tensor")

@impl(quantized_decomposed_lib, "matmul_mx", "CompositeExplicitAutograd")
def matmul_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if hasattr(input, 'meta') and 'scale' in input.meta:
        input = input * input.meta['scale']
    if hasattr(weight, 'meta') and 'scale' in weight.meta:
        weight = weight * weight.meta['scale']
    return torch.matmul(input, weight)
