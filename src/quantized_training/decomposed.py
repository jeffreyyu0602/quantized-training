import torch
from torch.library import Library, impl

from .fake_quantize import _quantize


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
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
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
    input = _quantize(input, qvalues)
    # TODO: this annotation doesn't work right now
    if not hasattr(input, 'meta'):
        input.meta = {}
    input.meta['dtype'] = dtype
    return input * scale
