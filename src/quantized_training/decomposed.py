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

    return _quantize(input / scale, qvalues)
    print('quantize')
    print(scale)
    print(input)
    output = _quantize(input / scale, qvalues)
    print(output)
    return output

# Note: dtype/qvalues are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_symmetric(Tensor input, Tensor scale, str dtype, Tensor qvalues) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_symmetric", "CompositeExplicitAutograd")
def dequantize_symmetric(
        input: torch.Tensor,
        scale: torch.Tensor,
        dtype: str,
        qvalues: torch.Tensor,
) -> torch.Tensor:
    return input * scale
    print('dequantize')
    print(scale)
    print(input)
    print(input * scale)
    return input * scale
