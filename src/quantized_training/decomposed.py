import math
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch.library import Library, impl

from .mx_utils import _reshape_to_blocks, _shared_exponents


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_ops", "DEF")

quantized_decomposed_lib.define(
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, "
    "SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor"
)

@impl(quantized_decomposed_lib, "conv2d", "CompositeExplicitAutograd")
def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    return F.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )

quantized_decomposed_lib.define(
    "linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"
)

@impl(quantized_decomposed_lib, "linear", "CompositeExplicitAutograd")
def linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return F.linear(input, weight, bias)

quantized_decomposed_lib.define(
    "max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, "
    "int[2] dilation=1, bool ceil_mode=False) -> Tensor"
)

@impl(quantized_decomposed_lib, "max_pool2d", "CompositeExplicitAutograd")
def max_pool2d(
    self: torch.Tensor,
    kernel_size: Union[int, Tuple[int]] = 1,
    stride: Union[int, Tuple[int]] = None,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
) -> torch.Tensor:
    return F.max_pool2d(
        self.permute(0, 3, 1, 2),
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    ).permute(0, 2, 3, 1)

quantized_decomposed_lib.define(
    "adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor"
)

@impl(quantized_decomposed_lib, "adaptive_avg_pool2d", "CompositeExplicitAutograd")
def adaptive_avg_pool2d(self: torch.Tensor, output_size: Union[int, Tuple[int]] = 1) -> torch.Tensor:
    return F.adaptive_avg_pool2d(self.permute(0, 3, 1, 2), output_size).permute(0, 2, 3, 1)

def expand(input, shape, block_size):
    while input.ndim < len(shape):
        input = input.unsqueeze(0)

    # Repeat the input along each dimension to match the target shape
    for dim in range(len(shape)):
        if input.shape[dim] != shape[dim]:
            input = torch.repeat_interleave(input, block_size, dim)

    # If the shape is not a multiple of block_size, we may need to slice
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
    "quantize(Tensor input, Tensor scale, str? dtype, Tensor code, int? block_size=None, Tensor quant_code=None, Tensor? zero_point=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize", "CompositeExplicitAutograd")
def quantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: Optional[str],
    code: torch.Tensor,
    block_size: Optional[int] = None,
    quant_code: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ Quantization for the Tensor using scales and zero points to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (torch.Tensor): quantization parameter for quantization
       zero_point (torch.Tensor): zero point for quantization, default is None
       dtype (str): requested dtype (e.g. int8) for output Tensor
       code (torch.Tensor): quantization map for mapping from float to quantized values
       block_size (int): block size for microscaling, default is None
       quant_code (torch.Tensor): codebook for quantizing the outputs

    Returns:
       Tensor with requested dtype (e.g. int8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    if block_size is not None:
        scale = expand(scale, input.shape, block_size)
        if zero_point is not None:
            zero_point = expand(zero_point, input.shape, block_size)

    if zero_point is None:
        input = input / scale
    else:
        input = input / scale + zero_point

    return vmap(input, code)


quantized_decomposed_lib.define(
    "dequantize(Tensor input, Tensor scale, Tensor? zero_point, str? dtype, Tensor code, Tensor? zero_point=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize", "CompositeExplicitAutograd")
def dequantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    dtype: Optional[str],
    code: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ Dequantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (torch.Tensor): quantization parameter for affine quantization
       dtype (str): requested dtype (e.g. int24) for input Tensor
       code (torch.Tensor): quantization map for mapping from float to quantized values

    Returns:
       Tensor with floating point types, note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """

    if dtype is not None:
        input = vmap(input, code)
    return input * scale if zero_point is None else (input - zero_point) * scale


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
        input = input_code[input.to(torch.long)].to(input.dtype)
    if weight_code is not None:
        weight = weight_code[weight.to(torch.long)].to(weight.dtype)

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
        input = input_code[input.to(torch.long)].to(input.dtype)
    if weight_code is not None:
        weight = weight_code[weight.to(torch.long)].to(weight.dtype)

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
        self = input_code[self.to(torch.long)].to(self.dtype)
    if weight_code is not None:
        other = weight_code[other.to(torch.long)].to(other.dtype)

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


def to_csr(tensor: torch.Tensor):
    assert not tensor.is_sparse, "Expected a dense tensor (not PyTorch's sparse format)"

    tensor = tensor.reshape(-1, tensor.shape[-1])
    rows, cols = tensor.shape
    data = []
    indices = []
    indptr = [0]

    nnz = 0
    for row in range(rows):
        row_data = tensor[row]
        for col in range(cols):
            val = row_data[col].item()
            if val != 0:
                data.append(val)
                indices.append(col)
                nnz += 1
        indptr.append(nnz)

    data = torch.tensor(data, dtype=tensor.dtype)
    indices = torch.tensor(indices, dtype=torch.int32)
    indptr = torch.tensor(indptr, dtype=torch.int32)

    return data, indices, indptr


quantized_decomposed_lib.define(
    "filter_outlier(Tensor input, float threshold=6.0) -> (Tensor, Tensor, Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "filter_outlier", "CompositeExplicitAutograd")
def filter_outlier(input: torch.Tensor, threshold: float = 6.0) -> Tuple[torch.Tensor]:
    """Filter out outliers in the input tensor based on a threshold.

    Args:
        input (torch.Tensor): Input tensor.
        threshold (float): Threshold for filtering out outliers.

    Returns:
        torch.Tensor: Filtered tensor.
    """
    is_outlier = torch.abs(input) > threshold
    inlier = torch.where(is_outlier, 0, input)
    outliers = torch.where(is_outlier, input, 0)
    csr = to_csr(outliers)
    return inlier, *csr


quantized_decomposed_lib.define(
    "spmm_csr(Tensor data, Tensor indices, Tensor indptr, Tensor B, Tensor? B_scale=None, "
    "Tensor? B_code=None, int? block_size=None, bool weight_transposed=False) -> Tensor"
)


@impl(quantized_decomposed_lib, "spmm_csr", "CompositeExplicitAutograd")
def spmm_csr(
    data: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor] = None,
    B_code: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    weight_transposed=False
) -> torch.Tensor:
    """
    Performs SpMM: Y = A @ B where A is in CSR format.

    Args:
        data    : 1D tensor of non-zero values in A (shape [nnz])
        indices : 1D tensor of column indices for each non-zero value (shape [nnz])
        indptr  : 1D tensor of row pointers (shape [num_rows + 1])
        B       : Dense matrix of shape [K, N]

    Returns:
        Y       : Dense matrix of shape [M, K]
    """
    M = indptr.numel() - 1
    K = B.shape[1] if weight_transposed else B.shape[0]
    Y = torch.zeros((M, K), dtype=data.dtype, device=data.device)

    if B_code is not None:
        B = B_code[B.to(torch.long)]
    if B_scale is not None:
        B = B * expand(B_scale, B.shape, block_size)

    if weight_transposed:
        B = B.T

    for row in range(M):
        start = indptr[row].item()
        end = indptr[row + 1].item()
        for i in range(start, end):
            col = indices[i].item()
            Y[row] += data[i] * B[:,col]

    return Y
