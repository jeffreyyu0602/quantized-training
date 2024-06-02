import math
import torch

__all__ = [
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "_quantize_elemwise_core",
]

def quantize_to_fp8_e4m3(
    input: torch.Tensor,
    mbits: int = 3,
    fp8_max: float = 448,
    fp8_min: float = 2 ** -6,
) -> torch.Tensor:
    raw_bits = input.clone().to(torch.float).view(torch.int32)
    exp = ((raw_bits & 0x7f800000) >> 23) - 127
    fraction = (raw_bits & 0x7fffff) | 0x800000

    min_exp = math.floor(math.log2(fp8_min))
    nf_mask = 23 - mbits + torch.clamp(min_exp - exp, min=0)
    lb = (fraction & (1 << nf_mask)).bool()
    gb = (fraction & (1 << (nf_mask - 1))).bool()
    sb = (fraction & ((1 << (nf_mask - 1)) - 1)).bool()
    rb = (lb & gb) | (gb & sb)

    nf_mask_clamped = torch.clamp(nf_mask, max=23)
    raw_bits &= (-1 << nf_mask_clamped)
    raw_bits = torch.where(rb, raw_bits + (1 << nf_mask_clamped), raw_bits)

    output = raw_bits.view(torch.float)
    output = torch.clamp(output, min=-fp8_max, max=fp8_max)
    output = torch.where(torch.abs(input) <= fp8_min * (2 ** -(mbits + 1)), 0, output)

    output = torch.where(input == 0, 0, output)
    output = torch.where(torch.isfinite(input), output, torch.nan)
    return output.to(input.dtype)


def quantize_to_fp8_e5m2(
    input: torch.Tensor,
    mbits: int = 2,
    fp8_max: float = 57344,
    fp8_min: float = 2 ** -14,
) -> torch.Tensor:
    raw_bits = input.clone().to(torch.float).view(torch.int32)
    exp = ((raw_bits & 0x7f800000) >> 23) - 127
    fraction = (raw_bits & 0x7fffff) | 0x800000

    min_exp = math.floor(math.log2(fp8_min))
    nf_mask = 23 - mbits + torch.clamp(min_exp - exp, min=0)
    lb = (fraction & (1 << nf_mask)).bool()
    gb = (fraction & (1 << (nf_mask - 1))).bool()
    sb = (fraction & ((1 << (nf_mask - 1)) - 1)).bool()
    rb = (lb & gb) | (gb & sb)

    nf_mask_clamped = torch.clamp(nf_mask, max=23)
    raw_bits &= (-1 << nf_mask_clamped)
    raw_bits = torch.where(rb, raw_bits + (1 << nf_mask_clamped), raw_bits)

    output = raw_bits.view(torch.float)
    output = torch.clamp(output, min=-fp8_max, max=fp8_max)
    output = torch.where(torch.abs(input) <= fp8_min * (2 ** -(mbits + 1)), 0, output)

    output = torch.where(input == 0, 0, output)
    output = torch.where(torch.isfinite(input), output, torch.nan)
    return output.to(input.dtype)


def write_fp8_values():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_tensor = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)

    e4m3_values = quantize_to_fp8_e4m3(input_tensor)
    with open('fp8_e4m3.txt', 'w') as file:
        for v1, v2 in zip(input_tensor, e4m3_values):
            file.write(f"{v1.item()}\t{v2.item()}\n")

    e5m2_values = quantize_to_fp8_e5m2(input_tensor)
    with open('fp8_e5m2.txt', 'w') as file:
        for v1, v2 in zip(input_tensor, e5m2_values):
            file.write(f"{v1.item()}\t{v2.item()}\n")


# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------
# Never explicitly compute 2**(-exp) since subnorm numbers have
# exponents smaller than -126
def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


def _get_min_norm(ebits):
    """ Valid for all float formats """
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round='nearest',
                            saturate_normals=False, allow_denorm=True):
    """ Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    # handle Inf/NaN
    out[A == float("Inf")] = float("Inf")
    out[A == -float("Inf")] = -float("Inf")
    out[A == float("NaN")] = float("NaN")

    return out


def get_float_fq_fn(dtype):
    import re
    if (match := re.match(r"fp(\d+)_e(\d+)m(\d+)", dtype)):
        nbits, ebits, mbits = map(int, match.groups())
        assert nbits == ebits + mbits + 1
    else:
        raise ValueError("String does not match the required pattern")

    mbits = mbits + 2

    # Reclaim most of bit patterns used for special values if ebits < 4
    emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)

    if dtype != "fp8_e4m3":
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
    else:
        max_norm = 2**emax * 1.75  # e4m3 has custom max_norm
    print(ebits, mbits, emax, max_norm)

    return lambda A: _quantize_elemwise_core(A, mbits, ebits, max_norm, "even", True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="fp8_e4m3")
    args = parser.parse_args()

    fq_fn = get_float_fq_fn(args.dtype)
    input_tensor = torch.arange(2 ** 16, dtype=torch.int16).view(torch.bfloat16)
    quantized_tensor = fq_fn(input_tensor)
    with open('quantized_tensor.out', 'w') as file:
        for v1, v2 in zip(input_tensor, quantized_tensor):
            file.write(f"{v1.item()}\t{v2.item()}\n")
