import math
import torch

__all__ = ["quantize_to_posit"]

def quantize_to_posit(
    input: torch.Tensor,
    nbits: int = 8,
    es: int = 1,
    round_to_even: bool = True,
    return_pbits: bool = False,
) -> torch.Tensor:
    raw_bits = input.clone().to(torch.float).view(torch.int32)
    scale = ((raw_bits & 0x7f800000) >> 23) - 127
    fraction = raw_bits & 0x7fffff
    r = scale >= 0

    max_scale = (nbits - 2) * (1 << es)
    regime_dominated = torch.where(r, scale > max_scale, scale < -max_scale)

    run = torch.where(r, 1 + (scale >> es), -(scale >> es))
    regime = torch.where(r, (1 << (run + 1)) - 1, 0) ^ 1
    exponent = scale % (1 << es)
    pt_bits = (regime << (23 + es)) | (exponent << 23) | fraction

    # Check last, guard, and sticky bit for rounding
    len = 2 + run + es + 23
    lb_mask = 1 << (len - nbits)
    gb_mask = lb_mask >> 1
    sb_mask = gb_mask - 1

    lb = (pt_bits & lb_mask) != 0
    gb = (pt_bits & gb_mask) != 0
    sb = (pt_bits & sb_mask) != 0
    rb = ((lb & gb) | (gb & sb)) & ~regime_dominated

    # Truncate exponent bits
    ne_mask = torch.clamp(2 + run + es - nbits, min=0, max=es)
    scale &= (-1 << ne_mask)
    scale = torch.clamp(scale, min=-max_scale, max=max_scale)

    # Truncate fraction bits
    nf_mask = torch.clamp(len - nbits, min=0, max=23)
    fraction &= (-1 << nf_mask)

    output = ((scale + 127) << 23) | fraction
    output = torch.where(rb, output + (1 << (nf_mask + ne_mask)), output)
    output = output.view(torch.float) * torch.sign(input)

    if round_to_even:
        # |0|0000000|1|1|- only round up if exponent == 1
        threshold = math.pow(2, math.floor(-(nbits - 1) * (1 << es) + 2 ** (es - 1)))
        output = torch.where(input.abs() < threshold, 0, output)

    # Special values: 0 and NaN
    output = torch.where(input == 0, 0, output)
    output = torch.where(torch.isfinite(input), output, torch.nan)
    output = output.to(input.dtype)

    if return_pbits:
        pt_bits >>= len - nbits
        pt_bits &= (1 << (nbits - 1)) - 1 # Mask out any bits left of the sign bit including the sign bit
        pt_bits[rb] += 1
        pt_bits *= torch.sign(input).int()
        return output, pt_bits

    return output

def write_posit_values():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_tensor = torch.arange(2 ** 16, dtype=torch.int16, device=device).view(torch.bfloat16)
    posit_values = quantize_to_posit(input_tensor, 8, 1, round_to_even=True)
    with open('posit8_1.txt', 'w') as file:
        for v1, v2 in zip(input_tensor, posit_values):
            file.write(f"{v1.item()}\t{v2.item()}\n")

if __name__ == "__main__":
    write_posit_values()
