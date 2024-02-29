import math
import torch

__all__ = [
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
]

def quantize_to_fp8_e4m3(
    input: torch.Tensor,
    mbits: int = 3,
    fp8_max: float = 448,
    fp8_min: float = 2 ** -6,
) -> torch.Tensor:
    raw_bits = input.float().view(torch.int32)
    exp = ((raw_bits & 0x7f800000) >> 23) - 127
    fraction = (raw_bits & 0x7fffff) | 0x800000

    min_exp = math.floor(math.log2(fp8_min))
    nf_mask = 23 - mbits + torch.clamp(min_exp - exp, min=0)
    lb = (fraction & (1 << nf_mask)).bool()
    gb = (fraction & (1 << (nf_mask - 1))).bool()
    sb = (fraction & ((1 << (nf_mask - 1)) - 1)).bool()
    rb = (lb & gb) | (gb & sb)

    nf_mask_clamped = torch.clamp(nf_mask, max=23)
    raw_bits = torch.where(rb, raw_bits + (1 << nf_mask_clamped), raw_bits)
    raw_bits &= (-1 << nf_mask_clamped)
    output = raw_bits.view(torch.float)

    output = torch.clamp(output, min=-fp8_max, max=fp8_max)
    output = torch.where(torch.abs(input) <= fp8_min * (2 ** -(mbits + 1)), 0, output)
    output = torch.where(torch.isfinite(input), output, torch.nan)

    return output.to(input.dtype)

def quantize_to_fp8_e5m2(
    input: torch.Tensor,
    mbits: int = 2,
    fp8_max: float = 57344,
    fp8_min: float = 2 ** -14,
) -> torch.Tensor:
    raw_bits = input.float().view(torch.int32)
    exp = ((raw_bits & 0x7f800000) >> 23) - 127
    fraction = (raw_bits & 0x7fffff) | 0x800000

    min_exp = math.floor(math.log2(fp8_min))
    nf_mask = 23 - mbits + torch.clamp(min_exp - exp, min=0)
    lb = (fraction & (1 << nf_mask)).bool()
    gb = (fraction & (1 << (nf_mask - 1))).bool()
    sb = (fraction & ((1 << (nf_mask - 1)) - 1)).bool()
    rb = (lb & gb) | (gb & sb)

    nf_mask_clamped = torch.clamp(nf_mask, max=23)
    raw_bits = torch.where(rb, raw_bits + (1 << nf_mask_clamped), raw_bits)
    raw_bits &= (-1 << nf_mask_clamped)
    output = raw_bits.view(torch.float)

    output = torch.clamp(output, min=-fp8_max, max=fp8_max)
    output = torch.where(torch.abs(input) <= fp8_min * (2 ** -(mbits + 1)), 0, output)
    output = torch.where(torch.isfinite(input), output, torch.nan)

    return output.to(input.dtype)

def main():
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

if __name__ == "__main__":
    main()