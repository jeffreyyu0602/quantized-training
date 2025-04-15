import torch


def create_normal_map(offset=0.9677083, use_extra_value=True, k=4):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError("Scipy is required for `create_normal_map`.") from ie

    num_values = 2 ** (k - 1)
    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, num_values + 1)[:-1]).tolist()
        v2 = [0]  ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1])).tolist()
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1]).tolist()
        v2 = [0] * 2  ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1])).tolist()

    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 2 ** k

    return values


def quantize_to_nf(
    input: torch.Tensor,
    k: int = 4,
    use_extra_value=True,
    int_bits=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes input tensor to normal distribution values.

    Args:
        input: The input tensor to quantize
        k: Bit-width for quantization (2^k values)
        use_extra_value: Whether to use asymmetric quantization with an extra value
        int_bits: If specified, scales values to integers with this many bits

    Returns:
        tuple containing:
            - indices: Tensor of quantized indices
            - values: The corresponding quantization values/levels
    """
    values = create_normal_map(k=k, use_extra_value=use_extra_value)

    if int_bits is not None:
        scale_factor = 2**(int_bits - 1) - 1
        values = torch.round(values * scale_factor)

    values = values.to(device=input.device, dtype=input.dtype)
    input = torch.clamp(input, min=values.amin(), max=values.amax())
    indices = torch.argmin(torch.abs(values - input.unsqueeze(-1)), dim=-1)

    return indices, values
