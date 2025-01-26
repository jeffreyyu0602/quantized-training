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
):
    values = create_normal_map(k=k, use_extra_value=use_extra_value)
    if int_bits is not None:
        values = torch.round(values * (2**int_bits - 1))
    values = values.to(input.device, dtype=input.dtype)

    indices = torch.argmin(torch.abs(values - input.unsqueeze(-1)), dim=-1)
    indices = torch.where(input < values.amin(), 0, indices)
    indices = torch.where(input > values.amax(), 2 ** k - 1, indices)

    return indices, values
