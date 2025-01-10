import torch

from .fp8 import quantize_to_fp8_e4m3


def create_normal_map(offset=0.9677083, use_extra_value=True, k=4):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError(
            "Scipy is required for `create_normal_map`. Install `bitsandbytes` with the `[test]` extra.",
        ) from ie

    num_values = 2 ** (k - 1)
    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, num_values + 1)[:-1]).tolist()
        v2 = [0]
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1])).tolist()
    else:
        # v1 = norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1]).tolist()
        # v2 = [0] * 2
        # v3 = (-norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1])).tolist()
        cdfs = torch.linspace(offset, 0.5, num_values)
        cdfs[-1] = (cdfs[-2] + 0.5) / 2
        v1 = norm.ppf(cdfs).tolist()
        v2 = [0]
        v3 = (-norm.ppf(cdfs[:-1])).tolist()

    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 2 ** k

    return values


def quantize_to_nf(
    input: torch.Tensor,
    k: int = 4,
    use_extra_value=False,
    use_fp8_values=False,
):
    values = create_normal_map(k=k, use_extra_value=use_extra_value)
    values = values.to(dtype=torch.bfloat16, device=input.device)
    if use_fp8_values:
        values = quantize_to_fp8_e4m3(values)
    indices = torch.argmin(torch.abs(values - input.unsqueeze(-1)), dim=-1)
    return values[indices]
