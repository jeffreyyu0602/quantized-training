import re
from collections import namedtuple

import torch.nn as nn

from .fake_quantize import FusedAmaxObsFakeQuantize

__all__ = [
    "QConfig",
    "get_default_qconfig",
]

class QConfig(namedtuple('QConfig', ['activation', 'weight', 'error'])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8))

    """
    def __new__(cls, activation, weight, error):
        return super().__new__(cls, activation, weight, error)

def parse_dtype(dtype: str):
    if "," in dtype:
        dtypes = dtype.split(",")
        assert len(dtypes) == 2, f"Invalid data types: {dtype}."
        dtype_fwd, dtype_bwd = dtypes
    elif re.search(r'^FP8(\.MIXED)?$', dtype, re.IGNORECASE):
        dtype_fwd, dtype_bwd = ("E4M3", "E5M2")
    else:
        dtype_fwd, dtype_bwd = (dtype, dtype)
    return (dtype_fwd, dtype_bwd)

def get_default_qconfig(
    dtype: str,
    activation: bool = False,
    weight: bool = False,
    error: bool = False,
    scaling_fwd: bool = False,
    scaling_bwd: bool = False,
    max_fwd: float = 64.0,
    max_bwd: float = 64.0,
    amax_history_len: int = 10
) -> QConfig:
    dtype_fwd, dtype_bwd = parse_dtype(dtype)

    default_fake_quant = FusedAmaxObsFakeQuantize.with_args(
        dtype=dtype_fwd,
        quant_max=max_fwd,
        amax_history_len=amax_history_len,
        quantize_per_tensor=scaling_fwd
    )

    error_fake_quant = FusedAmaxObsFakeQuantize.with_args(
        dtype=dtype_bwd,
        quant_max=max_bwd,
        amax_history_len=amax_history_len,
        quantize_per_tensor=scaling_bwd
    )

    return QConfig(
        activation=default_fake_quant if activation else nn.Identity,
        weight=default_fake_quant if weight else nn.Identity,
        error=error_fake_quant if error else nn.Identity,
    )