from collections import namedtuple
from torch import nn
from .fake_quantize import FusedAmaxObsFakeQuantize

__all__ = [
    "QConfig",
    "get_qconfig",
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

def _create_fake_quant(qconfig, record_histogram):
    if qconfig is None:
        return nn.Identity

    return FusedAmaxObsFakeQuantize.with_args(
        **qconfig.to_dict(),
        record_histogram=record_histogram
    )

def get_qconfig(activation, weight, error, record_histogram=False,
                force_scale_power_of_two=False):
    return QConfig(
        activation=_create_fake_quant(activation, record_histogram),
        weight=_create_fake_quant(weight, record_histogram),
        error=_create_fake_quant(error, record_histogram)
    )

    import torch
    from torch.ao.quantization import (
        MovingAverageMinMaxObserver,
        FusedMovingAvgObsFakeQuantize,
    )
    act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )
    return QConfig(
        activation=act_fake_quant,
        weight=_create_fake_quant(weight, record_histogram),
        error=_create_fake_quant(error, record_histogram)
    )