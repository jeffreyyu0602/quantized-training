import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Union, Tuple

from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase
from torch.fx import Node

from quantized_training.fake_quantize import FusedAmaxObsFakeQuantize

__all__ = [
    "QuantizationSpec",
]

class QScheme(Enum):
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"
    MICROSCALING = "microscaling"

ABBREV_MAP = {
    'qmax': 'quant_max',
    'qs': 'qscheme',
    'ahl': 'amax_history_len',
    'ax': 'ch_axis',
    'bs': 'block_size',
    'scale': 'scale_dtype',
    'outlier': 'outlier_threshold',
}

PARAMS_TYPE = {
    'quant_max': float,
    'qscheme': QScheme,
    'amax_history_len': int,
    'ch_axis': int,
    'block_size': int,
    'scale_dtype': str,
    'outlier_threshold': float,
}

def get_max_val(dtype):
    if (match := re.fullmatch(r'int(\d+)', dtype, re.IGNORECASE)):
        nbits = int(match.group(1))
        return 2 ** (nbits - 1) - 1

    if (match := re.fullmatch(r"fp(\d+)_e(\d+)m(\d+)", dtype, re.IGNORECASE)):
        ebits = int(match.group(2))
        mbits = int(match.group(3)) + 2
        emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)
        if dtype == "fp8_e4m3":
            return 2**emax * 1.75
        return 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)

    if (match := re.fullmatch(r'posit(\d+)_(\d+)', dtype)):
        nbits = int(match.group(1))
        es = int(match.group(2))
        if nbits == 8 and es == 1:
            return 64
        return (2 ** (2 ** es)) ** (nbits - 2)

    if (match := re.fullmatch(r'nf(\d+)(?:_(\d+))?', dtype)):
        if match.group(2) is not None:
            return 2 ** (int(match.group(2)) - 1) - 1
        return 1

    return None

@dataclass(eq=True)
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, qscheme, quant_max etc.
    """

    dtype: str
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = FusedAmaxObsFakeQuantize
    quant_max: Optional[float] = None
    qscheme: Optional[QScheme] = None
    amax_history_len: Optional[int] = None
    ch_axis: Optional[int] = None
    block_size: Optional[int] = None
    scale_dtype: Optional[str] = None
    outlier_threshold: Optional[float] = None
    is_dynamic: bool = False  # required by sharing nodes

    @staticmethod
    def from_str(s):
        assert(s != None), "String quantization_spec == None"
        s = s.lower()
        fields = s.split(',')

        params = {'dtype': fields[0]}
        for item in fields[1:]:
            key, value = item.split('=')
            key = ABBREV_MAP.get(key, key)
            if key not in PARAMS_TYPE:
                raise ValueError(f"Unknown argument: {key}")
            params[key] = PARAMS_TYPE[key](value)

        if (qscheme := params.get('qscheme', None)) is not None:
            params.setdefault('quant_max', float(get_max_val(params['dtype'])))
            if qscheme != QScheme.MICROSCALING:
                params.setdefault('amax_history_len', 16)

        return QuantizationSpec(**params)

    def __post_init__(self):
        if self.qscheme is not None and self.quant_max is None:
            raise ValueError("quant_max is required for quantization.")

        if self.qscheme== QScheme.MICROSCALING and self.block_size is None:
            raise ValueError("block_size is required for microscaling.")

EdgeOrNode = Union[Tuple[Node, Node], Node]

@dataclass(eq=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    """ quantization spec for the Tensors whose quantization parameters are derived from other Tensors
    """
    derived_from: List[EdgeOrNode]
    derive_qparams_fn: Callable[[List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]]
    dtype: str
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[QScheme] = None
