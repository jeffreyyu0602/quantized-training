import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase

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
}

PARAMS_TYPE = {
    'quant_max': float,
    'qscheme': QScheme,
    'amax_history_len': int,
    'ch_axis': int,
    'block_size': int,
}

def get_default_qmax(dtype):
    if dtype == "posit8_1":
        return 64

    if (match := re.fullmatch(r'int(\d+)', dtype)):
        nbits = int(match.group(1))
        return 2 ** (nbits - 1) - 1

    if (match := re.fullmatch(r"fp(\d+)_e(\d+)m(\d+)", dtype)):
        ebits = int(match.group(2))
        mbits = int(match.group(3)) + 2
        emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)
        if dtype == "fp8_e4m3":
            return 2**emax * 1.75
        return 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)

    return None

@dataclass
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, qscheme, quant_max etc.
    """

    dtype: str
    observer_or_fake_quant_ctr: Optional[_ObserverOrFakeQuantizeConstructor] = None
    quant_max: Optional[float] = None
    qscheme: Optional[QScheme] = None
    amax_history_len: Optional[int] = None
    ch_axis: Optional[int] = None
    block_size: Optional[int] = None
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

        if 'qscheme' in params:
            params.setdefault('quant_max', get_default_qmax(params['dtype']))
            params.setdefault('amax_history_len', 50)

        return QuantizationSpec(**params)

    def __post_init__(self):
        if self.quant_max is None and self.qscheme is not None:
            raise ValueError("quant_max is required for quantization.")

        if self.block_size is None and self.qscheme== QScheme.MICROSCALING:
            raise ValueError("block_size is required for microscaling.")
