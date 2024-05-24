from dataclasses import dataclass, asdict
from enum import Enum, IntEnum
from typing import Optional

from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor

__all__ = [
    "QuantizationSpec",
]


class RoundingMode(IntEnum):
    nearest = 0
    floor = 1
    even = 2

    @staticmethod
    def string_enums():
        return [s.name for s in list(RoundingMode)]


class QScheme(Enum):
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"
    PER_VECTOR_SYMMETRIC = "per_vector_symmetric"
    MICROSCALING = "microscaling"


DTYPE_TO_QUANT_MAX = {
    "int8": 127,
    "int4": 7,
    "posit8_1": 64,
    "fp8_e4m3": 448,
    "fp8_e5m2": 57344,
    "fp4_e2m1": 6,
}

ABBREV_MAP = {
    'dt': 'dtype',
    'qmax': 'quant_max',
    'qs': 'qscheme',
    'ahl': 'amax_history_len',
    'ax': 'ch_axis',
    'bs': 'block_size',
}

PARAMS_TYPE = {
    'dtype': str,
    'quant_max': float,
    'qscheme': QScheme,
    'amax_history_len': int,
    'ch_axis': int,
    'block_size': int,
}

@dataclass
class QuantizationSpec:
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, qscheme, quant_max etc.
    """

    dtype: str
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = None
    quant_max: Optional[float] = None
    qscheme: Optional[QScheme] = None
    amax_history_len: Optional[int] = None
    ch_axis: Optional[int] = None
    block_size: Optional[int] = None

    @staticmethod
    def from_str(s):
        assert(s != None), "String elem_format == None"
        s = s.lower()
        fields = s.split(',')

        params = {
            'dtype': fields[0],
            'quant_max': DTYPE_TO_QUANT_MAX.get(fields[0]),
            'amax_history_len': 50,
        }

        for item in fields[1:]:
            key, value = item.split('=')
            key = ABBREV_MAP.get(key, key)
            if key not in PARAMS_TYPE:
                raise ValueError(f"Unknown argument: {key}")
            params[key] = PARAMS_TYPE[key](value)

        return QuantizationSpec(**params)

    def __post_init__(self):
        if self.quant_max is None and self.qscheme is not None and self.qscheme != QScheme.MICROSCALING:
            raise ValueError("quant_max is required for quantization.")

        if self.ch_axis is None and self.qscheme in [
            QScheme.PER_CHANNEL_SYMMETRIC,
            QScheme.PER_VECTOR_SYMMETRIC,
            QScheme.MICROSCALING,
        ]:
            raise ValueError("Ch_axis is required for per-channel and microscaling qscheme.")

        if self.block_size is None and self.qscheme in [
            QScheme.PER_VECTOR_SYMMETRIC, QScheme.MICROSCALING
        ]:
            raise ValueError("Block_size is required for microscaling qscheme.")

    def to_dict(self):
        return asdict(self)
