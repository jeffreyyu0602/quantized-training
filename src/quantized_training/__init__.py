from .fake_quantize import *
from .fp8 import *
from .posit import *
from .qconfig import *
from .quantize import *
from .quantize_pt2e import *
from .quantizer import *
from .training_args import *
from .utils import *
from .histogram import *

__all__ = [
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QuantizationSpec",
    "add_training_args",
    "convert",
    "get_qconfig",
    "get_quantized_model",
    "get_quantizer"
    "plot_layer_distribution",
    "plot_layer_range",
    "prepare",
    "prepare_pt2e",
    "propagate_config",
    "quantize",
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "quantize_to_posit",
    "replace_softmax",
    "run_task",
]

class qscheme: ...

# Defined in quantized_training/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
per_vector_symmetric: qscheme = QScheme.PER_VECTOR_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING
