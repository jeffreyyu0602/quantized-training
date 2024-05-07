from typing import Dict, Any, Callable

import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import peft.tuners.lora as lora

from transformers.activations import ACT2CLS
from transformers.models import bert, distilbert, gpt2, llama, mobilebert, roberta, whisper
from transformers.pytorch_utils import Conv1D

from .modules import qat as nnqat
from .modules import quantizable


DEFAULT_QAT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    nn.Conv2d: nnqat.Conv2d,
    nn.Conv3d: nnqat.Conv3d,
    nn.Linear: nnqat.Linear,
    lora.Linear: nnqat.LoraLinear,
    # Intrinsic modules:
    nni.ConvBn1d: nnqat.ConvBn1d,
    nni.ConvBn2d: nnqat.ConvBn2d,
    nni.ConvBn3d: nnqat.ConvBn3d,
}

HF_TRANSFORMER_MODULE_MAPPINGS: Dict[Callable, Any] = {
    bert.modeling_bert.BertSelfAttention: quantizable.BertSelfAttention,
    bert.modeling_bert.BertSelfOutput: quantizable.BertSelfOutput,
    bert.modeling_bert.BertOutput: quantizable.BertOutput,
    distilbert.modeling_distilbert.TransformerBlock: quantizable.TransformerBlock,
    gpt2.modeling_gpt2.GPT2Block: quantizable.GPT2Block,
    llama.modeling_llama.LlamaDecoderLayer: quantizable.LlamaDecoderLayer,
    mobilebert.modeling_mobilebert.MobileBertSelfAttention: quantizable.MobileBertSelfAttention,
    mobilebert.modeling_mobilebert.MobileBertSelfOutput: quantizable.MobileBertSelfOutput,
    mobilebert.modeling_mobilebert.FFNOutput: quantizable.FFNOutput,
    mobilebert.modeling_mobilebert.MobileBertOutput: quantizable.MobileBertOutput,
    roberta.modeling_roberta.RobertaSelfAttention: quantizable.BertSelfAttention,
    roberta.modeling_roberta.RobertaSelfOutput: quantizable.BertSelfOutput,
    roberta.modeling_roberta.RobertaOutput: quantizable.BertOutput,
    whisper.modeling_whisper.WhisperEncoderLayer: quantizable.WhisperEncoderLayer,
    whisper.modeling_whisper.WhisperDecoderLayer: quantizable.WhisperDecoderLayer,
}


def get_modules(module):
    return [getattr(nn, name) for name in module.__all__]


ACTFNS = [
    value[0] if isinstance(value, tuple) else value
    for value in ACT2CLS.values()
]

QCONFIG_PROPAGATE_MODULE_CLASS_LIST = {
    'activation': get_modules(nn.modules.activation) + ACTFNS,
    'batchnorm': get_modules(nn.modules.batchnorm),
    'gemm': get_modules(nn.modules.conv) + [
        nn.Linear,
        Conv1D,
        quantizable.MatmulFunctional,
    ],
    'layernorm': [
        nn.LayerNorm,
        llama.modeling_llama.LlamaRMSNorm,
        mobilebert.modeling_mobilebert.NoNorm,
    ],
    'pooling': get_modules(nn.modules.pooling),
    'residual': [quantizable.AddFunctional],
    'scaling': [quantizable.MulFunctional],
}

aten = torch.ops.aten
QUANTIZATION_OPERATORS = {
    "activation": [
        (aten._softmax.default, (0,)),
        (aten.gelu.default, (0,)),
        (aten.relu.default, (0,)),
        (aten.sigmoid.default, (0,)),
    ],
    "batchnorm": [
        (aten._native_batch_norm_legit.default, (0,)),
        (aten._native_batch_norm_legit_no_training.default, (0,)),
    ],
    # "gemm": [
    #     (aten.convolution.default, (0,)),
    #     (aten.addmm.default, (1,)),
    #     aten.bmm.default,
    #     (aten.mm.default, (0,)),
    # ],
    "gemm": [
        (aten.conv2d.default, (0,)),
        (aten.linear.default, (0,)),
    ],
    "layernorm": [
        (aten.native_layer_norm.default, (0,))
    ],
    "residual": [aten.add.Tensor],
    "scaling": [aten.div.Tensor],
}
