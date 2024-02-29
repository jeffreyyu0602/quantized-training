from typing import Dict, Any, Callable

import torch.ao.nn.intrinsic as nni
import peft.tuners.lora as lora
from torch import nn

from transformers.activations import ACT2CLS
from transformers.models import bert, distilbert, gpt2, llama, mobilebert, roberta, whisper
from transformers.pytorch_utils import Conv1D

from .modules import qat, quantizable


def get_modules(module):
    return [getattr(nn, name) for name in module.__all__]

ACTFNS = [value[0] if isinstance(value, tuple) else value for value in ACT2CLS.values()]

QCONFIG_PROPAGATE_MODULE_CLASS_LIST = {
    'act': get_modules(nn.modules.activation) + ACTFNS,
    'batchnorm': get_modules(nn.modules.batchnorm),
    'gemm': [
        *get_modules(nn.modules.conv),
        nn.Linear,
        Conv1D,
        quantizable.MatmulFunctional,
    ],
    'norm': [
        *get_modules(nn.modules.normalization),
        llama.modeling_llama.LlamaRMSNorm,
        mobilebert.modeling_mobilebert.NoNorm,
    ],
    'pooling': get_modules(nn.modules.pooling),
    'residual': [quantizable.AddFunctional],
    'scaling': [quantizable.MulFunctional],
}

DEFAULT_QAT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Conv2d: qat.Conv2d,
    nn.Conv3d: qat.Conv3d,
    nn.Linear: qat.Linear,
    lora.Linear: qat.LoraLinear,
    # Intrinsic modules:
    nni.ConvBn2d: qat.ConvBn2d,
}

DEFAULT_CUSTOM_MODULE_MAPPINGS : Dict[Callable, Any] = {
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