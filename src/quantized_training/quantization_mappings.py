from typing import Dict, Any, Callable

import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import peft.tuners.lora as lora

from transformers.activations import GELUActivation
from transformers.models import bert, distilbert, gpt2, llama, mobilebert, roberta, whisper
from transformers.pytorch_utils import Conv1D

import quantized_training.modules.qat as nnqat
from quantized_training.modules import quantizable


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

TRANSFORMER_MODULE_MAPPINGS: Dict[Callable, Any] = {
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


QCONFIG_PROPAGATE_MODULE_CLASS_LIST = {
    'activation': [
        nn.ReLU,
        nn.GELU,
        nn.Softmax,
        GELUActivation,
    ],
    'gemm': [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.Linear,
        Conv1D,
        quantizable.MatmulFunctional,
    ],
    'layernorm': [
        nn.LayerNorm,
        llama.modeling_llama.LlamaRMSNorm,
        mobilebert.modeling_mobilebert.NoNorm,
    ],
    'residual': [
        quantizable.AddFunctional,
    ],
    'scaling': [
        quantizable.MulFunctional,
    ],
}
