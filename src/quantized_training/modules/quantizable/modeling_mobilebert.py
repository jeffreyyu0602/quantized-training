# MIT License
#
# Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

import transformers.models.mobilebert.modeling_mobilebert as modeling_mobilebert
from transformers.models.mobilebert.modeling_mobilebert import NORM2FN

from .functional_modules import AddFunctional, MulFunctional, MatmulFunctional


class MobileBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.value = nn.Linear(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.qk_matmul = MatmulFunctional()
        self.av_matmul = MatmulFunctional()
        self.attn_scaling = MulFunctional()
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.qk_matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.attn_scaling(attention_scores, 1.0 / math.sqrt(self.attention_head_size))
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = self.av_matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @classmethod
    def from_observed(cls, other):
        assert hasattr(other, 'config'), "The float module must have 'config'"
        converted = cls(other.config)
        for name, child in other.named_children():
            converted._modules[name] = child
        return converted


class MobileBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.true_hidden_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, eps=config.layer_norm_eps)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.residual = AddFunctional()

    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        layer_outputs = self.dense(hidden_states)
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(self.residual(layer_outputs, residual_tensor))
        return layer_outputs

    @classmethod
    def from_observed(cls, other):
        assert hasattr(other, 'config'), "The float module must have 'config'"
        converted = cls(other.config)
        for name, child in other.named_children():
            converted._modules[name] = child
        return converted


class OutputBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.residual = AddFunctional()

    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(self.residual(layer_outputs, residual_tensor))
        return layer_outputs


class MobileBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = OutputBottleneck(config)
        self.residual = AddFunctional()

    def forward(
        self, intermediate_states: torch.Tensor, residual_tensor_1: torch.Tensor, residual_tensor_2: torch.Tensor
    ) -> torch.Tensor:
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        else:
            layer_output = self.LayerNorm(self.residual(layer_output, residual_tensor_1))
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        return layer_output
    
    @classmethod
    def from_observed(cls, other):
        assert hasattr(other, 'config'), "The float module must have 'config'"
        converted = cls(other.config)

        for name, module in other.named_children():
            if isinstance(module, modeling_mobilebert.OutputBottleneck):
                for submodule_name, submodule in module.named_children():
                    converted._modules[name]._modules[submodule_name] = submodule
            else:
                converted._modules[name] = module

        return converted


class FFNOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, eps=config.layer_norm_eps)
        self.residual = AddFunctional()

    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.LayerNorm(self.residual(layer_outputs, residual_tensor))
        return layer_outputs
    
    @classmethod
    def from_observed(cls, other):
        assert hasattr(other, 'config'), "The float module must have 'config'"
        converted = cls(other.config)
        for name, child in other.named_children():
            converted._modules[name] = child
        return converted