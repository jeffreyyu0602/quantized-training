import logging
from typing import List, Union, Optional
import math

import torch
from torch import nn
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from transformers import GenerationConfig, PreTrainedModel
from transformers.cache_utils import Cache, StaticCache
from transformers.utils import is_torch_greater_or_equal
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv


__all__ = [
    "TorchExportableModuleWithStaticCache",
    "convert_and_export_with_split_cache",
    "generate",
    "swap_llama_attention",
]

logger = logging.getLogger(__name__)

use_llama_attention_kivi = False


def process_logits(scores: torch.Tensor, eos_token_id: torch.Tensor) -> torch.Tensor:
    vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
    eos_token_mask = torch.isin(vocab_tensor, eos_token_id)
    scores_processed = scores.clone()
    scores_processed = torch.where(eos_token_mask, -math.inf, scores)
    return scores_processed


def generate(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: Union[int, List[int]] = None,
    model_decode: torch.nn.Module = None,
):
    device = model.device
    generation_config = model.generation_config

    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            eos_token_id = {eos_token_id}
        else:
            eos_token_id = set(eos_token_id)
    elif generation_config.eos_token_id is not None:
        eos_token_id = {generation_config.eos_token_id}
    else:
        eos_token_id = set()

    # Initial forward pass to get logits and prefill KV cache
    with torch.no_grad():
        outputs = model(prompt_token_ids)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

    # pre-process distribution
    logits = process_logits(logits, torch.tensor(list(eos_token_id), device=device))
    # print("Prefill logits:", logits)

    current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    response_tokens = prompt_token_ids[0].tolist() + [current_token.item()]

    seq_length = past_key_values.get_seq_length()

    # Copy KV caches to the exported decode model
    if model_decode is not None:
        for i, layer in enumerate(past_key_values.layers):
            model_decode.get_buffer(f"key_cache_{i}")[:, :, : seq_length, :] = layer.keys
            model_decode.get_buffer(f"value_cache_{i}")[:, :, : seq_length, :] = layer.values

    device = model_decode.device if model_decode is not None else device

    # Generate tokens iteratively
    for step in range(1, max_new_tokens):
        with torch.no_grad():
            if model_decode is not None:
                logits = model_decode(
                    input_ids=current_token.to(device),
                    cache_position=torch.tensor([len(response_tokens) - 1], dtype=torch.long, device=device),
                )
            else:
                outputs = model(input_ids=current_token, past_key_values=past_key_values)
                logits = outputs.logits
                past_key_values = outputs.past_key_values

            # print(f"Step {step} logits:", logits)

        if len(response_tokens) < min_length - 1:
            logits = process_logits(logits, torch.tensor(list(eos_token_id), device=device))

        current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        response_tokens.append(current_token.item())

        # Check stopping criteria
        if len(response_tokens) >= min_length and current_token.item() in eos_token_id:
            break

    return torch.tensor([response_tokens], dtype=torch.long, device=device)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    key_residual: torch.Tensor = None,
    value_residual: torch.Tensor = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if key_residual is not None:
        key_states_residual = repeat_kv(key_residual, module.num_key_value_groups)
        attn_weights_residual = torch.matmul(query, key_states_residual.transpose(2, 3)) * scaling
        attn_weights = torch.cat([attn_weights, attn_weights_residual], dim=-1)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : attn_weights.shape[-1]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if value_residual is not None:
        value_states_residual = repeat_kv(value_residual, module.num_key_value_groups)
        attn_output = torch.matmul(attn_weights[:, :, :, : value_states.shape[-2]], value_states)
        attn_output_residual = torch.matmul(attn_weights[:, :, :, value_states.shape[-2]:], value_states_residual)
        attn_output = attn_output + attn_output_residual
    else:
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttentionKIVI(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_values_residual = kwargs.get("past_key_values_residual", None)

        if past_key_values_residual is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position_residual")}
            key_states_residual, value_states_residual = past_key_values_residual.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                past_key_values.layers[self.layer_idx].keys,
                past_key_values.layers[self.layer_idx].values,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                key_residual=key_states_residual,
                value_residual=value_states_residual,
                **kwargs,
            )
        else:
            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    

def swap_llama_attention(model: PreTrainedModel) -> PreTrainedModel:
    """
    Swap the attention module in a LLaMA model with the custom LlamaAttention module.

    Args:
        model (`PreTrainedModel`): The pretrained LLaMA model to modify.
    Returns:
        `PreTrainedModel`: The modified model with the custom attention module.
    """
    global use_llama_attention_kivi
    use_llama_attention_kivi = True

    logger.info("Using custom LlamaAttention module.")

    def swap_module(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LlamaAttention):
                device = assert_and_get_unique_device(child)
                dtype = next(child.parameters()).dtype
                new_attn = LlamaAttentionKIVI(child.config, child.layer_idx).to(device=device, dtype=dtype)
                new_attn.load_state_dict(child.state_dict(), strict=True)
                setattr(module, name, new_attn)
                logger.info(f"Replaced {full_name} with LlamaAttentionKIVI")
            else:
                swap_module(child, full_name)

        return module

    return swap_module(model)


def create_causal_mask_residual(
    target_length: int,
    prefill_length: int,
    max_length: int,
    cache_position: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((1, target_length), fill_value=min_dtype, dtype=dtype)
    position = torch.arange(target_length)
    causal_mask *= (
        ((position > prefill_length) & (position < max_length)) | (position > max_length + cache_position)
    ).reshape(1, -1)
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)
    return causal_mask


class TorchExportableModuleWithStaticCache(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM to `StaticCache`. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.

    Note:
        This class is specifically designed to support export process using `torch.export`
        in a way that ensures the model can be further lowered and run efficiently in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        prefill_length: int,
        max_new_tokens: int,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes the wrapper module with the pretrained model.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap. The model must have caching
                enabled and use a 'static' caching implementation.
            batch_size (`Optional[int]`): The batch size of the model. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we raise a ValueError.
            max_cache_len (`Optional[int]`): The maximum cache length for generation. Same mechanism as `batch_size` if
                not provided.
            device (`Optional[torch.device]`): The device to use. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we use `model.device` (no error is raised).

        Raises:
            AssertionError: If the pretrained model does not have caching enabled or if it does
            not use a 'static' caching implementation in `model.generation_config`.
            ValueError: If `batch_size` or `max_cache_len` is not provided, either as an argument or in `cache_config`.
        """
        super().__init__()

        config = model.config.get_text_config()
        generation_config = model.generation_config

        # Sanity checks
        if generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config` in `model`."
            )
        if not generation_config.use_cache:
            raise AssertionError(
                "The model must have caching enabled to be exported with static caching. "
                "Please set `generation_config.use_cache=True`."
            )
        if generation_config.cache_implementation != "static":
            raise AssertionError(
                "The model must use a 'static' caching implementation to be exported with static caching. "
                "Please set `generation_config.cache_implementation='static'`."
            )

        cache_config = {} if generation_config.cache_config is None else generation_config.cache_config

        # Ensure batch_size and max_cache_len are set
        if batch_size is None:
            batch_size = cache_config.get("batch_size", None)
            if batch_size is None:
                raise ValueError("batch_size must be provided, either as an argument or in cache_config.")
        if max_cache_len is None:
            max_cache_len = cache_config.get("max_cache_len", None)
            if max_cache_len is None:
                raise ValueError("max_cache_len must be provided, either as an argument or in cache_config.")
        # Infer device if not provided
        if device is None:
            device = cache_config.get("device", model.device)

        self.max_cache_len = max_cache_len

        self.model = model
        self.static_cache = StaticCache(max_cache_len=prefill_length, config=config)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        dtype = self.model.dtype
        # We need this call to initialize all the layers (otherwise it's done lazily, which is not exportable)
        self.static_cache.early_initialization(batch_size, num_heads, head_dim, dtype, device)
        for i in range(len(self.static_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.layers[i].keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.layers[i].values, persistent=False)

        self.static_cache_residual = StaticCache(max_cache_len=max_new_tokens, config=config)
        self.static_cache_residual.early_initialization(batch_size, num_heads, head_dim, dtype, device)
        for i in range(len(self.static_cache_residual)):
            self.register_buffer(f"key_cache_residual_{i}", self.static_cache_residual.layers[i].keys, persistent=False)
            self.register_buffer(f"value_cache_residual_{i}", self.static_cache_residual.layers[i].values, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        cache_position_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            inputs_embeds (`torch.Tensor`): Tensor representing current input embeddings to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.

        This forward adapter serves two primary purposes:

        1. **Making the Model `torch.export`-Compatible**:
            The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
            enabling the model to be exportable using `torch.export` without encountering issues.

        2. **Ensuring Compatibility with `ExecuTorch` runtime**:
            The adapter matches the model's forward signature with that in `executorch/extension/llm/runner`,
            ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.
        """
        past_key_values = self.static_cache

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=1,
            target_length=self.max_cache_len,
            dtype=self.model.dtype,
            cache_position=cache_position,
            batch_size=input_ids.shape[0],
        )

        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=True,
            past_key_values_residual=self.static_cache_residual,
            cache_position_residual=cache_position_residual,
        )
        if hasattr(outs, "logits"):
            # Returned outputs is `CausalLMOutputWithPast`
            return outs.logits
        else:
            # Returned the `last_hidden_state` from `BaseModelOutputWithPast`
            return outs.last_hidden_state
        
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    @staticmethod
    def generate(
        model: torch.nn.Module,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
        min_length: int = 0,
        eos_token_id: Union[int, List[int]] = None,
        model_decode: torch.nn.Module = None,
        key_quantizer=None,
        value_quantizer=None,
    ):
        device = model.device
        generation_config = model.generation_config

        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = {eos_token_id}
            else:
                eos_token_id = set(eos_token_id)
        elif generation_config.eos_token_id is not None:
            eos_token_id = {generation_config.eos_token_id}
        else:
            eos_token_id = set()

        # Initial forward pass to get logits and prefill KV cache
        with torch.no_grad():
            outputs = model(prompt_token_ids)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        # pre-process distribution
        logits = process_logits(logits, torch.tensor(list(eos_token_id), device=device))
        # print("Prefill logits:", logits)

        current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        response_tokens = prompt_token_ids[0].tolist() + [current_token.item()]

        seq_length = past_key_values.get_seq_length()
        # print(f"Prompt length: {seq_length}")

        def get_quantized_value_and_scale(input, activation_post_process):
            activation_post_process(input)
            input = torch.ops.quantized_ops.quantize(
                input,
                activation_post_process.code,
                activation_post_process.scale,
                block_size=activation_post_process.block_size,
            )
            return input, activation_post_process.scale

        for i, layer in enumerate(past_key_values.layers):
            cache_len = model_decode.get_buffer(f"value_cache_{i}").shape[2]
            assert seq_length <= cache_len, f"seq_length {seq_length} exceeds cache size {cache_len}"

            # transpose key cache if needed
            key_cache_len = model_decode.get_buffer(f"key_cache_{i}").shape[2]
            if key_cache_len != cache_len:
                key_state = layer.keys.transpose(2, 3)
                slices = (slice(None), slice(None), slice(None), slice(0, key_state.shape[3]))
            else:
                key_state = layer.keys
                slices = (slice(None), slice(None), slice(0, key_state.shape[2]), slice(None))

            if key_quantizer is not None:
                for quantizer in key_quantizer[:-1]:
                    key_state = quantizer(key_state)

                if f"key_cache_{i}_scale" in model_decode._buffers:
                    key_state, key_scale = get_quantized_value_and_scale(key_state, key_quantizer[-1])
                    model_decode.get_buffer(f"key_cache_{i}_{key_quantizer[-1].dtype}")[slices] = key_state

                    slices = (slice(None), slice(None), slice(0, key_scale.shape[2]), slice(0, key_scale.shape[3]))
                    model_decode.get_buffer(f"key_cache_{i}_scale")[slices] = key_scale
                else:
                    key_state = key_quantizer[-1](key_state)
                    model_decode.get_buffer(f"key_cache_{i}")[slices] = key_state
            else:
                model_decode.get_buffer(f"key_cache_{i}")[slices] = key_state

            if value_quantizer is not None:
                value_state = layer.values
                for quantizer in value_quantizer[:-1]:
                    value_state = quantizer(value_state)

                if f"value_cache_{i}_scale" in model_decode._buffers:
                    value_state, value_scale = get_quantized_value_and_scale(value_state, value_quantizer[-1])
                    model_decode.get_buffer(f"value_cache_{i}_{value_quantizer[-1].dtype}")[:, :, : value_state.shape[2], :] = value_state
                    model_decode.get_buffer(f"value_cache_{i}_scale")[:, :, : value_scale.shape[2], :] = value_scale
                else:
                    value_state = value_quantizer[-1](value_state)
                    model_decode.get_buffer(f"value_cache_{i}")[:, :, : value_state.shape[2], :] = value_state
            else:
                model_decode.get_buffer(f"value_cache_{i}")[:, :, : layer.values.shape[2], :] = layer.values

        device = model_decode.device

        for step in range(1, max_new_tokens):
            with torch.no_grad():
                cache_len = model_decode.get_buffer("value_cache_0").shape[2]
                residual_len = model_decode.get_buffer("value_cache_residual_0").shape[2]

                causal_mask = create_causal_mask_residual(
                    target_length=cache_len + residual_len,
                    prefill_length=seq_length,
                    max_length=cache_len,
                    cache_position=step - 1,
                    dtype=next(model_decode.parameters()).dtype,
                )

                try:
                    logits = model_decode(
                        input_ids=current_token.to(device),
                        cache_position=torch.tensor([len(response_tokens) - 1], dtype=torch.long, device=device),
                        cache_position_residual=torch.tensor([step - 1], dtype=torch.long, device=device),
                        attention_mask=causal_mask.to(device),
                    )
                except:
                    from quantized_training import ShapeProp
                    ShapeProp(model_decode).propagate(
                        current_token.to(device),
                        torch.tensor([len(response_tokens) - 1], dtype=torch.long, device=device),
                        torch.tensor([step - 1], dtype=torch.long, device=device),
                        causal_mask.to(device),
                    )

                # print(f"Step {step} logits:", logits)

            if len(response_tokens) < min_length - 1:
                logits = process_logits(logits, torch.tensor(list(eos_token_id), device=device))

            current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            response_tokens.append(current_token.item())

            if len(response_tokens) >= min_length and current_token.item() in eos_token_id:
                break

        return torch.tensor([response_tokens], dtype=torch.long, device=device)


def convert_and_export_with_split_cache(
    model: PreTrainedModel,
    max_len: int = 4096,
    max_new_tokens: int = 512,
    example_input_ids: Optional[torch.Tensor] = None,
    example_cache_position: Optional[torch.Tensor] = None,
    example_cache_position_residual: Optional[torch.Tensor] = None,
    example_attention_mask: Optional[torch.Tensor] = None,
    dynamic_shapes: Optional[dict] = None,
    strict: Optional[bool] = None,
):
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`,
    ensuring the exported model is compatible with `ExecuTorch`.

    Args:
        model (`PreTrainedModel`): The pretrained model to be exported.
        example_input_ids (`Optional[torch.Tensor]`): Example input token id used by `torch.export`.
        example_cache_position (`Optional[torch.Tensor]`): Example current cache position used by `torch.export`.
        dynamic_shapes(`Optional[dict]`): Dynamic shapes used by `torch.export`.
        strict(`Optional[bool]`): Flag to instruct `torch.export` to use `torchdynamo`.

    Returns:
        Exported program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
    """
    if not is_torch_greater_or_equal("2.6", accept_dev=True):
        raise ImportError("torch >= 2.6 is required.")
    
    max_cache_len = max_len + max_new_tokens

    config_dict = model.generation_config.to_dict()

    config_dict.update({
        "use_cache": True,
        "cache_implementation": "static",
        "cache_config": {
            "batch_size": 1,
            "max_cache_len": max_cache_len,
            "device": str(model.device),
        }
    })

    model.generation_config = GenerationConfig(**config_dict)

    with torch.no_grad():
        # TODO: The default inputs only work for text models. We need to add support for vision/audio models.
        example_input_ids = (
            example_input_ids
            if example_input_ids is not None
            else torch.tensor([[1]], dtype=torch.long, device=model.device)
        )
        example_cache_position = (
            example_cache_position
            if example_cache_position is not None
            else torch.tensor([0], dtype=torch.long, device=model.device)
        )
        example_cache_position_residual = (
            example_cache_position_residual
            if example_cache_position_residual is not None
            else torch.tensor([0], dtype=torch.long, device=model.device)
        )
        example_attention_mask = (
            example_attention_mask
            if example_attention_mask is not None
            else torch.ones((1, max_cache_len), dtype=model.dtype, device=model.device)[None, None, :, :]
        )

        exported_program = torch.export.export(
            TorchExportableModuleWithStaticCache(model, max_len, max_new_tokens),
            args=(),
            kwargs={
                "input_ids": example_input_ids,
                "cache_position": example_cache_position,
                "cache_position_residual": example_cache_position_residual,
                "attention_mask": example_attention_mask,
            },
            dynamic_shapes=dynamic_shapes,
            strict=strict if strict is not None else True,
        )

        return exported_program
