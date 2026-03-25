# coding=utf-8
# Qwen3 model definition for QTIP quantized inference.
# Based on model/llama.py, adapted for Qwen3 architecture differences:
# - q_norm / k_norm (RMSNorm on Q/K after projection)
# - Qwen3Config instead of LlamaConfig
# - No pretraining_tp
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, is_torchdynamo_compiling

from lib.linear.quantized_linear import QuantizedLinear

logger = logging.get_logger(__name__)

# Try to import Qwen3Config; fall back to creating it from LlamaConfig if not available
try:
    from transformers import Qwen3Config
except ImportError:
    from transformers.models.llama.configuration_llama import LlamaConfig as Qwen3Config


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length),
                                 fill_value=min_dtype,
                                 dtype=dtype,
                                 device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length,
                                    device=device) > cache_position.reshape(
                                        -1, 1)
        causal_mask = causal_mask[None,
                                  None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :
                                       mask_length] + attention_mask[:, None,
                                                                     None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :
                        mask_length] = causal_mask[:, :, :, :
                                                   mask_length].masked_fill(
                                                       padding_mask, min_dtype)
    return causal_mask


class Qwen3RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config=None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer("inv_freq",
                                 self.original_inv_freq,
                                 persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float()
                     @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class Qwen3MLP(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        td_x = config.quip_params['td_x']
        td_y = config.quip_params['td_y']
        L = config.quip_params['L']
        K = config.quip_params['K']
        V = config.quip_params['V']
        tlut_bits = config.quip_params['tlut_bits']
        decode_mode = config.quip_params['decode_mode']

        if config.quip_params.get('skip_list', None) is None:
            config.quip_params['skip_list'] = []

        if f'{layer_idx}_gate' not in config.quip_params['skip_list']:
            self.gate_proj = QuantizedLinear(self.hidden_size,
                                             self.intermediate_size,
                                             td_x, td_y, L, K, V,
                                             tlut_bits, decode_mode,
                                             dtype=config.torch_dtype,
                                             bias=False)
        else:
            self.gate_proj = nn.Linear(self.hidden_size,
                                       self.intermediate_size,
                                       dtype=config.torch_dtype,
                                       bias=False)

        if f'{layer_idx}_up' not in config.quip_params['skip_list']:
            self.up_proj = QuantizedLinear(self.hidden_size,
                                           self.intermediate_size,
                                           td_x, td_y, L, K, V,
                                           tlut_bits, decode_mode,
                                           dtype=config.torch_dtype,
                                           bias=False)
        else:
            self.up_proj = nn.Linear(self.hidden_size,
                                     self.intermediate_size,
                                     dtype=config.torch_dtype,
                                     bias=False)

        if f'{layer_idx}_down' not in config.quip_params['skip_list']:
            self.down_proj = QuantizedLinear(self.intermediate_size,
                                             self.hidden_size,
                                             td_x, td_y, L, K, V,
                                             tlut_bits, decode_mode,
                                             dtype=config.torch_dtype,
                                             bias=False)
        else:
            self.down_proj = nn.Linear(self.intermediate_size,
                                       self.hidden_size,
                                       dtype=config.torch_dtype,
                                       bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3Attention(nn.Module):
    """Qwen3 attention with optional q_norm/k_norm."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        td_x = config.quip_params['td_x']
        td_y = config.quip_params['td_y']
        L = config.quip_params['L']
        K = config.quip_params['K']
        K_v_proj = config.quip_params.get('K_v_proj', K)
        V = config.quip_params['V']
        tlut_bits = config.quip_params['tlut_bits']
        decode_mode = config.quip_params['decode_mode']

        attention_bias = getattr(config, 'attention_bias', False)

        if config.quip_params.get('skip_list', None) is None:
            config.quip_params['skip_list'] = []

        if f'{layer_idx}_q' not in config.quip_params['skip_list']:
            self.q_proj = QuantizedLinear(self.hidden_size,
                                          self.num_heads * self.head_dim,
                                          td_x, td_y, L, K, V,
                                          tlut_bits, decode_mode,
                                          dtype=config.torch_dtype,
                                          bias=attention_bias)
        else:
            self.q_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    dtype=config.torch_dtype,
                                    bias=attention_bias)

        if f'{layer_idx}_k' not in config.quip_params['skip_list']:
            self.k_proj = QuantizedLinear(self.hidden_size,
                                          self.num_key_value_heads * self.head_dim,
                                          td_x, td_y, L, K, V,
                                          tlut_bits, decode_mode,
                                          dtype=config.torch_dtype,
                                          bias=attention_bias)
        else:
            self.k_proj = nn.Linear(self.hidden_size,
                                    self.num_key_value_heads * self.head_dim,
                                    dtype=config.torch_dtype,
                                    bias=attention_bias)

        if f'{layer_idx}_v' not in config.quip_params['skip_list']:
            self.v_proj = QuantizedLinear(self.hidden_size,
                                          self.num_key_value_heads * self.head_dim,
                                          td_x, td_y, L, K_v_proj, V,
                                          tlut_bits, decode_mode,
                                          dtype=config.torch_dtype,
                                          bias=attention_bias)
        else:
            self.v_proj = nn.Linear(self.hidden_size,
                                    self.num_key_value_heads * self.head_dim,
                                    dtype=config.torch_dtype,
                                    bias=attention_bias)

        if f'{layer_idx}_o' not in config.quip_params['skip_list']:
            self.o_proj = QuantizedLinear(self.num_heads * self.head_dim,
                                          self.hidden_size,
                                          td_x, td_y, L, K, V,
                                          tlut_bits, decode_mode,
                                          dtype=config.torch_dtype,
                                          bias=attention_bias)
        else:
            self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                    self.hidden_size,
                                    dtype=config.torch_dtype,
                                    bias=attention_bias)

        # Qwen3-specific: q_norm and k_norm
        self.use_qk_norm = getattr(config, 'use_qk_norm', False)
        if self.use_qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        # Apply q_norm / k_norm if present
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Use SDPA
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states.to(query_states.device),
            value_states.to(query_states.device),
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Model(Qwen3PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache,
                                        List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens,
                                          past_seen_tokens +
                                          inputs_embeds.shape[1],
                                          device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                               cache_position, past_key_values,
                                               output_attentions)
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[
                    2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length(
        ) if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (attention_mask.shape[-1] if isinstance(
                attention_mask, torch.Tensor) else past_seen_tokens +
                             sequence_length + 1)

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype)

        return causal_mask


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache,
                                        List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            pass
        logits = self.lm_head(
            hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
                position_ids = position_ids.clone(
                    memory_format=torch.contiguous_format)

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {
                "input_ids":
                input_ids.clone(memory_format=torch.contiguous_format),
                "inputs_embeds": None
            }

        if isinstance(past_key_values,
                      StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs[
                    "inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update({
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        })
        return model_inputs
