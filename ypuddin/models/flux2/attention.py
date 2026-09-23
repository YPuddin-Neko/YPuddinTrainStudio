"""Model-local Klein processors for the DTK public FlashAttention API.

Projection/normalization/rotary ordering follows Diffusers 0.40's Flux2 processors
(Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team;
Apache-2.0). Only the attention call differs.
No Diffusers registry, package function or other model's processor is modified.
"""

import inspect

import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor,
    _get_qkv_projections,
    apply_rotary_emb,
)


def public_flash_function():
    from flash_attn import flash_attn_func

    if not callable(flash_attn_func) or "deterministic" not in inspect.signature(flash_attn_func).parameters:
        raise RuntimeError("DTK FlashAttention must provide flash_attn_func with deterministic backward")
    return flash_attn_func


def _check_processor(processor, attention_mask):
    if attention_mask is not None or processor._parallel_config is not None:
        raise ValueError("Klein DTK FlashAttention does not support attention masks or context parallelism")


def _flash(query, key, value):
    if query.device.type != "cuda" or not torch.version.hip:
        raise ValueError("Klein DTK FlashAttention requires a HIP GPU")
    if query.dtype not in (torch.float16, torch.bfloat16) or any(
        x.dtype != query.dtype or x.device != query.device for x in (key, value)
    ):
        raise ValueError("Klein DTK FlashAttention requires matching FP16/BF16 Q, K and V")
    # The public function owns autograd and the vendor's matching backward kernel.
    # DDP/FSDP shard training state; they do not require context-parallel wrappers.
    from flash_attn import flash_attn_func

    return flash_attn_func(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )


class DtkFlux2AttnProcessor(Flux2AttnProcessor):
    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        _check_processor(self, attention_mask)
        query, key, value, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query, key, value = (x.unflatten(-1, (-1, attn.head_dim)) for x in (query, key, value))
        query, key = attn.norm_q(query), attn.norm_k(key)
        if attn.added_kv_proj_dim is not None:
            eq, ek, ev = (x.unflatten(-1, (-1, attn.head_dim)) for x in (eq, ek, ev))
            query = torch.cat([attn.norm_added_q(eq), query], dim=1)
            key = torch.cat([attn.norm_added_k(ek), key], dim=1)
            value = torch.cat([ev, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        result = _flash(query, key, value).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_result, result = result.split_with_sizes(
                [encoder_hidden_states.shape[1], result.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_result = attn.to_add_out(encoder_result)
        result = attn.to_out[1](attn.to_out[0](result))
        return (result, encoder_result) if encoder_hidden_states is not None else result


class DtkFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):
    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        _check_processor(self, attention_mask)
        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv_size = (
            projected.shape[-1]
            * (3 * attn.inner_dim)
            // (3 * attn.inner_dim + attn.mlp_hidden_dim * attn.mlp_mult_factor)
        )
        qkv, mlp = projected.split([qkv_size, projected.shape[-1] - qkv_size], dim=-1)
        query, key, value = (x.unflatten(-1, (-1, attn.head_dim)) for x in qkv.chunk(3, dim=-1))
        query, key = attn.norm_q(query), attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        result = _flash(query, key, value).flatten(2, 3).to(query.dtype)
        return attn.to_out(torch.cat([result, attn.mlp_act_fn(mlp)], dim=-1))


_PROCESSORS = {
    Flux2AttnProcessor: DtkFlux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor: DtkFlux2ParallelSelfAttnProcessor,
}


def install_dtk_flash(model):
    public_flash_function()  # Fail before allocating weights or building caches.
    replacement = {}
    for name, processor in model.attn_processors.items():
        _check_processor(processor, None)
        cls = type(processor)
        if cls in _PROCESSORS.values():
            replacement[name] = processor
        elif cls in _PROCESSORS:
            replacement[name] = _PROCESSORS[cls]()
        else:
            raise ValueError(f"Unsupported Klein DTK FlashAttention processor: {cls.__name__}")
    if not replacement:
        raise ValueError("Klein DTK FlashAttention found no processors")
    model.set_attn_processor(replacement)


def restore_native_processors(model):
    reverse = {v: k for k, v in _PROCESSORS.items()}
    processors = getattr(model, "attn_processors", {})
    if any(type(p) in reverse for p in processors.values()):
        model.set_attn_processor(
            {name: reverse[type(p)]() if type(p) in reverse else p for name, p in processors.items()}
        )


def validate_dtk_flash(model):
    processors = model.attn_processors
    if not processors or any(type(p) not in _PROCESSORS.values() for p in processors.values()):
        raise ValueError("Klein DTK FlashAttention processors changed during training")
