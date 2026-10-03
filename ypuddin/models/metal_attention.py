"""Model-local Metal Flash attention for the mtlattn FP32 contract.

Only eligible backbone attention calls enter the portable Metal kernels. Native
SDPA retains unsupported semantics; an error from a selected kernel is never
hidden by retrying another implementation.
"""

import platform
import re
import sys
from copy import copy
from functools import lru_cache
from importlib import import_module, metadata

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode

from ypuddin.models import attention_check
from ypuddin.runtime_profiles import current_profile

METAL_FLASH_IMPLEMENTATION_ID = "mtlattn-fp32-v1"


def metal_flash_runtime_identity() -> dict[str, str]:
    """Validate the installed stack without loading model weights or GPU tensors."""
    system, machine = platform.system(), platform.machine().lower()
    macos = platform.mac_ver()[0]
    if system != "Darwin" or machine not in {"arm64", "aarch64"}:
        raise ValueError("Metal FlashAttention requires an Apple Silicon Mac")
    if not macos or int(macos.split(".")[0]) < 15:
        raise ValueError("Metal FlashAttention requires macOS 15 or newer")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise ValueError("Metal FlashAttention requires CPython 3.11 or 3.12")
    if current_profile() not in {"legacy", "macos-mps"}:
        raise ValueError(
            "Metal FlashAttention requires the Apple MPS deployment; use the Apple MPS launcher"
        )
    if not re.match(r"^2\.13\.", str(torch.__version__)):
        raise ValueError("Metal FlashAttention requires PyTorch 2.13.x with the matching mtlattn 0.4.1 build")
    if getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None):
        raise ValueError("Metal FlashAttention requires the Apple MPS PyTorch build, not CUDA or HIP")
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        raise ValueError("Metal FlashAttention requires an available Apple MPS device")
    try:
        version = metadata.version("mtlattn")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Metal FlashAttention requires mtlattn 0.4.1; install its matching build in Environment settings"
        ) from exc
    if version != "0.4.1":
        raise ValueError("Metal FlashAttention requires mtlattn 0.4.1")
    package = import_module("mtlattn")
    if not callable(getattr(package, "varlen_attention", None)) or not callable(
        getattr(getattr(package, "_C", None), "varlen_attention_bwd", None)
    ):
        raise RuntimeError("mtlattn must provide both variable-length forward and Metal backward kernels")
    return {
        "implementation": METAL_FLASH_IMPLEMENTATION_ID,
        "torch": str(torch.__version__),
        "mtlattn": version,
        "platform": system,
        "machine": machine,
        "macos": macos,
        "python": platform.python_version(),
    }


def require_metal_flash(device) -> dict[str, str]:
    if torch.device(device).type != "mps":
        raise ValueError("model.attention='metal_flash' requires Apple MPS; choose SDPA for CPU, CUDA or HIP")
    return metal_flash_runtime_identity()


@lru_cache(maxsize=1)
def _metal_flash_function():
    metal_flash_runtime_identity()
    return import_module("mtlattn").varlen_attention


def metal_flash_eligible(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False) -> bool:
    """The small, explicitly supported portable-kernel contract; no GPU work."""
    tensors = (query, key, value)
    if attn_mask is not None or dropout_p != 0.0 or is_causal:
        return False
    if any(t.ndim != 4 or t.device.type != "mps" or t.dtype != torch.float32 for t in tensors):
        return False
    if any(
        t.device != query.device
        or t.shape[0] != query.shape[0]
        or t.shape[1] != query.shape[1]
        or t.shape[3] != query.shape[3]
        for t in (key, value)
    ):
        return False
    if query.shape[-1] not in (64, 128) or key.shape[2] != value.shape[2]:
        return False
    if any(d <= 0 for t in tensors for d in t.shape):
        return False
    # cu_seqlens and the upstream stride parameters use signed/unsigned 32-bit
    # integers. Stay below the signed bound for generated offsets and strides.
    limit = 2**31 - 1
    return all(t.shape[0] * t.shape[2] <= limit and t.shape[1] * t.shape[3] <= limit for t in tensors)


def metal_flash_sdpa(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False
):
    if not metal_flash_eligible(query, key, value, attn_mask, dropout_p, is_causal):
        if attention_check.is_checking():
            raise NotImplementedError(
                "Metal FlashAttention 只处理 MPS 上无 mask、无 dropout、非因果、FP32、head dim 64 或 128 的输入；"
                f"当前为 {query.device.type}、{attention_check.dtype_label(query.dtype)}、"
                f"{'有' if attn_mask is not None else '无'} mask、head dim {query.shape[-1]}"
            )
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    function = _metal_flash_function()
    batch, heads, q_length, head_dim = query.shape
    kv_length = key.shape[2]
    q = query.permute(0, 2, 1, 3).contiguous().view(batch * q_length, heads, head_dim)
    k = key.permute(0, 2, 1, 3).contiguous().view(batch * kv_length, heads, head_dim)
    v = value.permute(0, 2, 1, 3).contiguous().view(batch * kv_length, heads, head_dim)
    cu_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * q_length
    cu_kv = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * kv_length
    result = function(q, k, v, cu_q, cu_kv, q_length, scale=scale, causal=False)
    return result.view(batch, q_length, heads, head_dim).permute(0, 2, 1, 3)


class _MetalFlashSDPAMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # TorchFunctionMode temporarily removes this mode while invoking this
        # handler, so native fallbacks do not recursively intercept themselves.
        if func is F.scaled_dot_product_attention:
            return metal_flash_sdpa(*args, **(kwargs or {}))
        return func(*args, **(kwargs or {}))


class _ScopedProcessor:
    def __init__(self, original):
        self.original = original
        self.delegate = copy(original)
        if hasattr(self.delegate, "_attention_backend"):
            from diffusers.models.attention_dispatch import AttentionBackendName

            self.delegate._attention_backend = AttentionBackendName.NATIVE

    def run(self, *args, **kwargs):
        with _MetalFlashSDPAMode():
            return self.delegate(*args, **kwargs)


class _SDXLMetalProcessor(_ScopedProcessor):
    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs
    ):
        return self.run(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)


class _Flux2MetalProcessor(_ScopedProcessor):
    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        return self.run(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)


class _Flux2ParallelMetalProcessor(_ScopedProcessor):
    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        return self.run(attn, hidden_states, attention_mask, image_rotary_emb)


def _processor_types(family):
    if family == "sdxl":
        from diffusers.models.attention_processor import AttnProcessor2_0

        return {AttnProcessor2_0: _SDXLMetalProcessor}
    if family == "flux2":
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2AttnProcessor,
            Flux2ParallelSelfAttnProcessor,
        )

        return {
            Flux2AttnProcessor: _Flux2MetalProcessor,
            Flux2ParallelSelfAttnProcessor: _Flux2ParallelMetalProcessor,
        }
    raise ValueError(f"Metal FlashAttention processors do not support model family {family!r}")


def install_metal_flash_processors(model, family):
    """Replace only known parameter-free processors, atomically, on this model."""
    supported = _processor_types(family)
    replacement = {}
    for name, processor in model.attn_processors.items():
        original = processor.original if type(processor) in supported.values() else processor
        if type(original) not in supported or getattr(original, "_parallel_config", None) is not None:
            raise ValueError(
                f"Unsupported {family} Metal FlashAttention processor: {type(original).__name__}"
            )
        replacement[name] = (
            processor if type(processor) in supported.values() else supported[type(original)](original)
        )
    if not replacement:
        raise ValueError("Metal FlashAttention found no backbone attention processors")
    model.set_attn_processor(replacement)


def restore_metal_flash_processors(model):
    processors = getattr(model, "attn_processors", {})
    wrappers = (_SDXLMetalProcessor, _Flux2MetalProcessor, _Flux2ParallelMetalProcessor)
    if any(type(p) in wrappers for p in processors.values()):
        model.set_attn_processor(
            {name: p.original if type(p) in wrappers else p for name, p in processors.items()}
        )


def validate_metal_flash_processors(model):
    wrappers = (_SDXLMetalProcessor, _Flux2MetalProcessor, _Flux2ParallelMetalProcessor)
    processors = model.attn_processors
    if not processors or any(type(p) not in wrappers for p in processors.values()):
        raise ValueError("Metal FlashAttention backbone processors changed after loading")
