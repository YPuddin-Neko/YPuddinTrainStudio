"""Merge an adapter file into base weights (full-precision or fp8 with re-quantization)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ypuddin.adapters import modules_from_tensors
from ypuddin.adapters.frozen import FP8_DTYPES, quantize_fp8


def merge_into_state_dict(
    base: dict[str, Tensor],
    adapter_tensors: dict[str, Tensor],
    metadata: dict[str, str] | None = None,
    *,
    prefix: str = "lora_unet",
    strength: float = 1.0,
    module_names: list[str] | None = None,
    requantize_fp8: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Tensor], list[str]]:
    """Returns ``(merged_state_dict, unmatched_adapter_modules)``.

    ``module_names`` (dotted names of the base model's linear layers) resolves the ambiguity of
    underscored kohya keys; without it a best-effort mapping over base keys is used. fp8 base
    weights are dequantized, merged in fp32 and re-quantized with a fresh per-tensor scale.
    """
    mods = modules_from_tensors(adapter_tensors, metadata or {}, prefix=prefix)
    names = module_names or [k[: -len(".weight")] for k in base if k.endswith(".weight") and base[k].dim() == 2]
    table = {n.replace(".", "_"): n for n in names}
    merged = dict(base)
    unmatched: list[str] = []
    for i, (key, (mod, dora)) in enumerate(mods.items()):
        dotted = table.get(key[len(prefix) + 1 :])
        if dotted is None or f"{dotted}.weight" not in base:
            unmatched.append(key)
            continue
        wkey = f"{dotted}.weight"
        w = base[wkey]
        scale = base.get(f"{dotted}.weight_scale")
        if w.dtype in FP8_DTYPES.values():
            w32 = w.float() * (scale.float() if scale is not None else 1.0)
        else:
            w32 = w.float()
        delta = mod.delta_weight().to(w32.device).float()
        new = w32 + strength * delta
        if dora is not None:
            norm = new.reshape(new.shape[0], -1).norm(dim=1, keepdim=True).clamp(min=1e-12)
            new = new * (dora.float().reshape(-1, 1) / norm)
        if w.dtype in FP8_DTYPES.values() or requantize_fp8:
            kind = requantize_fp8 or next(k for k, v in FP8_DTYPES.items() if v == w.dtype)
            q, s = quantize_fp8(new, kind)
            merged[wkey] = q
            merged[f"{dotted}.weight_scale"] = s
        else:
            merged[wkey] = new.to(w.dtype)
        if progress:
            progress(i + 1, len(mods))
    return merged, unmatched
