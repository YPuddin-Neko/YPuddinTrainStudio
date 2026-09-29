"""Merge an adapter file into base weights (full-precision or fp8 with re-quantization)."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from ypuddin.adapters import modules_from_tensors
from ypuddin.adapters.dora import decompose
from ypuddin.adapters.frozen import FP8_DTYPES, quantize_fp8
from ypuddin.adapters.linear import merged_bias

CONTAINER_PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.", "transformer.")


def _strip_container_prefix(key: str) -> str:
    for p in CONTAINER_PREFIXES:
        if key.startswith(p):
            return key[len(p) :]
    return key


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

    ``module_names`` (dotted names of the base model's linear and convolution layers) resolves the
    ambiguity of underscored kohya keys; without it a best-effort mapping over base keys is used. fp8
    base weights are dequantized, merged in fp32 and re-quantized with a fresh per-tensor scale.
    LyCORIS Full's ``diff_b`` is added to the layer bias.
    """
    # base checkpoints may carry a container prefix (anima-base: ``net.``, ComfyUI: ``model.diffusion_model.``);
    # adapter keys never do, so match on the stripped name and write back to the real key
    stripped_to_key = {}
    for k in base:
        if k.endswith(".weight") and base[k].dim() >= 2:
            stripped_to_key.setdefault(_strip_container_prefix(k)[: -len(".weight")], k[: -len(".weight")])
    names = module_names or list(stripped_to_key)
    table = {n.replace(".", "_"): n for n in names}
    mods = modules_from_tensors(adapter_tensors, metadata or {}, prefix=prefix)
    merged = dict(base)
    unmatched: list[str] = []
    for i, (key, (mod, dora)) in enumerate(mods.items()):
        dotted_stripped = table.get(key[len(prefix) + 1 :])
        dotted = stripped_to_key.get(dotted_stripped) if dotted_stripped is not None else None
        if dotted is None or f"{dotted}.weight" not in base:
            unmatched.append(key)
            continue
        wkey = f"{dotted}.weight"
        w = base[wkey]
        # ComfyUI ``fp8_scaled`` files store the per-tensor scale as ``scale_weight`` (+ a ``scaled_fp8`` marker);
        # ypuddin's own fp8 exports use ``weight_scale``. Read either, write back to the same key.
        scale_key = next((k for k in (f"{dotted}.scale_weight", f"{dotted}.weight_scale") if k in base), None)
        scale = base.get(scale_key) if scale_key else None
        if w.dtype in FP8_DTYPES.values():
            w32 = w.float() * (scale.float() if scale is not None else 1.0)
        else:
            w32 = w.float()
        delta = mod.delta_weight().to(w32.device).float()
        if delta.shape != w32.shape:
            if delta.numel() != w32.numel():
                unmatched.append(key)
                continue
            # A convolution's flattened factors, or a 1×1 kernel for a linear layer.
            delta = delta.reshape(w32.shape)
        if dora is None:
            new = w32 + strength * delta
        else:
            # Along the axis the stored magnitude's shape names; a strength moves from the base weight
            # toward the full DoRA weight, as ComfyUI and A1111 apply it.
            new = decompose(w32 + delta, dora)
            if strength != 1.0:
                new = w32 + strength * (new - w32)
        if mod.delta_bias() is not None:
            bkey = f"{dotted}.bias"
            if bkey not in base:
                unmatched.append(key)
                continue
            merged[bkey] = merged_bias(base[bkey], mod, strength, base[bkey].dtype)
        if w.dtype in FP8_DTYPES.values() or requantize_fp8:
            kind = requantize_fp8 or next(k for k, v in FP8_DTYPES.items() if v == w.dtype)
            q, s = quantize_fp8(new, kind)
            merged[wkey] = q
            merged[scale_key or f"{dotted}.weight_scale"] = s
        else:
            merged[wkey] = new.to(w.dtype)
        if progress:
            progress(i + 1, len(mods))
    return merged, unmatched
