"""Merge an adapter file into base weights (full-precision or fp8 with re-quantization)."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from ypuddin.adapters import modules_from_tensors
from ypuddin.adapters.dora import decompose
from ypuddin.adapters.frozen import FP8_DTYPES, quantize_fp8

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

    def target(key: str) -> str | None:
        dotted_stripped = table.get(key[len(prefix) + 1 :])
        dotted = stripped_to_key.get(dotted_stripped) if dotted_stripped is not None else None
        return dotted if dotted is not None and f"{dotted}.weight" in base else None

    # Files keep a low-rank convolution's kernel only inside flattened factors; the base weight names it.
    kernels = {}
    for key in {k.partition(".")[0] for k in adapter_tensors}:
        dotted = target(key)
        if dotted is not None:
            kernels[key] = tuple(base[f"{dotted}.weight"].shape[2:])
    mods = modules_from_tensors(adapter_tensors, metadata or {}, prefix=prefix, kernels=kernels)
    merged = dict(base)
    unmatched: list[str] = []
    for i, (key, (mod, dora)) in enumerate(mods.items()):
        dotted = target(key)
        if dotted is None:
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
            delta = delta.reshape(w32.shape)  # e.g. a 1×1 kernel for a linear layer
        new = w32 + strength * delta
        if dora is not None:
            new = decompose(new, dora)  # along the axis the stored magnitude's shape names
        delta_bias = getattr(mod, "delta_bias", None)
        bias_delta = delta_bias() if callable(delta_bias) else None
        if bias_delta is not None:
            bkey = f"{dotted}.bias"
            if bkey not in base:
                unmatched.append(key)
                continue
            merged[bkey] = (base[bkey].float() + strength * bias_delta.float()).to(base[bkey].dtype)
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
