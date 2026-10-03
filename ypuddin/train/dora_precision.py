"""Resolve DoRA precision from the prepared layers used by the training plan."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

from ypuddin.adapters.components import ComponentAdapterSet
from ypuddin.adapters.dora import auto_merge_dtype
from ypuddin.adapters.dora_contract import (
    STATE_KEY,
    compute_contract,
    validate_resume_contract,
    validate_warm_start,
)

_DTYPES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def precision_report(adapters, cfg, compute_dtype: torch.dtype) -> dict:
    layers = [layer for layer in adapters.layers.values() if layer.dora is not None]
    base_dtypes, auto_dtypes, merge_dtypes = set(), set(), set()
    mismatch = False
    for layer in layers:
        base_dtype = layer.base.weight.dtype
        auto_dtype = auto_merge_dtype(layer.base, compute_dtype)
        merge_dtype = layer.dora.merge_dtype if layer.dora.compute_mode == "comfyui" else torch.float32
        base_dtypes.add(_DTYPES.get(base_dtype, str(base_dtype).removeprefix("torch.")))
        auto_dtypes.add(_DTYPES[auto_dtype])
        merge_dtypes.add(_DTYPES[merge_dtype])
        mismatch |= layer.dora.compute_mode == "comfyui" and merge_dtype != auto_dtype
    # A resume point or warm-start file fixes the computation; the plan checks it matches instead.
    continuing = bool(cfg.checkpoint.resume or cfg.adapter.resume_weights)
    reasons, messages = [], []
    if layers and cfg.adapter.dora_compute_mode == "standard" and not continuing:
        reasons.append("standard_mode")
        messages.append(
            "标准模式按 FP32 计算 DoRA；出图软件融合导出权重时使用自身的精度和算法，效果与训练计算有差异。"
            "默认训练预览显示 ComfyUI 融合后的效果。"
        )
    if mismatch and not continuing:
        reasons.append("manual_merge_dtype_mismatch")
        auto = " / ".join(item.upper() for item in sorted(auto_dtypes))
        selected = " / ".join(item.upper() for item in sorted(merge_dtypes))
        messages.append(
            f"当前底模对应的自动融合精度为 {auto}，手动选择为 {selected}。"
            "请确认实际出图软件使用相同的融合精度，否则可能影响出图效果。"
        )
    return {
        "active": bool(layers),
        "compute_mode": cfg.adapter.dora_compute_mode,
        "base_dtypes": sorted(base_dtypes),
        "auto_merge_dtypes": sorted(auto_dtypes),
        "merge_dtypes": sorted(merge_dtypes),
        "save_dtype": cfg.checkpoint.save_dtype,
        "confirmation_required": bool(reasons),
        "confirmation_reasons": reasons,
        "confirmation_message": "\n".join(messages) or None,
    }


def _single_output_records(adapters, saved):
    """The saved record without single-output layers this plan leaves without DoRA. The trainer keeps
    DoRA on those layers when their resume point holds a magnitude, and checks the full record then."""
    if not isinstance(saved, dict) or not isinstance(saved.get("layers"), dict):
        return saved
    components = adapters.components.values() if isinstance(adapters, ComponentAdapterSet) else (adapters,)
    single = {
        component.export_key(name)
        for component in components
        for name, layer in getattr(component, "layers", {}).items()
        if layer.dora is None and layer.adapter.out_features == 1
    }
    layers = {key: value for key, value in saved["layers"].items() if key not in single}
    if layers == saved["layers"]:
        return saved
    return {**saved, "layers": layers} if layers else None


def resume_errors(adapters, cfg) -> list[dict[str, str]]:
    errors = []
    if cfg.checkpoint.resume:
        path = Path(cfg.checkpoint.resume).expanduser() / "state.json"
        if path.is_file():
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                extra = metadata.get("progress", {}).get("extra", {})
                saved = _single_output_records(adapters, extra.get(STATE_KEY))
                validate_resume_contract(compute_contract(adapters), saved)
            except (OSError, UnicodeError, ValueError, AttributeError) as error:
                errors.append({"loc": "checkpoint.resume", "msg": str(error)})
    if cfg.adapter.resume_weights:
        try:
            with safe_open(str(Path(cfg.adapter.resume_weights).expanduser()), framework="pt", device="cpu") as weights:
                validate_warm_start(adapters, weights.metadata() or {})
        except (OSError, ValueError, RuntimeError, SafetensorError) as error:
            errors.append({"loc": "adapter.resume_weights", "msg": str(error)})
    return errors
