"""Resolve DoRA precision from the prepared layers used by the training plan."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

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
        auto_dtype = compute_dtype if getattr(layer.base, "is_fp8", False) else base_dtype
        merge_dtype = layer.dora.merge_dtype if layer.dora.compute_mode == "comfyui" else torch.float32
        base_dtypes.add(_DTYPES.get(base_dtype, str(base_dtype).removeprefix("torch.")))
        auto_dtypes.add(_DTYPES[auto_dtype])
        merge_dtypes.add(_DTYPES[merge_dtype])
        mismatch |= layer.dora.compute_mode == "comfyui" and merge_dtype != auto_dtype
    reasons, messages = [], []
    if layers and cfg.adapter.dora_compute_mode == "standard" and not (
        cfg.checkpoint.resume or cfg.adapter.resume_weights
    ):
        reasons.append("standard_mode")
        messages.append("标准模式使用 FP32 计算 DoRA；在低精度下加载导出权重时，出图效果可能与训练预览不同。")
    if mismatch:
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


def resume_errors(adapters, cfg) -> list[dict[str, str]]:
    errors = []
    if cfg.checkpoint.resume:
        path = Path(cfg.checkpoint.resume).expanduser() / "state.json"
        if path.is_file():
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                extra = metadata.get("progress", {}).get("extra", {})
                validate_resume_contract(compute_contract(adapters), extra.get(STATE_KEY))
            except (OSError, UnicodeError, ValueError, AttributeError) as error:
                errors.append({"loc": "checkpoint.resume", "msg": str(error)})
    if cfg.adapter.resume_weights:
        try:
            with safe_open(str(Path(cfg.adapter.resume_weights).expanduser()), framework="pt", device="cpu") as weights:
                validate_warm_start(adapters, weights.metadata() or {})
        except (OSError, ValueError, RuntimeError, SafetensorError) as error:
            errors.append({"loc": "adapter.resume_weights", "msg": str(error)})
    return errors
