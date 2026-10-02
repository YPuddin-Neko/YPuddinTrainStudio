"""Preserve DoRA computation semantics across training resumes and portable warm starts."""

from __future__ import annotations

import json
from typing import Any

import torch

from .components import ComponentAdapterSet
from .inject import DORA_ALGOS

METADATA_KEY = "ypuddin.dora_compute"
STATE_KEY = "dora_compute"
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def validate_active_config(adapters, config, *, compute_dtype: torch.dtype | None = None) -> None:
    """Reject changed save/compute settings before gathering or writing weights."""
    if config.training.mode != "adapter":
        return
    requested = config.adapter
    for layer in getattr(adapters, "layers", {}).values():
        enabled = requested.dora and layer.adapter.kind in DORA_ALGOS
        dora = layer.dora
        matches = enabled == (dora is not None)
        if dora is not None:
            matches = matches and (dora.compute_mode, dora.axis) == (requested.dora_compute_mode, requested.dora_axis)
            if dora.compute_mode == "comfyui":
                merge_dtype = (compute_dtype or torch.float32) if getattr(layer.base, "is_fp8", False) else layer.base.weight.dtype
                if requested.dora_merge_dtype != "auto":
                    merge_dtype = _DTYPES[requested.dora_merge_dtype]
                matches = matches and (dora.merge_dtype, dora.save_dtype) == (merge_dtype, _DTYPES[config.checkpoint.save_dtype])
        if not matches:
            raise ValueError("DoRA 计算设置或权重保存精度在训练期间发生变化，不能保存。请恢复启动训练时的设置。")


def compute_contract(adapters) -> dict[str, Any] | None:
    components = adapters.components.values() if isinstance(adapters, ComponentAdapterSet) else (adapters,)
    layers = {}
    for component in components:
        for name, layer in getattr(component, "layers", {}).items():
            dora = layer.dora
            if dora is None or dora.compute_mode == "standard":
                continue
            layers[component.export_key(name)] = [
                dora.axis,
                str(dora.merge_dtype).removeprefix("torch."),
                str(dora.save_dtype).removeprefix("torch."),
            ]
    return {"version": 1, "mode": "comfyui", "layers": layers} if layers else None


def validate_resume_contract(current, saved) -> None:
    if current != saved:
        raise ValueError(
            "DoRA 计算模式、融合精度或保存精度与原训练不一致，不能继续训练。"
            "请恢复原设置；没有计算模式记录的旧权重使用标准模式。"
        )


def portable_contract(contract) -> dict[str, Any] | None:
    if contract is None:
        return None
    return {
        "version": contract["version"],
        "mode": contract["mode"],
        "precisions": sorted({tuple(value) for value in contract["layers"].values()}),
    }


def export_contract_metadata(adapters) -> dict[str, str]:
    contract = portable_contract(compute_contract(adapters))
    return {METADATA_KEY: json.dumps(contract, separators=(",", ":"))} if contract else {}


def validate_warm_start(adapters, metadata: dict[str, str]) -> None:
    raw = metadata.get(METADATA_KEY)
    try:
        saved = json.loads(raw) if raw is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("权重文件中的 DoRA 计算设置无效") from exc
    # JSON normalizes precision tuples to lists.
    current = json.loads(json.dumps(portable_contract(compute_contract(adapters))))
    validate_resume_contract(current, saved)


def canonical_adapter_contract(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    if not config.get("dora") or config.get("dora_compute_mode", "standard") == "standard":
        config.pop("dora_compute_mode", None)
        config.pop("dora_merge_dtype", None)
    return config
