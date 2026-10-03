"""Preserve DoRA computation semantics across training resumes and portable warm starts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .components import ComponentAdapterSet
from .dora import resolve_merge_dtype
from .inject import wants_dora

METADATA_KEY = "ypuddin.dora_compute"
STATE_KEY = "dora_compute"
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def validate_active_config(adapters, config, *, compute_dtype: torch.dtype | None = None) -> None:
    """Reject changed save/compute settings before gathering or writing weights."""
    if config.training.mode != "adapter":
        return
    requested = config.adapter
    for layer in getattr(adapters, "layers", {}).values():
        dora = layer.dora
        # A single-output layer keeps DoRA only where a resumed state held its magnitude.
        enabled = wants_dora(requested, layer.adapter.kind, layer.adapter.out_features, keep=dora is not None)
        matches = enabled == (dora is not None)
        if dora is not None:
            matches = matches and (dora.compute_mode, dora.axis) == (requested.dora_compute_mode, requested.dora_axis)
            if dora.compute_mode == "comfyui":
                merge_dtype = resolve_merge_dtype(requested.dora_merge_dtype, layer.base, compute_dtype)
                matches = matches and (dora.merge_dtype, dora.save_dtype) == (merge_dtype, _DTYPES[config.checkpoint.save_dtype])
        if not matches:
            raise ValueError("DoRA 计算设置或权重保存精度在训练期间发生变化，不能保存。请恢复启动训练时的设置。")


def saved_magnitudes(path: str | Path) -> set[str]:
    """Layers whose resume point holds a DoRA magnitude: layer names from the training or sharded state,
    export keys from the oldest states, which kept only the exported adapter."""
    path = Path(path).expanduser()
    for name, suffix in (
        ("training.safetensors", ".dora.dora_scale"),
        ("model.safetensors", ".dora.dora_scale"),
        ("adapter.safetensors", ".dora_scale"),
    ):
        if (path / name).is_file():
            with safe_open(str(path / name), framework="pt", device="cpu") as tensors:
                return {key.removesuffix(suffix) for key in tensors.keys() if key.endswith(suffix)}
    return set()


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
