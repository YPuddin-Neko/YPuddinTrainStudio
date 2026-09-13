"""Krea Raw/Turbo identity; equal tensor geometry is not variant evidence.

The downloader records an already verified catalog digest in a small sidecar,
bound to the ordinary file's current stat signature. Unknown local weights need
an explicit variant; names such as ``turbo.safetensors`` are never evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TURBO_TRAINING_ERROR = "Krea 2 Turbo 是蒸馏推理模型，仅用于采样；训练请使用 Krea 2 Raw。"
KNOWN_VARIANTS = {
    "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7": "raw",
    "48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61": "raw",
    "eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1": "turbo",
    "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3": "turbo",
}


def _signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".ypuddin.json")


def record_verified_variant(path: Path, sha256: str) -> None:
    """Called only after a download/local catalog verification has checked SHA256."""
    if sha256 not in KNOWN_VARIANTS:
        return
    target = _sidecar(path)
    partial = target.with_name(target.name + ".tmp")
    data = {"version": 1, "sha256": sha256, "signature": _signature(path)}
    try:
        partial.write_text(json.dumps(data), encoding="utf-8")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def verified_variant(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path).expanduser()
    try:
        sidecar = _sidecar(path)
        if sidecar.stat().st_size > 8192:
            return None
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if data.get("version") == 1 and data.get("signature") == _signature(path):
            return KNOWN_VARIANTS.get(data.get("sha256"))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def resolve_variant(path: str | Path | None, requested: str = "raw") -> str:
    known = verified_variant(path)
    if requested not in {"auto", "raw", "turbo"}:
        raise ValueError("Krea 2 variant must be auto, raw or turbo")
    if known:
        if requested != "auto" and requested != known:
            raise ValueError(f"Krea 2 文件已验证为 {known}，与所选 {requested} 不符，请确认模型版本。")
        return known
    if requested == "auto":
        raise ValueError("Krea 2 Raw/Turbo 权重形状相同，无法自动区分；请按发布说明选择 Raw 或 Turbo。")
    return requested
