"""Display categories for exported adapters and full model components."""

from __future__ import annotations

import json
from pathlib import Path


def export_type(kind: str, path: Path, *, family: str | None = None, config_json: str | None = None) -> str | None:
    if kind in {"weights", "comfyui", "kohya"}:
        return "lora"
    if kind != "model":
        return None
    if path.is_dir():
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if isinstance(manifest, dict) and manifest.get("format") == "ypuddin-full-model-v1":
                family = manifest.get("family") or family
        except (OSError, ValueError):
            pass
    if not family and config_json:
        try:
            config = json.loads(config_json)
            if isinstance(config, dict) and isinstance(config.get("model"), dict):
                family = config["model"].get("family")
        except (TypeError, ValueError):
            pass
    if not isinstance(family, str):
        return None
    if family in {"sdxl", "sdxl_base_v1-0"}:
        return "checkpoint"
    if family in {"anima", "flux2", "krea2"}:
        return "diffusion_model"
    return None
