"""Resolve model loading precision from bounded checkpoint metadata."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

_NAMES = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}


def checkpoint_precision(path: str | None) -> str | None:
    if not path:
        return None
    root = Path(path).expanduser()
    if root.is_dir():
        # Diffusers roots include VAE/text weights with independent precisions.
        root = next((root / name for name in ("transformer", "unet") if (root / name).is_dir()), root)
        files = sorted(root.glob("*.safetensors"))
    else:
        files = [root] if root.suffix == ".safetensors" else []
    counts: Counter[str] = Counter()
    for file in files:
        try:
            with file.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 2 <= size <= 32 * 1024 * 1024 or size + 8 > file.stat().st_size:
                    continue
                tensors = json.loads(stream.read(size))
            for name, tensor in tensors.items():
                if name == "__metadata__" or not isinstance(tensor, dict):
                    continue
                shape = tensor.get("shape", [])
                dtype = tensor.get("dtype", "")
                if len(shape) >= 2 and (dtype in _NAMES or dtype.startswith("F8_")):
                    counts[dtype] += math.prod(shape)
        except (OSError, ValueError, TypeError, OverflowError):
            continue
    if not counts:
        metadata = root / "config.json"
        try:
            if root.is_dir() and metadata.stat().st_size <= 1024 * 1024:
                config = json.loads(metadata.read_text(encoding="utf-8"))
                return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
                    config.get("torch_dtype", config.get("dtype"))
                )
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return None
    dtype = counts.most_common(1)[0][0]
    # FP8 describes checkpoint storage, not an activation dtype. The family
    # loader still validates whether it supports that checkpoint format.
    return "bf16" if dtype.startswith("F8_") else _NAMES[dtype]


def model_load_precision(model, device_type: str | None = None) -> str:
    if device_type in {"cpu", "mps"}:
        return "fp32"
    if model.dtype != "auto":
        return model.dtype
    return checkpoint_precision(model.dit_path) or "bf16"
