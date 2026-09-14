"""Header-only validation of formats the Anima loaders can actually consume."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open


def check_unquantized_checkpoint(path: str | Path, component: str) -> None:
    """Do not allow a normal dtype cast to silently discard quantization scales.

    This is separate from quantizing an already-loaded BF16/FP16/FP32 frozen DiT
    through memory.base_precision. HF directories inspect only their selected
    standard safetensors files, not unrelated models in the same directory.
    """
    root = Path(path).expanduser()
    if root.is_dir():
        model = root / "model.safetensors"
        index = root / "model.safetensors.index.json"
        if model.is_file():
            files = [model]
        elif index.is_file():
            names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            files = []
            for name in sorted(names):
                file = root / name
                if not file.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"{component} 权重索引包含目录之外的分片")
                files.append(file)
        else:
            return  # The existing loader reports missing/unrecognized model files.
    else:
        files = [root] if root.suffix.lower() == ".safetensors" else []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if (
                    key.rsplit(".", 1)[-1] == "scaled_fp8"
                    or key.endswith((".weight_scale", ".scale_weight"))
                    or weights.get_slice(key).get_dtype().startswith("F8")
                ):
                    raise ValueError(
                        f"{component} 暂不支持直接加载现成 FP8 或带量化缩放的权重；"
                        "请选择 BF16、FP16 或 FP32 原始权重。"
                        "底模设置中的 FP8 是启动时量化，并不代表支持现成 FP8 文件。"
                    )
