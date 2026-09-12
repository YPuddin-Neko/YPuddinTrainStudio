"""Bounded local metadata inspection; never deserialize pickle or load model tensors."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

HEADER_LIMIT = 32 * 1024 * 1024
CONFIG_LIMIT = 2 * 1024 * 1024
MAX_FILES = 1024


def _json(path: Path, limit: int) -> dict[str, Any]:
    if path.stat().st_size > limit:
        raise ValueError(f"{path.name}: metadata exceeds the inspection limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: metadata must be a JSON object")
    return value


def _header(path: Path) -> dict[str, dict]:
    # Read only the bounded JSON header; safe_open validates offsets and shapes
    # through its mmap without materializing any tensor or allocating GPU memory.
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 2 <= length <= HEADER_LIMIT or length + 8 > path.stat().st_size:
            raise ValueError(f"{path.name}: invalid or oversized safetensors header")
    from safetensors import SafetensorError, safe_open

    tensors = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                item = handle.get_slice(key)
                tensors[key] = {"shape": item.get_shape(), "dtype": item.get_dtype()}
    except SafetensorError as error:
        raise ValueError(f"{path.name}: invalid safetensors metadata") from error
    return tensors


def _key(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "net.", "first_stage_model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def inspect_model(path: Path, *, allowed=None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if allowed and not allowed(path):
        raise ValueError("model path is outside allowed storage roots")
    if not path.exists():
        raise ValueError("model path does not exist")
    root = path if path.is_dir() else path.parent
    config_path = root / "config.json"
    if config_path.is_file() and allowed and not allowed(config_path.resolve()):
        raise ValueError("model config is outside allowed storage roots")
    config = _json(config_path, CONFIG_LIMIT) if config_path.is_file() else {}
    files = []
    mapping = {}
    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            raise ValueError(
                "Only safetensors files and local model/tokenizer directories can be inspected; pickle is never executed"
            )
        files = [path]
    else:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            if not index.resolve().is_relative_to(root) or allowed and not allowed(index.resolve()):
                raise ValueError("shard index is outside the selected directory")
            mapping = _json(index, CONFIG_LIMIT).get("weight_map", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("safetensors shard index has no weight_map")
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
                raise ValueError("invalid safetensors shard mapping")
            names = set(mapping.values())
            if len(names) > MAX_FILES:
                raise ValueError("too many model shards")
            for name in sorted(names):
                file = (root / name).resolve()
                if (
                    not file.is_relative_to(root)
                    or file.suffix.lower() != ".safetensors"
                    or not file.is_file()
                ):
                    raise ValueError("model shard is missing or outside the selected directory")
                files.append(file)
        else:
            files = sorted(root.glob("*.safetensors"))
            if len(files) > MAX_FILES:
                raise ValueError("too many weight files in the selected directory")
    shapes = {}
    dtypes = Counter()
    weight_dtypes = Counter()
    metadata_bytes = 0
    for file in files:
        with file.open("rb") as stream:
            metadata_bytes += int.from_bytes(stream.read(8), "little")
        if metadata_bytes > 64 * 1024 * 1024:
            raise ValueError("combined model headers exceed the inspection limit")
        if path.is_dir() and not file.resolve().is_relative_to(root):
            raise ValueError("model shard is outside the selected directory")
        if allowed and not allowed(file.resolve()):
            raise ValueError("model shard is outside allowed storage roots")
        for key, item in _header(file).items():
            key = _key(key)
            if key in shapes:
                raise ValueError(
                    "multiple checkpoints contain duplicate tensor keys; select one file or a complete sharded model directory"
                )
            shapes[key] = item["shape"]
            count = math.prod(item["shape"])
            dtypes[item["dtype"]] += count
            if key.endswith("weight") and len(item["shape"]) >= 2:
                weight_dtypes[item["dtype"]] += count
    if mapping and not {_key(key) for key in mapping} <= shapes.keys():
        raise ValueError("model shard index references missing tensors")
    dtype_names = {
        "F32": "fp32",
        "F16": "fp16",
        "BF16": "bf16",
        "F8_E4M3": "fp8",
        "F8_E4M3FN": "fp8",
        "F8_E5M2": "fp8",
    }
    floating = {
        dtype_names[k] for k, count in (weight_dtypes or dtypes).items() if count and k in dtype_names
    }
    dtype = next(iter(floating)) if len(floating) == 1 else "mixed" if floating else None
    if "fp8" in floating and any(
        key.endswith((".scale_weight", ".weight_scale")) or key == "scaled_fp8" for key in shapes
    ):
        dtype = "fp8"
    family, kind, evidence, candidates = None, None, [], []
    if (
        "x_embedder.proj.1.weight" in shapes
        and len(shapes["x_embedder.proj.1.weight"]) == 2
        and any(key.startswith("llm_adapter.") for key in shapes)
        and any(key.startswith("blocks.0.") for key in shapes)
    ):
        family, kind = "anima", "dit"
        evidence.append("Anima patch embedding + LLM adapter + transformer block keys")
    elif {"first.weight", "txtfusion.projector.weight", "blocks.0.attn.wq.weight"} <= shapes.keys() and len(
        shapes["first.weight"]
    ) == 2:
        family, kind = "krea2", "dit"
        evidence.append("Krea 2 single-stream projection + layerwise text-fusion keys")
    elif (
        any(key.startswith("encoder.") for key in shapes)
        and any(key.startswith("decoder.") for key in shapes)
        and "quant_conv.weight" in shapes
    ):
        kind = "vae"
        if shapes.get("post_quant_conv.weight", [])[:2] == [16, 16] and len(shapes["quant_conv.weight"]) == 5:
            candidates = ["anima", "krea2"]
            evidence.append("Qwen-Image causal VAE geometry; shared by Anima and Krea 2")
        else:
            evidence.append("Encoder / decoder / quantization convolution keys; family compatibility unknown")
    else:
        model_type = str(config.get("model_type", ""))
        text_cfg = config.get("text_config", {})
        text_cfg = text_cfg if isinstance(text_cfg, dict) else {}
        embed = next(
            (
                shape
                for key, shape in shapes.items()
                if key.endswith("embed_tokens.weight") and len(shape) == 2
            ),
            None,
        )
        decoder = any(".self_attn.q_proj.weight" in key for key in shapes)
        if embed and decoder:
            kind = "text_encoder"
            if model_type in ("qwen3_vl", "qwen3_vl_text") or text_cfg.get("model_type") == "qwen3_vl_text":
                family = "krea2"
                evidence.append("Qwen3-VL config and decoder tensor keys")
            elif embed == [151936, 2560] and any("q_norm.weight" in key for key in shapes):
                family = "krea2"
                evidence.append("Qwen3-VL-4B decoder embedding and query-normalization geometry")
            elif embed == [151936, 1024] and any("q_norm.weight" in key for key in shapes):
                family = "anima"
                evidence.append("Qwen3-0.6B decoder embedding and query-normalization geometry")
            else:
                evidence.append("Text decoder keys found; training family compatibility unknown")
        elif (
            not files
            and (root / "tokenizer_config.json").is_file()
            and any(
                (root / name).is_file()
                for name in ("tokenizer.json", "vocab.json", "spiece.model", "tokenizer.model")
            )
        ):
            kind = "tokenizer"
            evidence.append("Local tokenizer configuration and vocabulary assets")
    if family:
        candidates = [family]
    warnings = []
    if not family:
        warnings.append(
            "模型系列尚不能唯一确定，请确认兼容系列。"
            if candidates
            else "未识别模型系列；请根据模型来源确认，不按文件名猜测。"
        )
    if not kind:
        warnings.append("未识别组件类型，请手动确认。")
    if dtype == "mixed":
        warnings.append("矩阵权重包含多种精度，已标记为 mixed。")
    if not dtype and kind != "tokenizer":
        warnings.append("未识别浮点权重精度，保留未知。")
    return {
        "path": str(files[0] if path.is_dir() and kind in ("dit", "vae") and len(files) == 1 else path),
        "family": family,
        "family_candidates": candidates,
        "kind": kind,
        "dtype": dtype,
        "dtypes": dict(dtypes),
        "confidence": "high" if family and kind else "partial" if kind else "unknown",
        "evidence": evidence,
        "warnings": warnings,
        "files_inspected": len(files),
    }
