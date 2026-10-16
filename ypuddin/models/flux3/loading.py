"""Local, strict loading of BFL's undistilled FLUX 3 Image Klein weights."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .dit import DiT, DiTConfig

BASE_CONFIG = DiTConfig()


def runtime_error() -> str | None:
    from packaging.version import InvalidVersion, Version

    try:
        current = version("transformers")
        if Version(current) >= Version("5.8.1"):
            return None
    except (PackageNotFoundError, InvalidVersion):
        current = "未安装"
    return (f"FLUX.3 需要 Transformers 5.8.1 或更新版本，当前为 {current}。"
            "请通过运行环境准备新的 CUDA／CPU／MPS 环境，或重新运行对应启动入口补齐依赖。")


def component_paths(cfg):
    root = Path(cfg.dit_path or ".").expanduser()
    directory = root if root.is_dir() else root.parent
    dit = root / "model.safetensors" if root.is_dir() else root
    vae = Path(cfg.vae_path).expanduser() if cfg.vae_path else directory / "ae.safetensors"
    text = Path(cfg.text_encoder_path).expanduser() if cfg.text_encoder_path else directory / "text_encoder"
    processor = Path(cfg.tokenizer_path).expanduser() if cfg.tokenizer_path else text
    return root, dit, text, vae, processor


def transformer_config(path: Path) -> DiTConfig:
    """Validate every key and shape from the header without loading weight payloads."""
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError("FLUX.3 请选择 Base 的 model.safetensors 或包含它的完整本地目录。")
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        if any(key.startswith("ref_time_in.") for key in keys):
            raise ValueError("FLUX.3 蒸馏版使用不同的时间条件；训练请选择 Klein Base。")
        if any(
            key.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))
            or weights.get_slice(key).get_dtype() not in {"BF16", "F32"}
            for key in keys
        ):
            raise ValueError("FLUX.3 训练需要 BF16 或 FP32 权重，不支持 FP8／NVFP4 推理权重。")
        with torch.device("meta"):
            expected = DiT(config=BASE_CONFIG).state_dict()
        missing, unexpected = sorted(set(expected) - keys), sorted(keys - set(expected))
        if missing or unexpected:
            raise ValueError(f"FLUX.3 Base 权重不完整或类型不符：missing={missing[:3]}, unexpected={unexpected[:3]}")
        mismatched = [key for key, tensor in expected.items()
                      if tuple(weights.get_slice(key).get_shape()) != tuple(tensor.shape)]
        if mismatched:
            raise ValueError(f"FLUX.3 Base 权重尺寸不符：{mismatched[:3]}")
    return BASE_CONFIG


def load_transformer(path: Path, config: DiTConfig, *, device, dtype, checkpointing="none"):
    transformer_config(path)
    if dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("FLUX.3 主模型计算精度请选择 BF16 或 FP32。")
    with torch.device("meta"):
        model = DiT(config=config, checkpointing=checkpointing)
    state = load_file(str(path), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()


def config_dict(config: DiTConfig) -> dict:
    return asdict(config)
