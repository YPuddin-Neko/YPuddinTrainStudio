"""Strict local loading of the released MOSS-Image Base components."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ypuddin.runtime_imports import LIBRARY_IMPORTS

from .dit import MOSSImageTransformer

BASE_CONFIG = dict(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096,
                   freq_dim=256, latent_channels=128, rope_base=10000.0)


def runtime_error() -> str | None:
    from packaging.version import Version

    for package, minimum in (("torch", "2.9.0"), ("transformers", "5.8.1"), ("diffusers", "0.40.0")):
        try:
            if Version(version(package)) < Version(minimum):
                return f"MOSS-Image 需要 {package} {minimum} 或更新版本。"
        except (PackageNotFoundError, ValueError):
            return f"MOSS-Image 需要安装 {package} {minimum} 或更新版本。"
    try:
        with LIBRARY_IMPORTS:
            from diffusers import AutoencoderKLFlux2  # noqa: F401
            from transformers import Qwen3_5Model  # noqa: F401
    except (ImportError, RuntimeError) as error:
        return f"当前模型运行环境缺少 MOSS-Image 所需组件：{error}"
    return None


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"MOSS-Image 本地配置不存在或过大：{path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MOSS-Image 配置需要 JSON 对象：{path.name}")
    return data


def tokenizer_directory(text: Path) -> Path:
    directory = text if text.is_dir() else text.parent
    sibling = directory.parent / "tokenizer"
    return sibling if sibling.is_dir() else directory


def component_paths(cfg):
    root = Path(cfg.dit_path or ".").expanduser()
    directory = root if root.is_dir() else root.parent
    if directory.name == "transformer":
        directory = directory.parent
    dit = root / "transformer" if root.is_dir() and (root / "transformer").is_dir() else root
    references = {}
    manifest = directory / ".ypuddin-moss-image-components.json"
    if manifest.exists():
        data = read_json(manifest)
        references = data.get("components")
        if (data.get("format") != "ypuddin-moss-image-components" or type(data.get("version")) is not int
                or data["version"] != 1 or not isinstance(references, dict)
                or set(references) != {"text_encoder", "vae", "tokenizer"}
                or any(not isinstance(p, str) or not Path(p).is_absolute() for p in references.values())):
            raise ValueError("MOSS-Image 组件引用需要 version=1 及文本编码器、VAE、分词器的绝对路径。")
        references = {key: Path(value) for key, value in references.items()}
    text = Path(cfg.text_encoder_path).expanduser() if cfg.text_encoder_path else references.get("text_encoder", directory / "text_encoder")
    vae = Path(cfg.vae_path).expanduser() if cfg.vae_path else references.get("vae", directory / "vae")
    tokenizer = tokenizer_directory(text)
    if not cfg.text_encoder_path:
        tokenizer = references.get("tokenizer", tokenizer)
    tokenizer = Path(cfg.tokenizer_path).expanduser() if cfg.tokenizer_path else tokenizer
    return root, dit, text, vae, tokenizer


def weight_files(path: Path, basename: str = "model"):
    path = Path(path)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError("MOSS-Image 权重需要 safetensors 格式。")
        return [path], None
    index = path / f"{basename}.safetensors.index.json"
    if index.is_file():
        mapping = read_json(index).get("weight_map")
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())):
            raise ValueError("MOSS-Image 权重分片索引无效。")
        files = [path / name for name in sorted(set(mapping.values()))]
    else:
        mapping, files = None, [path / f"{basename}.safetensors"]
    if any(not p.is_file() or p.suffix != ".safetensors" or not p.resolve().is_relative_to(path.resolve()) for p in files):
        raise ValueError("MOSS-Image 缺少本地权重分片。")
    return files, mapping


def validate_weights(path: Path, expected: dict, *, basename="model", integer_keys=()) -> int:
    files, mapping = weight_files(path, basename)
    shapes, elements = {}, 0
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if key in shapes or mapping is not None and (key not in mapping or (path / mapping[key]).resolve() != file.resolve()):
                    raise ValueError(f"MOSS-Image 权重分片索引不符或重复：{key}")
                tensor = weights.get_slice(key)
                allowed = {"BF16", "F32"} | ({"I64", "I32"} if key in integer_keys else set())
                if tensor.get_dtype() not in allowed:
                    raise ValueError(f"MOSS-Image 需要未量化 BF16 或 FP32 权重：{key}")
                shapes[key] = tuple(tensor.get_shape())
                elements += expected[key].numel() if key in expected else 0
    missing, extra = set(expected) - shapes.keys(), shapes.keys() - set(expected)
    wrong = [key for key in shapes.keys() & expected.keys() if shapes[key] != tuple(expected[key].shape)]
    if mapping is not None and set(mapping) != set(shapes):
        raise ValueError("MOSS-Image 权重分片缺少索引中的张量。")
    if missing or extra or wrong:
        raise ValueError(f"MOSS-Image 权重与配置不符：missing={sorted(missing)[:3]}, unexpected={sorted(extra)[:3]}, shapes={sorted(wrong)[:3]}")
    return elements


def transformer_config(path: Path) -> dict:
    path = Path(path)
    config = read_json((path if path.is_dir() else path.parent) / "config.json")
    if config.get("_class_name") != "MOSSImageTransformer" or config.get("quantization_config"):
        raise ValueError("请选择 MOSS-Image Base 原始未量化 DiT。")
    if any(config.get(key) != value for key, value in BASE_CONFIG.items()):
        raise ValueError("MOSS-Image DiT 配置与 Base 结构不符。")
    with torch.device("meta"):
        expected = MOSSImageTransformer(**BASE_CONFIG).state_dict()
    validate_weights(path, expected, basename="diffusion_pytorch_model")
    return dict(BASE_CONFIG)


def load_transformer(path: Path, config: dict, *, device, dtype, checkpointing="none"):
    if dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("MOSS-Image 主模型计算精度请选择 BF16 或 FP32。")
    transformer_config(path)
    with torch.device("meta"):
        model = MOSSImageTransformer(**config, checkpointing=checkpointing)
    for file in weight_files(path, "diffusion_pytorch_model")[0]:
        model.load_state_dict(load_file(str(file), device="cpu"), strict=False, assign=True)
    return model.to(device=device, dtype=dtype).eval()
