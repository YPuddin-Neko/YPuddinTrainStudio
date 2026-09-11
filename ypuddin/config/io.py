"""Loading, merging, hashing and dumping of :class:`TrainConfig`."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import tomli_w

from .schema import TrainConfig

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``patch`` onto ``base`` (dicts merge, everything else replaces)."""
    out: dict[str, Any] = dict(base)
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    if p.suffix.lower() == ".toml":
        return tomllib.loads(text)
    raise ValueError(f"unsupported config format: {p.suffix} (use .toml or .json)")


def _coerce_scalar(text: str) -> Any:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text[:1] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn ``["loop.epochs=5", "adapter.rank=full"]`` into a nested dict."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        cur = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = _coerce_scalar(value.strip())
    return out


def load_config(
    path: str | Path | None = None,
    *,
    presets: Iterable[str | Path | Mapping[str, Any]] = (),
    overrides: Iterable[str] | Mapping[str, Any] = (),
    base: Mapping[str, Any] | None = None,
) -> TrainConfig:
    """Resolve ``base <- presets... <- file <- overrides`` into a validated config."""
    data: dict[str, Any] = dict(base or {})
    for preset in presets:
        patch = preset if isinstance(preset, Mapping) else read_config_file(preset)
        data = deep_merge(data, patch)
    if path is not None:
        data = deep_merge(data, read_config_file(path))
    patch = overrides if isinstance(overrides, Mapping) else parse_overrides(overrides)
    data = deep_merge(data, patch)
    return TrainConfig.model_validate(data)


def config_hash(config: TrainConfig | Mapping[str, Any]) -> str:
    data = config.to_dict() if isinstance(config, TrainConfig) else dict(config)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def absolute_paths(config: TrainConfig) -> TrainConfig:
    """Freeze local paths against the submitting process cwd before a service job changes cwd."""
    cfg = config.model_copy(deep=True)
    fields = [
        (cfg.model, "dit_path"),
        (cfg.model, "text_encoder_path"),
        (cfg.model, "vae_path"),
        (cfg.model, "tokenizer_path"),
        (cfg.dataset, "cache_dir"),
        (cfg.adapter, "resume_weights"),
        (cfg.checkpoint, "output_dir"),
        (cfg.checkpoint, "resume"),
        (cfg.sampling, "prompts_file"),
        (cfg.logging, "events_path"),
        *((src, "path") for src in cfg.dataset.sources),
        *((src, "path") for src in cfg.validation.sources),
    ]
    for obj, name in fields:
        value = getattr(obj, name)
        if value:
            setattr(obj, name, str(Path(value).expanduser().resolve()))
    return cfg


def _strip_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def dump_toml(config: TrainConfig) -> str:
    return tomli_w.dumps(_strip_none(config.to_dict()))


def write_config(config: TrainConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        p.write_text(dump_toml(config), encoding="utf-8")
