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
        return parse_toml(text)
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
        # Full-model exports carry native components next to this config. Rebind
        # only those generated component paths after copying/unpacking an artifact;
        # frozen VAE/tokenizer references retain their explicit original locations.
        manifest_path = Path(path).parent / "manifest.json"
        if Path(path).name == "config.toml" and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, dict) and manifest.get("format") == "ypuddin-full-model-v1":
                from .model_artifact import rebind_artifact_components

                data = rebind_artifact_components(data, manifest, Path(path).parent)

    patch = overrides if isinstance(overrides, Mapping) else parse_overrides(overrides)
    data = deep_merge(data, patch)
    return TrainConfig.model_validate(data)


# Fields added after checkpoints already existed, at the value those checkpoints trained with. At that
# value a field is left out of the hash, so older checkpoints still authenticate their config.
_LEGACY_VALUES: dict[str, dict[str, Any]] = {
    # Optional export and save settings, off.
    "checkpoint": {"save_state_every_epochs": None, "save_training_metadata": False, "state_dir": None},
    "logging": {"output_dir": None},
    # Center keeps the pixel geometry of older checkpoints.
    "dataset": {"crop_anchor": "center"},
    # SDXL's original single CLIP context and cache behavior.
    "model": {"sdxl_max_token_length": 75},
    # Without the switch, runs used nondeterministic kernels; true stays visible.
    "loop": {"deterministic": False},
    # Optional DDPM loss modifiers, off.
    "objective": {
        "scale_v_pred_loss_like_noise_pred": False,
        "v_pred_like_loss": 0.0,
        "debiased_estimation_loss": False,
    },
    # Linear layers only, with no separate convolution rank.
    "adapter": {"layer_types": "linear", "conv_rank": None, "conv_alpha": None},
    # Preview noise drawn from the seed on the CPU.
    "sampling": {"noise": "comfyui"},
}


def config_hash(config: TrainConfig | Mapping[str, Any]) -> str:
    data = config.to_dict() if isinstance(config, TrainConfig) else dict(config)
    for section, legacy in _LEGACY_VALUES.items():
        values = data.get(section)
        if isinstance(values, Mapping):
            data[section] = {
                key: value for key, value in values.items() if key not in legacy or value != legacy[key]
            }
    # Checkpoints from before the DoRA axis option trained the output axis; without DoRA it does nothing.
    adapter = data.get("adapter")
    if isinstance(adapter, Mapping) and (adapter.get("dora_axis") == "output" or not adapter.get("dora")):
        data["adapter"] = {key: value for key, value in adapter.items() if key != "dora_axis"}
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def absolute_paths(config: TrainConfig) -> TrainConfig:
    """Freeze local paths against the submitting process cwd before a service job changes cwd."""
    cfg = config.model_copy(deep=True)
    fields = [
        (cfg.model, "dit_path"),
        (cfg.model, "text_encoder_path"),
        (cfg.model, "text_encoder_2_path"),
        (cfg.model, "vae_path"),
        (cfg.model, "tokenizer_path"),
        (cfg.dataset, "cache_dir"),
        (cfg.adapter, "resume_weights"),
        (cfg.training, "resume_weights"),
        (cfg.checkpoint, "output_dir"),
        (cfg.checkpoint, "state_dir"),
        (cfg.checkpoint, "resume"),
        (cfg.sampling, "prompts_file"),
        (cfg.sampling, "output_dir"),
        (cfg.logging, "events_path"),
        (cfg.logging, "output_dir"),
        *((src, "path") for src in cfg.dataset.sources),
        *((src, "path") for src in cfg.validation.sources),
    ]
    for obj, name in fields:
        value = getattr(obj, name)
        if value:
            setattr(obj, name, str(Path(value).expanduser().resolve()))
    return cfg


_NULL_PATHS_KEY = "__ypuddin_nulls__"


def parse_toml(text: str) -> dict[str, Any]:
    """Read ordinary TOML and restore explicit nulls from our lossless export metadata.

    A missing setting in an ordinary/legacy TOML file still uses its schema default.
    Values explicitly added to an exported table take precedence over stale null metadata.
    """
    data = tomllib.loads(text)
    paths = data.pop(_NULL_PATHS_KEY, [])
    if not isinstance(paths, list):
        raise ValueError(f"{_NULL_PATHS_KEY} must be an array of key/index paths")
    for path in paths:
        if not isinstance(path, list) or not path or any(type(part) not in (str, int) for part in path):
            raise ValueError(f"invalid explicit-null path: {path!r}")
        current: Any = data
        for part in path[:-1]:
            if isinstance(current, dict) and isinstance(part, str) and part in current:
                current = current[part]
            elif isinstance(current, list) and type(part) is int and 0 <= part < len(current):
                current = current[part]
            else:
                raise ValueError(f"explicit-null metadata refers to a missing parent: {path!r}")
        final = path[-1]
        if isinstance(current, dict) and isinstance(final, str):
            current.setdefault(final, None)
        elif isinstance(current, list) and type(final) is int and 0 <= final < len(current):
            # TOML cannot represent a null array element. The exporter leaves an empty
            # string placeholder; an explicitly edited nonempty value takes precedence.
            if current[final] == "":
                current[final] = None
        else:
            raise ValueError(f"explicit-null metadata refers to an invalid target: {path!r}")
    return data


def dump_toml(config: TrainConfig) -> str:
    paths: list[list[str | int]] = []

    def pack(obj: Any, path: list[str | int]) -> Any:
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if value is None:
                    paths.append([*path, key])
                else:
                    result[key] = pack(value, [*path, key])
            return result
        if isinstance(obj, list):
            result = []
            for index, value in enumerate(obj):
                if value is None:
                    paths.append([*path, index])
                    result.append("")
                else:
                    result.append(pack(value, [*path, index]))
            return result
        return obj

    values = pack(config.to_dict(), [])
    if paths:
        values = {_NULL_PATHS_KEY: paths, **values}
    return "# Explicit nulls are preserved below because TOML has no null value.\n" + tomli_w.dumps(values)


def write_config(config: TrainConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        p.write_text(dump_toml(config), encoding="utf-8")
