"""Freeze the training and validation seeds before constructing datasets and generators."""

from __future__ import annotations

import secrets
from pathlib import Path

from ypuddin.config import TrainConfig
from ypuddin.config.io import config_hash_variants, load_config

PREVIEW_SEED_VERSION = 3


def _preview_version(extra: dict) -> int:
    version = extra.get("preview_seed_version", 1)
    if type(version) is not int or version not in (1, 2, PREVIEW_SEED_VERSION):
        raise ValueError("恢复点的预览种子版本无效。")
    return version


def _legacy_preview_defaults(cfg: TrainConfig, version: int) -> None:
    if version < PREVIEW_SEED_VERSION:
        if version == 1 and "seed" not in cfg.sampling.model_fields_set:
            cfg.sampling.seed = 0
        for prompt in cfg.sampling.prompts:
            if "seed" not in prompt.model_fields_set:
                prompt.seed = None


def load_checkpoint_recipe(path: Path, metadata: dict, *, overrides=()) -> TrainConfig:
    """Interpret omitted fields in the checkpoint's recipe using its original defaults."""
    cfg = load_config(path, overrides=overrides)
    version = _preview_version(metadata.get("progress", {}).get("extra", {}))
    _legacy_preview_defaults(cfg, version)
    if version < PREVIEW_SEED_VERSION:
        for section, default in ((cfg.loop, 42), (cfg.validation, 1234)):
            if "seed" not in section.model_fields_set:
                section.seed = default
    return cfg


def _legacy_seeds(checkpoint: Path, metadata: dict, cfg: TrainConfig) -> dict[str, int]:
    if cfg.dataset.resolution_mode == "native":
        from ypuddin.train.native_batch_contract import authenticated_native_recipe

        original = authenticated_native_recipe(checkpoint, metadata=metadata)
        if original is not None:
            return {"loop": original.loop.seed, "validation": original.validation.seed}
    digest = metadata.get("config_hash")
    for folder in (checkpoint, checkpoint.parent, checkpoint.parent.parent):
        path = folder / "config.toml"
        if path.is_file():
            try:
                original = load_checkpoint_recipe(path, metadata)
            except (OSError, ValueError):
                continue
            if digest in config_hash_variants(original):
                return {"loop": original.loop.seed, "validation": original.validation.seed}
    raise ValueError("恢复点缺少原训练或验证种子，请使用原任务的配置和 config.toml。")


def resolve_training_seeds(cfg: TrainConfig, *, planning: bool = False) -> TrainConfig:
    """Return a private execution recipe. A fresh estimate uses a fixed representative order."""
    sections = ("loop", "validation")
    pending = [name for name in sections if getattr(cfg, name).seed == -1]
    if not pending and not cfg.checkpoint.resume:
        return cfg
    resolved = cfg.model_copy(deep=True)
    frozen = {}
    if cfg.checkpoint.resume:
        from ypuddin.train.native_batch_contract import read_native_resume_metadata

        checkpoint = Path(cfg.checkpoint.resume).expanduser()
        meta = read_native_resume_metadata(checkpoint)
        extra = meta.get("progress", {}).get("extra", {})
        _legacy_preview_defaults(resolved, _preview_version(extra))
        if "training_seeds" in extra:
            frozen = extra["training_seeds"]
            for name in sections:
                value = frozen.get(name) if isinstance(frozen, dict) else None
                requested = getattr(cfg, name).seed
                if type(value) is not int or value < 0 or requested not in (-1, value):
                    raise ValueError(f"{name}.seed 与恢复点的记录不符，请沿用原任务的种子。")
        elif pending:
            frozen = _legacy_seeds(checkpoint, meta, cfg)
    for name in pending:
        if cfg.checkpoint.resume:
            value = frozen.get(name) if isinstance(frozen, dict) else None
            if type(value) is not int or value < 0:
                raise ValueError(f"恢复点缺少有效的 {name}.seed，请使用原任务的种子。")
        else:
            value = 0 if planning else secrets.randbelow(2**32)
        getattr(resolved, name).seed = value
    return resolved
