"""Version the native sampler and its optimizer-update boundaries."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig
from ypuddin.config.io import config_hash_variants, load_config, write_config

STATE_KEY = "native_batch_contract"
CURRENT_VERSION = 3
SUPPORTED_VERSIONS = frozenset({1, 2, 3})


def _mode(version: int) -> str:
    return "physical" if version == 2 else "area"


def native_batch_contract(cfg: TrainConfig, *, world_size: int | None = None) -> dict[str, Any] | None:
    if cfg.dataset.resolution_mode != "native":
        return None
    world = cfg.loop.gpu_count if world_size is None else world_size
    contract = {
        "version": cfg.dataset.native_batching_version,
        "batch_size": cfg.dataset.batch_size,
        "grad_accum": cfg.loop.grad_accum,
        "seed": cfg.loop.seed,
        "world_size": world,
        "distributed_strategy": cfg.loop.distributed_strategy if world > 1 else "single",
    }
    if cfg.dataset.native_batching_version == 3:
        contract["mode"] = "area"
    return contract


def _recipe_mode(config: TrainConfig, version: int) -> TrainConfig | None:
    """Only an absent selection may inherit the authenticated historical strategy."""
    if "native_batch_mode" in config.dataset.model_fields_set and config.dataset.native_batch_mode != _mode(version):
        return None
    config.dataset.native_batch_mode = _mode(version)
    return config


def read_native_resume_metadata(path: str | Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reject malformed containers before any checkpoint identity is interpreted."""
    if metadata is None:
        metadata = json.loads((Path(path) / "state.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("恢复点元数据必须是对象，无法读取训练状态。")
    progress = metadata.get("progress", {})
    if not isinstance(progress, dict) or not isinstance(progress.get("extra", {}), dict):
        raise ValueError("恢复点的 progress 或 extra 记录无效，无法读取训练状态。")
    if metadata.get("config_hash") is not None and not isinstance(metadata["config_hash"], str):
        raise ValueError("恢复点的配置哈希记录无效，无法核对训练配置。")
    return metadata


def _archived_recipe(folder: Path, metadata: dict[str, Any]) -> Path | None:
    saved_hash = metadata.get("config_hash")
    if not isinstance(saved_hash, str) or len(saved_hash) != 16 or any(c not in "0123456789abcdef" for c in saved_hash):
        return None
    return folder / ".resume-recipes" / f"{saved_hash}.toml"


def authenticated_native_recipe(
    path: str | Path, *, metadata: dict[str, Any] | None = None, config: TrainConfig | None = None,
) -> TrainConfig | None:
    """Read only a native recipe whose original hash matches the frozen checkpoint."""
    path = Path(path)
    metadata = read_native_resume_metadata(path, metadata)
    extra = metadata.get("progress", {}).get("extra", {})
    contract = extra.get(STATE_KEY)
    if STATE_KEY in extra and not isinstance(contract, dict):
        return None
    version = contract.get("version") if isinstance(contract, dict) else 1
    if type(version) is not int or version not in SUPPORTED_VERSIONS:
        return None
    original = None
    sources = [source for folder in (path, path.parent, path.parent.parent)
               for source in (folder / "config.toml", _archived_recipe(folder, metadata))]
    for source in sources:
        if source is None or not source.is_file():
            continue
        try:
            candidate = load_config(source, overrides={"dataset": {"native_batching_version": version}})
        except (OSError, ValueError):
            continue
        candidate = _recipe_mode(candidate, version)
        if candidate is None:
            continue
        if metadata.get("config_hash") in config_hash_variants(candidate):
            original = candidate
            break
    if original is None and config is not None:
        candidate = config.model_copy(deep=True)
        candidate.dataset.native_batching_version = version
        candidate.checkpoint.resume = None
        candidate = _recipe_mode(candidate, version)
        if candidate is not None and metadata.get("config_hash") in config_hash_variants(candidate):
            original = candidate
    if original is not None and original.dataset.resolution_mode == "native":
        return original
    return None


def preserve_native_resume_recipe(cfg: TrainConfig, record_dir: Path) -> None:
    """Keep the authenticated recipe before a same-directory resume replaces config.toml."""
    if cfg.dataset.resolution_mode != "native" or not cfg.checkpoint.resume:
        return
    metadata = read_native_resume_metadata(cfg.checkpoint.resume)
    original = authenticated_native_recipe(cfg.checkpoint.resume, metadata=metadata, config=cfg)
    target = _archived_recipe(record_dir, metadata)
    if original is None or target is None:
        return
    if target.is_file():
        saved = load_config(target)
        if metadata["config_hash"] not in config_hash_variants(saved):
            raise ValueError("已保留的恢复配置与检查点哈希不同，不能覆盖。")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".toml")
    os.close(descriptor)
    try:
        write_config(original, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_native_batch_contract(
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    require_native: bool = False,
    config: TrainConfig | None = None,
) -> dict[str, Any] | None:
    """Recover old native boundaries only from the checkpoint's authenticated recipe."""
    path = Path(path)
    metadata = read_native_resume_metadata(path, metadata)
    extra = metadata.get("progress", {}).get("extra", {})
    if STATE_KEY in extra:
        return extra[STATE_KEY]
    if not require_native and "native_max_pixels" not in extra:
        return None
    original = authenticated_native_recipe(path, metadata=metadata, config=config)
    if original is None:
        raise ValueError(
            "旧原生恢复点缺少可核对的训练配置，无法确定原批量策略；"
            "请保留原训练目录中未修改的 config.toml，或从导出权重开始新任务。"
        )
    if original.dataset.resolution_mode != "native":
        return None
    world = metadata.get("world_size", original.loop.gpu_count)
    # Earlier torchrun recipes could leave gpu_count=1 while the actual process
    # group was larger. Its checkpointed RNG envelope records the real topology.
    rng_path = path / "rng.pt"
    if rng_path.is_file() and "world_size" not in metadata:
        import torch

        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        distributed = rng.get("distributed")
        world = distributed.get("world_size") if isinstance(distributed, dict) else 1
    if type(world) is not int or world < 1:
        raise ValueError("旧原生恢复点的显卡数量记录无效，不能精确恢复。")
    return native_batch_contract(original, world_size=world)


def resolve_native_batching_config(cfg: TrainConfig, *, world_size: int | None = None) -> TrainConfig:
    """An omitted version inherits the frozen checkpoint, never the current default."""
    if not cfg.checkpoint.resume:
        if cfg.dataset.resolution_mode != "native":
            return cfg
        resolved = cfg.model_copy(deep=True)
        if cfg.dataset.native_batching_version == 1:
            # Already frozen legacy jobs retain their original sampler and update boundaries.
            if "native_batch_mode" in cfg.dataset.model_fields_set and cfg.dataset.native_batch_mode != "area":
                raise ValueError("旧原生任务使用按面积拆分，不能改用物理批次继续执行。")
            resolved.dataset.native_batch_mode = "area"
        else:
            resolved.dataset.native_batching_version = 3 if cfg.dataset.native_batch_mode == "area" else 2
        return resolved
    saved = read_native_batch_contract(
        cfg.checkpoint.resume, require_native=cfg.dataset.resolution_mode == "native", config=cfg
    )
    resolved = cfg
    if saved is not None and cfg.dataset.resolution_mode == "native":
        version = saved.get("version") if isinstance(saved, dict) else None
        if type(version) is not int or version not in SUPPORTED_VERSIONS:
            raise ValueError("恢复点的原生批量策略版本无效，不能精确恢复。")
        if (
            "native_batching_version" in cfg.dataset.model_fields_set
            and cfg.dataset.native_batching_version != version
        ):
            raise ValueError("原生批量策略版本与恢复点不同，不能精确恢复；请沿用原任务或从导出权重开始新任务。")
        if "native_batch_mode" in cfg.dataset.model_fields_set and cfg.dataset.native_batch_mode != _mode(version):
            raise ValueError("原生批次方式与恢复点不同，不能精确恢复；请沿用原任务或从导出权重开始新任务。")
        resolved = cfg.model_copy(deep=True)
        resolved.dataset.native_batching_version = version
        resolved.dataset.native_batch_mode = _mode(version)
    validate_native_batch_contract(native_batch_contract(resolved, world_size=world_size), saved)
    return resolved


def validate_native_batch_contract(expected: dict[str, Any] | None, saved: Any) -> None:
    if expected is None and saved is None:
        return
    if expected is None:
        raise ValueError("恢复点使用原生分辨率训练，不能改用分桶策略精确恢复；请沿用原设置或开始新任务。")
    if saved is None:
        raise ValueError(
            "恢复点未使用相同的原生批量策略，不能精确恢复；请沿用原任务或从导出权重开始新任务。"
        )
    if (
        not isinstance(saved, dict)
        or saved.keys() != expected.keys()
        or any(type(saved.get(key)) is not int for key in ("version", "batch_size", "grad_accum", "seed", "world_size"))
        or saved.get("version") not in SUPPORTED_VERSIONS
        or (saved.get("version") == 3 and saved.get("mode") != "area")
        or any(saved.get(key, 0) < 1 for key in ("batch_size", "grad_accum", "world_size"))
        or not isinstance(saved.get("distributed_strategy"), str)
    ):
        raise ValueError("恢复点的原生批量策略记录无效，不能精确恢复。")
    if saved != expected:
        raise ValueError(
            "原生批量策略、批量大小、梯度累积、随机种子或显卡分配与恢复点不同，"
            "不能精确恢复；请沿用原设置或从导出权重开始新任务。"
        )


def validate_native_batch_resume(cfg: TrainConfig, saved: Any, *, world_size: int | None = None) -> None:
    validate_native_batch_contract(native_batch_contract(cfg, world_size=world_size), saved)
