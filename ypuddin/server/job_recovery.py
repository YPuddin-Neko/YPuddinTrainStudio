"""Inspect published image checkpoints without loading training tensors or pickle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .job_paths import job_config, state_directory

_METADATA_LIMIT = 32 * 1024 * 1024
_NO_STATE = "没有完整的训练恢复点，请重试以创建新任务。"


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{path.name} 不能是符号链接")
    with path.open("rb") as stream:
        data = stream.read(_METADATA_LIMIT + 1)
    if len(data) > _METADATA_LIMIT:
        raise ValueError(f"{path.name} 过大")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 格式无效")
    return value


def _tensor_file(path: Path) -> None:
    from safetensors import safe_open

    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
    if not 2 <= header_size <= _METADATA_LIMIT:
        raise ValueError(f"{path.name} 文件头无效")
    # The native reader checks offsets against file length without reading tensors.
    with safe_open(str(path), framework="np"):
        pass


def _torch_file(path: Path) -> None:
    # torch.save publishes a ZIP container. Reading its directory detects an
    # interrupted write without executing pickle or scanning optimizer storage.
    with ZipFile(path) as archive:
        entries = archive.infolist()
        names = {item.filename for item in entries}
        roots = {name.rsplit("/", 1)[0] for name in names if name.endswith("/data.pkl")}
        if len(names) != len(entries) or len(roots) != 1:
            raise ValueError(f"{path.name} 容器无效")
        root = roots.pop()
        if f"{root}/version" not in names or archive.getinfo(f"{root}/data.pkl").file_size == 0:
            raise ValueError(f"{path.name} 容器不完整")
        if any(item.header_offset < 0 or item.header_offset + item.compress_size > archive.start_dir for item in entries):
            raise ValueError(f"{path.name} 容器被截断")


def checkpoint_problem(path: Path, config: dict[str, Any] | None = None) -> str | None:
    """Check file structure; Trainer still validates payloads and exact-resume contracts."""
    try:
        if path.is_symlink() or not path.is_dir() or path.name.startswith(".") or ".tmp" in path.name:
            return "恢复点目录不存在或尚未写入完成。"
        meta = _json(path / "state.json")
        version = meta.get("format", 1)
        if type(version) is not int or version not in (1, 2, 3, 4):
            return "恢复点格式不受支持。"
        progress, sampler = meta.get("progress"), meta.get("sampler")
        if not isinstance(progress, dict) or not isinstance(sampler, dict):
            return "恢复点缺少训练进度或采样状态。"
        for key in ("step", "epoch", "batch_in_epoch", "samples_seen", "nan_skips", "total_steps", "steps_per_epoch"):
            value = progress.get(key, 0)
            if type(value) is not int or value < 0:
                return "恢复点的训练进度无效。"
        if not isinstance(progress.get("extra", {}), dict):
            return "恢复点的训练进度无效。"
        kind = meta.get("training_kind", "adapter")
        if kind not in ("adapter", "full-model") or (
            version != 4 and (version == 3) != (kind == "full-model")
        ):
            return "恢复点的训练类型无效。"
        required = ["optimizer.pt", "scheduler.pt", "rng.pt"]
        required.append("model.safetensors" if version in (3, 4) else "adapter.safetensors")
        if version == 2:
            required.append("training.safetensors")
        if (path / "ema.safetensors").exists():
            required.append("ema.safetensors")
        if version == 4:
            if meta.get("strategy") != "fsdp2" or _json(path / "complete.json").get("format") != 4:
                return "分片恢复点缺少有效的完成标记。"
            world = meta.get("world_size")
            ranks = meta.get("sampler_ranks")
            if (type(world) is not int or world < 1 or not isinstance(ranks, list)
                    or len(ranks) != world or any(not isinstance(item, dict) for item in ranks)):
                return "分片恢复点缺少各显卡的采样状态。"
            if not isinstance(meta.get("model_layout"), dict) or not meta["model_layout"]:
                return "分片恢复点缺少模型布局。"
        if config is not None:
            loop = config.get("loop") or {}
            expected_kind = "full-model" if (config.get("training") or {}).get("mode") == "full" else "adapter"
            if kind != expected_kind:
                return "恢复点的训练类型与任务配置不同。"
            if (version == 4) != (loop.get("distributed_strategy", "ddp") == "fsdp"):
                return "恢复点的分布方式与任务配置不同。"
            if version == 4 and any(meta.get(key) != expected for key, expected in (
                ("world_size", loop.get("gpu_count", 1)),
                ("grad_accum", loop.get("grad_accum", 1)),
                ("batch_size", (config.get("dataset") or {}).get("batch_size", 1)),
            )):
                return "恢复点的显卡数量、批量或梯度累积与任务配置不同。"
        for name in required:
            item = path / name
            if item.is_symlink() or not item.is_file() or item.stat().st_size == 0:
                return f"恢复点缺少完整的 {name}。"
            try:
                (_tensor_file if name.endswith(".safetensors") else _torch_file)(item)
            except Exception:  # noqa: BLE001 - malformed files must not break the job list
                return f"恢复点的 {name} 不完整或已损坏。"
        return None
    except (OSError, ValueError, TypeError, BadZipFile):
        return "恢复点元数据缺失、损坏或无法读取。"


def latest_complete_state(directory: Path, config: dict[str, Any] | None = None) -> str | None:
    """Ignore unpublished/partial states and retain the latest complete checkpoint."""
    try:
        candidates = sorted(directory.glob("state-*"), key=lambda item: item.stat().st_mtime_ns, reverse=True)
        return next((str(path) for path in candidates if checkpoint_problem(path, config) is None), None)
    except OSError:
        return None


def image_resume_state(job: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return a checkpoint path or the reason this interrupted job cannot resume."""
    if job.get("type") != "train":
        return None, "此任务不支持继续训练，请重试以创建新任务。"
    try:
        config = job_config(job)
        if job.get("resume_from"):
            path = Path(job["resume_from"])
            problem = checkpoint_problem(path, config)
            return (None, problem) if problem else (str(path), None)
        latest = latest_complete_state(state_directory(job), config)
        if latest:
            return latest, None
        initial = (config.get("checkpoint") or {}).get("resume")
        if initial:
            problem = checkpoint_problem(Path(initial), config)
            return (None, problem) if problem else (str(initial), None)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None, "任务的冻结配置无法读取。"
    return None, _NO_STATE


def recovery(job: dict[str, Any], *, worker_alive: bool = False) -> dict[str, Any]:
    """Project the checkpoint that the resume action will recheck before queueing."""
    result = {"can_resume": False, "checkpoint_path": None, "step": None,
              "reason": None, "worker_alive": worker_alive}
    if worker_alive:
        result["reason"] = "原任务进程尚未退出，不能再次启动。"
        return result
    path, reason = image_resume_state(job)
    if path is not None:
        try:
            step = _json(Path(path) / "state.json")["progress"].get("step", 0)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            result["reason"] = "恢复点元数据缺失、损坏或无法读取。"
            return result
        result.update(can_resume=True, checkpoint_path=path, step=step)
    else:
        result["reason"] = reason
    return result
