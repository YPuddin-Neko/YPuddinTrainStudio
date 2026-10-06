"""Freeze a native-resolution memory budget before training state or caches are created."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from ypuddin.config import TrainConfig
from ypuddin.data.dataset import DataConfigError


def clear_native_vram_resolution(config: dict[str, Any]) -> dict[str, Any]:
    """Remove a previous run's resolved area from a new recipe without mutating it."""
    result = dict(config)
    dataset = config.get("dataset")
    if isinstance(dataset, dict) and "native_max_pixels_resolved" in dataset:
        result["dataset"] = {key: value for key, value in dataset.items() if key != "native_max_pixels_resolved"}
    return result


def _enabled(cfg: TrainConfig) -> bool:
    return cfg.dataset.resolution_mode == "native" and cfg.dataset.native_max_pixels_mode == "auto_vram"


def _pixels(value: Any) -> int | None:
    return value if type(value) is int and 0 < value <= 67_108_864 else None


def _with_pixels(cfg: TrainConfig, pixels: int) -> TrainConfig:
    resolved = cfg.model_copy(deep=True)
    resolved.dataset.native_max_pixels_resolved = pixels
    return resolved


def _resume_pixels(cfg: TrainConfig) -> int:
    checkpoint = Path(cfg.checkpoint.resume).expanduser()
    try:
        metadata = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataConfigError("checkpoint.resume", f"无法读取原训练的图像面积：{error}") from error
    extra = metadata.get("progress", {}).get("extra", {})
    candidates = [extra.get("native_max_pixels")]
    for data in (metadata.get("data_plan"), extra.get("data_plan")):
        if isinstance(data, dict):
            candidates.append((data.get("native") or {}).get("max_pixels"))
    recorded = next((value for value in candidates if _pixels(value) is not None), None)
    if recorded is None:
        from ypuddin.config.io import read_config_file

        # Checkpoints normally carry the area directly. Older exported records
        # may instead keep the run's config or data plan beside the state directory.
        for directory in (checkpoint, checkpoint.parent):
            config_file = directory / "config.toml"
            if config_file.is_file():
                value = read_config_file(config_file).get("dataset", {}).get("native_max_pixels_resolved")
                if _pixels(value) is not None:
                    recorded = value
                    break
            data_file = directory / "data-plan.json"
            if data_file.is_file():
                value = json.loads(data_file.read_text(encoding="utf-8")).get("native", {}).get("max_pixels")
                if _pixels(value) is not None:
                    recorded = value
                    break
    supplied = cfg.dataset.native_max_pixels_resolved
    if recorded is not None and supplied is not None and recorded != supplied:
        raise DataConfigError("checkpoint.resume", "图像面积与原训练不同，不能精确恢复；请沿用原训练面积或开始新任务。")
    if recorded is None and supplied is None:
        raise DataConfigError("checkpoint.resume", "原训练未保存显存优先的图像面积，无法确定恢复尺寸；请使用原任务配置。")
    return recorded if recorded is not None else supplied


def device_capacity_mb(device: str | torch.device) -> float | None:
    """Maximum working set of the runtime device, excluding reserved system memory on MPS."""
    target = torch.device(device)
    if target.type == "cuda":
        return torch.cuda.get_device_properties(target).total_memory / 2**20
    if target.type == "mps":
        import psutil

        from .memory_budget import mps_memory_capacity_mb

        return mps_memory_capacity_mb(psutil.virtual_memory().total / 2**20)
    return None


def device_memory_budget_mb(device: str | torch.device, capacity_mb: float) -> float | None:
    """Measure a fresh run's available budget once, before freezing its area."""
    from .memory_budget import available_memory_budget

    target = torch.device(device)
    free = None
    if target.type == "cuda":
        free = torch.cuda.mem_get_info(target)[0] / 2**20
    elif target.type == "mps":
        import psutil

        free = psutil.virtual_memory().available / 2**20
    return available_memory_budget({"mem_total_mb": capacity_mb, "mem_free_mb": free})


def resolve_native_vram_config(
    cfg: TrainConfig,
    *,
    device: str | torch.device,
    gpu_total_mb: float | None = None,
    gpu_memory_budget_mb: float | None = None,
    plan_result: dict[str, Any] | None = None,
) -> TrainConfig:
    """Resolve once for a fresh run; explicit snapshots and resumed runs keep their geometry."""
    if not _enabled(cfg):
        return cfg
    if cfg.checkpoint.resume:
        return _with_pixels(cfg, _resume_pixels(cfg))
    if cfg.dataset.native_max_pixels_resolved is not None:
        return cfg
    if plan_result is None:
        if gpu_total_mb is None:
            gpu_total_mb = device_capacity_mb(device)
        if gpu_total_mb is not None and gpu_memory_budget_mb is None:
            gpu_memory_budget_mb = device_memory_budget_mb(device, gpu_total_mb)
        if not isinstance(gpu_total_mb, (int, float)) or not math.isfinite(gpu_total_mb) or gpu_total_mb <= 0:
            reason = "当前设备使用 CPU，无法按显存计算面积上限" if torch.device(device).type == "cpu" else "无法读取目标设备的显存容量"
            raise DataConfigError("dataset.native_max_pixels_mode", f"{reason}；请改用自动分辨率优先或自定义。")
        from .plan import plan

        plan_result = plan(cfg, device=device, gpu_total_mb=gpu_total_mb, gpu_memory_budget_mb=gpu_memory_budget_mb)
    native = plan_result.get("native") or {}
    if error := native.get("auto_vram_error"):
        raise DataConfigError("dataset.native_max_pixels_mode", str(error))
    if plan_result.get("ok") is False:
        issue = next(iter(plan_result.get("errors") or []), {})
        raise DataConfigError(issue.get("loc", "dataset.native_max_pixels_mode"), issue.get("msg", "无法确定显存优先的图像面积。"))
    pixels = _pixels(native.get("auto_vram_max_pixels"))
    if pixels is None:
        raise DataConfigError("dataset.native_max_pixels_mode", "无法确定显存优先的图像面积，请先检查训练配置。")
    return _with_pixels(cfg, pixels)


def resolve_distributed_native_vram_config(
    cfg: TrainConfig, *, device: torch.device, rank: int, world_size: int
) -> TrainConfig:
    """Resolve on rank zero against the smallest participating device, then share its area."""
    if not _enabled(cfg):
        return cfg
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("distributed native resolution requires an initialized process group")
    needs_budget = not cfg.checkpoint.resume and cfg.dataset.native_max_pixels_resolved is None
    local: dict[str, Any] = {"capacity_mb": None, "budget_mb": None, "error": None}
    if needs_budget:
        try:
            local["capacity_mb"] = device_capacity_mb(device)
            local["budget_mb"] = device_memory_budget_mb(device, local["capacity_mb"])
        except Exception as error:
            local["error"] = str(error)
    readings = [None] * world_size
    dist.all_gather_object(readings, local)
    result: list[Any] = [None]
    if rank == 0:
        try:
            if any(reading["error"] for reading in readings):
                raise ValueError("；".join(reading["error"] for reading in readings if reading["error"]))
            capacities = [reading["capacity_mb"] for reading in readings]
            budgets = [reading.get("budget_mb") for reading in readings]
            if needs_budget and any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in capacities
            ):
                raise ValueError("无法读取所有参与训练设备的显存容量，请使用手动图像面积上限。")
            planning_cfg = cfg.model_copy(update={"loop": cfg.loop.model_copy(update={"gpu_count": world_size})})
            resolved = resolve_native_vram_config(
                planning_cfg, device=device, gpu_total_mb=min(capacities) if needs_budget else None,
                gpu_memory_budget_mb=min(budgets) if needs_budget and all(value is not None for value in budgets) else None,
            )
            result[0] = {"pixels": resolved.dataset.native_max_pixels_resolved}
        except Exception as error:
            result[0] = {"error": str(error), "loc": getattr(error, "loc", "dataset.native_max_pixels_mode")}
    dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise DataConfigError(result[0]["loc"], result[0]["error"])
    return _with_pixels(cfg, result[0]["pixels"])
