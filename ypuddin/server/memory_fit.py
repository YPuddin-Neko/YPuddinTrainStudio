"""Whether a run's estimated peak memory fits the machine's accelerators.

The planner's peak is a heuristic, so admission keeps 5% headroom. A run that
exceeds even an idle device's capacity can never start, and must say so instead
of waiting in the queue.
"""

from __future__ import annotations

from typing import Any

from ypuddin.train.memory_budget import HEADROOM, available_memory_budget, device_capacity
from ypuddin.train.memory_budget import host_memory_error as host_memory_error


def gb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB"


def device_label(device: str) -> str:
    return f"GPU {device.split(':', 1)[1]}" if device.startswith("cuda:") else device.upper()


def capacity_shortfall(
    estimate_mb: float | None, devices: list[dict[str, Any]], count: int = 1
) -> dict[str, Any] | None:
    """The device that limits a run whose estimate cannot fit; None when it fits or is unknown."""
    if not estimate_mb:
        return None
    sized = sorted(
        (
            device
            for device in devices
            if device_capacity(device) is not None
        ),
        key=device_capacity,
        reverse=True,
    )
    if len(sized) < max(1, count):
        return None  # Missing devices are reported by the device-selection checks.
    limit = sized[max(1, count) - 1]
    capacity = device_capacity(limit)
    if estimate_mb <= capacity * HEADROOM:
        return None
    return {"estimate_mb": estimate_mb, "capacity_mb": capacity, "device": limit["device"]}


def shortfall_error(shortfall: dict[str, Any]) -> dict[str, str]:
    """A plan error; the interface words it for the reader's language."""
    return {
        "loc": "memory",
        "msg": (
            f"estimated peak memory {gb(shortfall['estimate_mb'])} exceeds "
            f"{device_label(shortfall['device'])} capacity {gb(shortfall['capacity_mb'])}"
        ),
    }


def shortfall_reason(shortfall: dict[str, Any]) -> str:
    """Why a queued run cannot start on this machine."""
    return (
        f"预计显存峰值 {gb(shortfall['estimate_mb'])}，超过 {device_label(shortfall['device'])} 的 "
        f"{gb(shortfall['capacity_mb'])} 容量，显卡空闲时也无法启动。"
        "开启梯度检查点、减小批大小或降低训练尺寸后可重新训练。"
    )


def fits_now(estimate_mb: float | None, device: dict[str, Any]) -> bool:
    """Whether the run can start on this device at the moment."""
    budget = available_memory_budget(device)
    return not estimate_mb or budget is None or estimate_mb <= budget
