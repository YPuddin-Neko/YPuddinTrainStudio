"""Shared capacity budget for resolution planning and queue admission."""

import math
import subprocess
import sys
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

HEADROOM = 0.95
IDLE_SHARE = 0.97


@lru_cache(maxsize=1)
def _isolated_mps_recommended_memory_mb() -> float | None:
    """Read Metal's working-set limit without initializing the HTTP process's allocator."""
    code = "import torch; print(torch.mps.recommended_max_memory() / 2**20)"
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def mps_memory_capacity_mb(system_total_mb: float, *, isolated: bool = False) -> float:
    """Metal's recommended working set leaves room for macOS and other applications."""
    import torch

    try:
        recommended = (
            _isolated_mps_recommended_memory_mb()
            if isolated
            else torch.mps.recommended_max_memory() / 2**20
        )
        if recommended is not None and math.isfinite(recommended) and recommended > 0:
            return min(system_total_mb, recommended)
    except (AttributeError, RuntimeError):
        pass
    return system_total_mb * 0.75


def device_capacity(device: Mapping[str, Any]) -> float | None:
    """Use the recommended GPU working set on unified-memory devices."""
    values = [device.get("mem_total_mb"), device.get("memory_capacity_mb")]
    valid = [float(value) for value in values if type(value) in (int, float) and math.isfinite(value) and value > 0]
    return min(valid) if valid else None


def available_memory_budget(device: Mapping[str, Any]) -> float | None:
    """The exact peak an idle or currently occupied device can admit."""
    total = device_capacity(device)
    free = device.get("mem_free_mb")
    if type(free) not in (int, float) or not math.isfinite(free):
        return total * HEADROOM if total is not None else None
    free = max(0.0, free)
    if total is not None:
        free = total if free >= total * IDLE_SHARE else min(free, total)
    return free * HEADROOM


def host_memory_error(estimate_mb: float | None, available_mb: float) -> dict[str, str] | None:
    """Offloaded optimizer/EMA storage must fit host RAM as well as device memory."""
    if not estimate_mb or estimate_mb <= available_mb * HEADROOM:
        return None
    return {
        "loc": "memory.host",
        "msg": (
            f"预计 CPU 优化器和 EMA 需要 {estimate_mb / 1024:.1f} GB 内存，"
            f"当前可用 {available_mb / 1024:.1f} GB；请减少训练参数、训练显卡数或关闭 CPU 卸载。"
        ),
    }
