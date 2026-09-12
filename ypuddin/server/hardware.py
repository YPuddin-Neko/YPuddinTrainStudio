"""Accelerator inventory shared by system stats and single-device job scheduling."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import plistlib
import shutil
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.parsers.expat import ExpatError

import psutil
import torch

_smi_lock = threading.Lock()
_smi_cache: tuple[float, list[dict[str, Any]]] = (0, [])
_apple_gpu_lock = threading.Lock()
_apple_gpu_cache: tuple[float, float | None] | None = None
_apple_sensors_lock = threading.Lock()
_apple_sensors_cache: tuple[float, dict[str, Any]] | None = None


def _apple_gpu_sensors() -> dict[str, Any]:
    """Bound native sensor reads to a disposable process, shared for two seconds.

    IOReport/SMC are private interfaces. A missing symbol, native crash or hung
    driver must not crash the web server or prevent other telemetry from updating.
    The standard-library-only worker uses this interpreter, without sudo/helpers.
    """
    empty = {
        "power_w": None,
        "temp_c": None,
        "temp_max_c": None,
        "temp_sensor_count": 0,
        "power_sample_seconds": None,
    }
    if sys.platform != "darwin":
        return empty
    global _apple_sensors_cache
    with _apple_sensors_lock:
        if _apple_sensors_cache is not None and time.monotonic() - _apple_sensors_cache[0] < 2:
            return dict(_apple_sensors_cache[1])
        reading = dict(empty)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("apple_sensors.py")),
                    "--chip",
                    _apple_name(),
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            data = json.loads(result.stdout)
            if isinstance(data, dict):

                def valid(value: Any, low: float, high: float = math.inf) -> bool:
                    return (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and low <= value <= high
                    )

                interval = data.get("power_sample_seconds")
                if valid(data.get("power_w"), 0) and valid(interval, 0.01, 5):
                    reading["power_w"] = data["power_w"]
                    reading["power_sample_seconds"] = interval
                count = data.get("temp_sensor_count")
                mean, peak = data.get("temp_c"), data.get("temp_max_c")
                if (
                    type(count) is int
                    and 1 <= count <= 128
                    and valid(mean, 0.01, 150)
                    and valid(peak, 0.01, 150)
                    and peak >= mean
                ):
                    reading.update(temp_c=mean, temp_max_c=peak, temp_sensor_count=count)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, OverflowError):
            pass
        _apple_sensors_cache = (time.monotonic(), reading)
        return dict(reading)


def _apple_gpu_utilization() -> float | None:
    """Apple's driver percentage for the whole GPU, not the Python process.

    IORegistry is readable without sudo. The key is a driver-provided percentage,
    not a cumulative counter; its averaging window is unspecified. Cache failures
    as well as readings, and never carry an old reading across a failed probe.
    """
    if sys.platform != "darwin":
        return None
    global _apple_gpu_cache
    with _apple_gpu_lock:
        if _apple_gpu_cache is not None and time.monotonic() - _apple_gpu_cache[0] < 2:
            return _apple_gpu_cache[1]
        utilization = None
        try:
            result = subprocess.run(
                ["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-a"],
                capture_output=True,
                timeout=2,
                check=True,
            )
            entries = plistlib.loads(result.stdout)
            if isinstance(entries, list):
                # MPS does not identify a registry device. Only an unambiguous
                # Apple AGX accelerator can be assigned to its single GPU row.
                apple = [
                    entry
                    for entry in entries
                    if isinstance(entry, dict) and str(entry.get("IOClass", "")).startswith("AGXAccelerator")
                ]
                if len(apple) == 1:
                    stats = apple[0].get("PerformanceStatistics")
                    value = stats.get("Device Utilization %") if isinstance(stats, dict) else None
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and 0 <= value <= 100
                    ):
                        utilization = float(value)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, OverflowError, ExpatError):
            pass
        _apple_gpu_cache = (time.monotonic(), utilization)
        return utilization


def _number(value: str) -> float | None:
    try:
        number = float(value.strip())
        return number if math.isfinite(number) and number >= 0 else None
    except ValueError:
        return None


def _nvidia_smi() -> list[dict[str, Any]]:
    """Read driver telemetry even on Windows without the optional NVML Python binding.

    Cache the subprocess result briefly: scheduler and HTTP/SSE callers share this probe.
    A missing/unsupported field is independent of the other measurements.
    """
    global _smi_cache
    with _smi_lock:
        if time.monotonic() - _smi_cache[0] < 2:
            return _smi_cache[1]
        executable = shutil.which("nvidia-smi")
        if not executable and os.name == "nt":
            for candidate in (
                Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/nvidia-smi.exe",
                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                / "NVIDIA Corporation/NVSMI/nvidia-smi.exe",
            ):
                if candidate.is_file():
                    executable = str(candidate)
                    break
        rows: list[dict[str, Any]] = []
        if executable:
            try:
                result = subprocess.run(
                    [
                        executable,
                        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=2,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                for row in csv.reader(io.StringIO(result.stdout)):
                    if len(row) != 9 or not row[0].strip().isdigit():
                        continue
                    total, used, util, temp, power, limit = [_number(x) for x in row[3:]]
                    rows.append(
                        {
                            "index": int(row[0]),
                            "uuid": row[1].strip(),
                            "name": row[2].strip(),
                            "mem_total_mb": int(total) if total is not None else None,
                            "mem_used_mb": int(used) if used is not None else None,
                            "mem_free_mb": int(max(0, total - used))
                            if total is not None and used is not None
                            else None,
                            "util_pct": util,
                            "temp_c": temp,
                            "power_w": power,
                            "power_limit_w": limit,
                        }
                    )
            except (OSError, subprocess.SubprocessError):
                pass
        _smi_cache = (time.monotonic(), rows)
        return rows


def _nvml_metrics(entries: list[dict[str, Any]]) -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            for entry in entries:
                try:
                    if entry.get("uuid"):
                        handle = pynvml.nvmlDeviceGetHandleByUUID(entry["uuid"])
                    else:
                        matches = []
                        for physical_index in range(pynvml.nvmlDeviceGetCount()):
                            candidate = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                            name = pynvml.nvmlDeviceGetName(candidate)
                            if isinstance(name, bytes):
                                name = name.decode(errors="replace")
                            if name == entry["name"]:
                                matches.append(candidate)
                        if len(matches) != 1:
                            continue  # Do not display another GPU's watts when identity is ambiguous.
                        handle = matches[0]
                except Exception:  # noqa: BLE001
                    continue
                readers = {
                    "util_pct": lambda handle=handle: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    "temp_c": lambda handle=handle: pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power_w": lambda handle=handle: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    "power_limit_w": lambda handle=handle: (
                        pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
                    ),
                }
                for key, read in readers.items():
                    try:
                        value = read()
                        if value is not None and math.isfinite(value) and value >= 0:
                            entry[key] = value
                            entry["telemetry_source"] = "nvml"
                    except Exception:  # noqa: BLE001
                        continue
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass


@lru_cache(maxsize=1)
def _apple_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        name = result.stdout.strip()
        if name.startswith("Apple"):
            return name + " GPU"
    except (OSError, subprocess.SubprocessError):
        pass
    return "Apple Silicon GPU"


def gpu_info(*, include_unavailable: bool = False, system_memory: Any | None = None) -> list[dict[str, Any]]:
    """Read accelerators, optionally reusing the caller's system-memory snapshot for MPS."""
    if torch.cuda.is_available():
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            entry: dict[str, Any] = {
                "index": i,
                "kind": "cuda",
                "device": f"cuda:{i}",
                "name": props.name,
                "mem_total_mb": round(props.total_memory / 2**20),
                "mem_used_mb": None,
                "util_pct": None,
                "temp_c": None,
                "power_w": None,
                "power_limit_w": None,
                "uuid": str(props.uuid) if getattr(props, "uuid", None) else None,
                "cuda_available": True,
                "telemetry_source": "torch",
            }
            try:
                free, total = torch.cuda.mem_get_info(i)
                entry.update(mem_used_mb=round((total - free) / 2**20), mem_free_mb=round(free / 2**20))
            except RuntimeError:
                pass
            out.append(entry)
        _nvml_metrics(out)
        fallback = (
            _nvidia_smi()
            if any(
                any(
                    g.get(key) is None
                    for key in (
                        "power_w",
                        "util_pct",
                        "temp_c",
                        "power_limit_w",
                        "mem_used_mb",
                        "mem_free_mb",
                    )
                )
                for g in out
            )
            else []
        )
        for entry in out:
            # CUDA devices can be reordered; match UUID, then an unambiguous name.
            matches = [g for g in fallback if g["uuid"] == entry.get("uuid")]
            if not matches and not entry.get("uuid"):
                matches = [g for g in fallback if g["name"] == entry["name"]]
            if len(matches) == 1:
                for key in ("util_pct", "temp_c", "power_w", "power_limit_w", "mem_used_mb", "mem_free_mb"):
                    if entry.get(key) is None and matches[0].get(key) is not None:
                        entry[key] = matches[0][key]
                        entry["telemetry_source"] = (
                            "nvml+nvidia-smi"
                            if entry["telemetry_source"].startswith("nvml")
                            else "nvidia-smi"
                        )
            if entry["power_w"] is None:
                entry["telemetry_note"] = "nvidia_power_unavailable"
        return out
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        vm = system_memory if system_memory is not None else psutil.virtual_memory()
        utilization = _apple_gpu_utilization()
        sensors = _apple_gpu_sensors()
        # This is system unified-memory usage, not a fabricated GPU-process allocation.
        return [
            {
                "index": 0,
                "kind": "mps",
                "device": "mps",
                "name": _apple_name(),
                "mem_total_mb": round(vm.total / 2**20),
                "mem_used_mb": round((vm.total - vm.available) / 2**20),
                "mem_free_mb": round(vm.available / 2**20),
                "memory_scope": "unified_system",
                "util_pct": utilization,
                **sensors,
                "power_source": "ioreport" if sensors["power_w"] is not None else None,
                "power_estimated": True if sensors["power_w"] is not None else None,
                "temperature_source": "smc" if sensors["temp_c"] is not None else None,
                "telemetry_source": "ioreg" if utilization is not None else "mps",
                "telemetry_note": "mps_power_unavailable" if sensors["power_w"] is None else None,
            }
        ]
    if include_unavailable:
        return [
            {
                **g,
                "kind": "cuda",
                "device": None,
                "cuda_available": False,
                "telemetry_source": "nvidia-smi",
                "telemetry_note": "cuda_runtime_unavailable",
            }
            for g in _nvidia_smi()
        ]
    return []
