"""Accelerator inventory shared by system stats and single-device job scheduling."""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.parsers.expat import ExpatError

import psutil
import torch

from ypuddin.runtime_profiles import current_profile

_smi_lock = threading.Lock()
_smi_cache: tuple[float, list[dict[str, Any]]] = (0, [])
_apple_gpu_lock = threading.Lock()
_apple_gpu_cache: tuple[float, float | None] | None = None
_apple_sensors_lock = threading.Lock()
_apple_sensors_cache: tuple[float, dict[str, Any]] | None = None
_hip_sensors_lock = threading.Lock()
_hip_sensors_cache: tuple[float, tuple, list[dict[str, Any]]] | None = None
_DRM_SYSFS = Path("/sys/class/drm")
_DRM_DEVICES = Path("/dev/dri")


def _sysfs_text(path: Path) -> str:
    try:
        with path.open(encoding="ascii") as stream:
            return stream.read(128).strip()
    except (OSError, UnicodeError):
        return ""


def _hip_unique_id(value: Any) -> str | None:
    """DTK encodes its 64-bit sysfs unique_id as 16 ASCII bytes in Torch's UUID."""
    raw = str(value or "").lower().removeprefix("gpu-").removeprefix("0x")
    if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", raw):
        try:
            raw = bytes.fromhex(raw.replace("-", "")).decode("ascii").lower()
        except UnicodeError:
            return None
    return raw if re.fullmatch(r"[0-9a-f]{16}", raw) and int(raw, 16) else None


def _pci_address(value: Any) -> str | None:
    raw = str(value or "").lower()
    return raw if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", raw) else None


def _drm_accessible(path: Path) -> bool:
    try:
        if not stat.S_ISCHR(path.stat().st_mode):
            return False
        # Existence/mode bits alone do not enforce shared-node device cgroups.
        # Opening a read-only fd checks actual access without a driver ioctl.
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        os.close(fd)
        return True
    except OSError:
        return False


def _hip_sensor_reading(device: Path) -> dict[str, Any]:
    def number(path: Path, scale: float = 1, maximum: float = math.inf):
        value = _number(_sysfs_text(path))
        return value / scale if value is not None and value / scale <= maximum else None

    reading: dict[str, Any] = {"util_pct": number(device / "gpu_busy_percent", maximum=100)}
    total = number(device / "mem_info_vram_total", 2**20)
    used = number(device / "mem_info_vram_used", 2**20)
    if total is not None and total > 0 and used is not None and used <= total:
        reading.update(mem_total_mb=round(total), mem_used_mb=round(used), mem_free_mb=round(total - used))
    for sensor in sorted((device / "hwmon").glob("hwmon*"))[:16]:
        if _sysfs_text(sensor / "name") not in ("hycu", "amdgpu"):
            continue
        # Both drivers expose package power in microwatts and edge temperature
        # in millidegrees C. Other power channels are components, not extra GPUs.
        reading["power_w"] = number(sensor / "power1_average", 1e6)
        reading["power_limit_w"] = number(sensor / "power1_cap", 1e6)
        # Older amdgpu hwmon implementations omit the optional edge label.
        if _sysfs_text(sensor / "temp1_label") in ("", "edge"):
            temperature = number(sensor / "temp1_input", 1e3, 150)
            reading["temp_c"] = temperature if temperature is not None and temperature > 0 else None
        if reading.get("power_w") is not None:
            reading.update(power_source="hwmon", power_estimated=False)
        if reading.get("temp_c") is not None:
            reading["temperature_source"] = "hwmon-edge"
        break
    return {key: value for key, value in reading.items() if value is not None}


def _hip_sysfs_metrics(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read only allocated DRM nodes, then join to Torch-visible identities.

    DRM card numbers, HIP ordinals and SMI indices differ on shared nodes. Never
    join by index/name or enumerate the host's other GPU sensor readings. Cache
    successes and failures for two seconds; masks/order are part of the key.
    """
    if sys.platform != "linux":
        return [{} for _ in entries]
    global _hip_sensors_cache
    key = tuple((entry.get("uuid"), entry.get("pci_bus_id")) for entry in entries)
    with _hip_sensors_lock:
        if (
            _hip_sensors_cache is not None
            and _hip_sensors_cache[1] == key
            and time.monotonic() - _hip_sensors_cache[0] < 2
        ):
            return [dict(item) for item in _hip_sensors_cache[2]]
        devices = {}
        try:
            for node in sorted(_DRM_DEVICES.iterdir())[:256]:
                if not re.fullmatch(r"(?:card\d+|renderD\d+)", node.name) or not _drm_accessible(node):
                    continue
                device = (_DRM_SYSFS / node.name / "device").resolve()
                if not _pci_address(device.name) or _sysfs_text(device / "vendor") not in (
                    "0x1d94",
                    "0x1002",
                ):
                    continue
                devices[device] = _hip_unique_id(_sysfs_text(device / "unique_id"))
        except OSError:
            pass
        rows = []
        for entry in entries:
            identity = _hip_unique_id(entry.get("uuid"))
            matches = [path for path, unique_id in devices.items() if identity and identity == unique_id]
            if not matches and (pci := _pci_address(entry.get("pci_bus_id"))):
                matches = [path for path in devices if path.name == pci]
            rows.append(_hip_sensor_reading(matches[0]) if len(matches) == 1 else {})
        _hip_sensors_cache = (time.monotonic(), key, rows)
        return [dict(item) for item in rows]


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


def apple_gpu_reading() -> dict[str, float]:
    """The Apple GPU's estimated power, mean temperature and driver load, where each is readable."""
    sensors = _apple_gpu_sensors()
    reading = {
        "power_w": sensors["power_w"],
        "temp_c": sensors["temp_c"],
        "util_pct": _apple_gpu_utilization(),
    }
    return {key: float(value) for key, value in reading.items() if value is not None}


class BackgroundReading:
    """The latest result of a slow reading, refreshed on its own thread.

    Apple's sensors take about a third of a second to read, so a training step takes the last
    reading instead of waiting. The thread ends on close(), or when nobody asked for a minute.
    """

    def __init__(
        self, read: Callable[[], dict[str, Any]], interval: float = 3.0, idle: float = 60.0
    ) -> None:
        self._read, self._interval, self._idle = read, interval, idle
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._value: dict[str, Any] = {}
        self._read_at = 0.0
        self._asked_at = 0.0

    def latest(self) -> dict[str, Any]:
        """The last reading while it is recent; empty until the first one arrives."""
        now = time.monotonic()
        with self._lock:
            self._asked_at = now
            if not self._closed.is_set() and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="gpu-reading", daemon=True)
                self._thread.start()
            recent = now - self._read_at <= max(10.0, 3 * self._interval)
            return copy.deepcopy(self._value) if recent else {}

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                value = self._read()
            except Exception:  # noqa: BLE001 - a failed reading is no reading
                value = {}
            with self._lock:
                self._value, self._read_at = copy.deepcopy(value), time.monotonic()
                unused = self._read_at - self._asked_at > self._idle
            if unused or self._closed.wait(self._interval):
                return

    def close(self) -> None:
        self._closed.set()


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
                        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
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
                    if len(row) != 10 or not row[0].strip().isdigit():
                        continue
                    total, used, free, util, temp, power, limit = [_number(x) for x in row[3:]]
                    if total is not None and total <= 0:
                        total = None
                    if total is not None:
                        used = used if used is None or used <= total else None
                        free = free if free is None or free <= total else None
                    rows.append(
                        {
                            "index": int(row[0]),
                            "uuid": row[1].strip(),
                            "name": row[2].strip(),
                            "mem_total_mb": int(total) if total is not None else None,
                            "mem_used_mb": int(used) if used is not None else None,
                            "mem_free_mb": int(free) if free is not None else None,
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


def _nvml_memory(pynvml: Any, handle: Any) -> dict[str, int]:
    """Keep driver reservations separate from application usage, as in nvidia-smi."""
    try:
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2)
        fields = ("total", "free", "used", "reserved")
    except Exception:  # noqa: BLE001
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # V1 merges reservations into used. Retain capacity/free, and let SMI
        # provide used rather than displaying driver reservations as workload.
        fields = ("total", "free")
    total = memory.total
    if not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
        return {}
    return {
        f"mem_{key}_mb": int(value / 2**20)
        for key in fields
        if isinstance(value := getattr(memory, key, None), (int, float))
        and math.isfinite(value)
        and 0 <= value <= total
    }


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
                try:
                    memory = _nvml_memory(pynvml, handle)
                    if memory:
                        entry.update(memory, telemetry_source="nvml")
                except Exception:  # noqa: BLE001
                    pass
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


def nvml_device_reading(uuid: str | None, name: str) -> dict[str, float]:
    """Power, temperature and load of one GPU from the driver, without a CUDA context."""
    entry: dict[str, Any] = {"uuid": uuid, "name": name}
    _nvml_metrics([entry])
    return {key: entry[key] for key in ("power_w", "temp_c", "util_pct") if key in entry}


def training_gpu_reading(entries: list[dict[str, Any]], primary_id: str) -> dict[str, Any]:
    """Sample participating devices without allocating tensors or changing the current device."""
    readings = [dict(entry) for entry in entries]
    try:
        if readings and readings[0]["kind"] in {"dtk", "rocm"}:
            for entry, reading in zip(readings, _hip_sysfs_metrics(entries), strict=True):
                entry.update(reading)
        elif readings and readings[0]["kind"] == "mps":
            readings[0].update(apple_gpu_reading(), name=_apple_name())
        else:
            _nvml_metrics([entry for entry in readings if entry.get("uuid") or entry.get("name")])
    except Exception:  # noqa: BLE001 - retain device identities when driver telemetry fails
        pass
    for entry in readings:
        entry.setdefault("name", f"GPU {entry['index']}")
    keys = ("id", "index", "name", "kind", "power_w", "temp_c", "util_pct", "mem_used_mb", "mem_total_mb")
    devices = [{key: entry[key] for key in keys if entry.get(key) is not None} for entry in readings]
    primary = next((entry for entry in devices if entry["id"] == primary_id), {})
    return {
        "gpu_devices": devices,
        **{f"gpu_{key}": primary[key] for key in ("power_w", "temp_c", "util_pct") if key in primary},
    }


@lru_cache(maxsize=1)
def _cuda_inventory() -> list[dict[str, Any]]:
    """Discover CUDA/HIP ordinals in a short-lived process, leaving the server GPU-free.

    Torch device properties initialize a primary CUDA context. Keep that allocation
    out of the long-lived HTTP process; the worker inherits device masks and ordering.
    Live memory readings come from the driver, not this temporary process.
    """
    code = """
import json, torch
out = []
for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
    p = torch.cuda.get_device_properties(i)
    out.append(dict(index=i, name=p.name, mem_total_mb=round(p.total_memory / 2**20),
                    uuid=str(p.uuid) if getattr(p, 'uuid', None) else None,
                    pci_bus_id=str(p.pci_bus_id) if getattr(p, 'pci_bus_id', None) else None,
                    compute_capability=[p.major, p.minor]))
print(json.dumps(out))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows = json.loads(result.stdout.strip().splitlines()[-1])
        return rows if isinstance(rows, list) else []
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []


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
    profile = current_profile()
    if profile.endswith("-cpu"):
        return []
    hip = getattr(torch.version, "hip", None)
    devices = _cuda_inventory() if profile != "macos-mps" and (hip or torch.version.cuda) else []
    if devices:
        out = []
        for device in devices:
            i = device["index"]
            entry: dict[str, Any] = {
                **device,
                "index": i,
                "kind": ("dtk" if profile == "linux-dtk" else "rocm") if hip else "cuda",
                "device": f"cuda:{i}",
                "mem_used_mb": None,
                "mem_free_mb": None,
                "util_pct": None,
                "temp_c": None,
                "power_w": None,
                "power_limit_w": None,
                "cuda_available": True,
                "hip_runtime": hip,
                "telemetry_source": "torch",
            }
            out.append(entry)
        if hip:
            # HIP deliberately reuses torch.cuda. NVIDIA telemetry APIs must never
            # be applied to these local indices, even if the machine also has NVIDIA cards.
            for entry, reading in zip(out, _hip_sysfs_metrics(out), strict=True):
                # Driver-free memory changes with other processes. Torch's visible
                # capacity remains an upper bound, including partitioned devices.
                if reading.get("mem_free_mb") is not None:
                    reading["mem_free_mb"] = min(reading["mem_free_mb"], entry["mem_total_mb"])
                entry.update(reading)
                entry["telemetry_source"] = "torch-hip+sysfs" if reading else "torch-hip"
                entry["telemetry_note"] = (
                    "hip_driver_metrics_unavailable"
                    if any(entry[key] is None for key in ("util_pct", "power_w", "temp_c"))
                    else None
                )
            return out
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
                needs_memory = entry.get("mem_used_mb") is None or entry.get("mem_free_mb") is None
                for key in (
                    "util_pct",
                    "temp_c",
                    "power_w",
                    "power_limit_w",
                    "mem_total_mb",
                    "mem_used_mb",
                    "mem_free_mb",
                ):
                    replace_memory = needs_memory and key.startswith("mem_")
                    if (entry.get(key) is None or replace_memory) and matches[0].get(key) is not None:
                        entry[key] = matches[0][key]
                        entry["telemetry_source"] = (
                            "nvml+nvidia-smi"
                            if entry["telemetry_source"].startswith("nvml")
                            else "nvidia-smi"
                        )
            if entry["power_w"] is None:
                entry["telemetry_note"] = "nvidia_power_unavailable"
        return out
    if (
        profile in ("legacy", "macos-mps")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        vm = system_memory if system_memory is not None else psutil.virtual_memory()
        utilization = _apple_gpu_utilization()
        sensors = _apple_gpu_sensors()
        # Apple reports system unified-memory usage, not per-process GPU allocation.
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
    if include_unavailable and not hip:
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
