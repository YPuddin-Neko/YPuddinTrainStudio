"""Bounded queue-only capability caching for the selected speech runtime and visible devices."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path

from ypuddin.tts.execution_config import parse_execution_config
from ypuddin.tts.validation_models import TtsDeviceSelection

_CACHE_SECONDS = 30.0
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def invalidate() -> None:
    with _lock:
        _cache.clear()


def _key(recipe: dict, requested: list[str], mode: str, inventory: list[dict]) -> str:
    paths = {}
    for field in ("python_path", "trainer_path", "model_path"):
        path = Path(recipe.get(field) or "").expanduser()
        try:
            info = path.stat()
            paths[field] = [str(path.resolve()), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
        except OSError:
            paths[field] = [str(path), None]
    return json.dumps({
        "requested": requested, "mode": mode, "paths": paths,
        "visibility": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES")},
        "inventory": [{key: gpu.get(key) for key in ("device", "uuid", "name", "compute_capability", "mem_total_mb")}
                      for gpu in inventory],
    }, sort_keys=True)


def eligible_inventory(job: dict, inventory: list[dict]) -> list[dict]:
    from ypuddin.tts.core import runtime_probe

    recipe = json.loads(job["config_json"])["tts"]
    requested = json.loads(job.get("gpu_devices_json") or "[]")
    mode = "train" if job["type"] == "tts_train" else "sample"
    key = _key(recipe, requested, mode, inventory)
    with _lock:
        cached = _cache.get(key)
        selection = copy.deepcopy(cached[1]) if cached and time.monotonic() - cached[0] < _CACHE_SECONDS else None
    if selection is None:
        report = runtime_probe(parse_execution_config(recipe), mode=mode, gpu_devices=requested)
        if not report["ok"]:
            raise ValueError("\n".join(str(error) for error in report["errors"]))
        selection = TtsDeviceSelection.model_validate(report.get("details", {}).get("devices")).model_dump()
        if selection["requested_devices"] != requested or key != _key(recipe, requested, mode, inventory):
            raise ValueError("显卡选择或运行环境在检查期间发生变化，请重新检查。")
        service_devices = {gpu["device"]: gpu for gpu in inventory}
        for device in selection["checked_devices"]:
            service_uuid = service_devices.get(device["device"], {}).get("uuid")
            runtime_uuid = device["uuid"]
            if service_uuid and runtime_uuid and str(service_uuid).lower().removeprefix("gpu-") != str(runtime_uuid).lower().removeprefix("gpu-"):
                raise ValueError("TTS Python 与服务识别的显卡身份不一致，请检查 CUDA_VISIBLE_DEVICES 和 CUDA_DEVICE_ORDER。")
        with _lock:
            if len(_cache) >= 128:
                _cache.clear()
            _cache[key] = time.monotonic(), copy.deepcopy(selection)
    eligible = set(selection["eligible_devices"])
    if requested and not set(requested) <= eligible:
        raise ValueError("所选显卡不满足此语音任务的运行要求。")
    candidates = [gpu for gpu in inventory if gpu["device"] in eligible]
    if not candidates:
        raise ValueError("当前没有满足此语音任务运行要求的可见显卡。")
    return candidates
