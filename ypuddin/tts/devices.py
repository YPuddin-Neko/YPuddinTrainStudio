"""Inspect service-visible CUDA indices in the selected TTS Python environment."""

from __future__ import annotations

import os
import re
from typing import Any

from .issues import TtsIssue


def _issue(device: str, code: str, message: str) -> dict:
    return TtsIssue(code=code, loc=["gpu_devices", device], message=message).model_dump()


def inspect_devices(torch: Any, requested: list[str], require_bf16: bool = True) -> dict:
    requested = list(requested)
    result = {
        "requested_devices": requested,
        "eligible_devices": [],
        "checked_devices": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
    }
    unavailable = None
    count = 0
    try:
        if getattr(torch.version, "hip", None) or not torch.version.cuda:
            raise ValueError("TTS 环境需要 NVIDIA CUDA 版 PyTorch。")
        if not torch.cuda.is_available():
            raise ValueError("TTS 环境无法使用 NVIDIA CUDA，请检查驱动与所选 Python 环境。")
        count = torch.cuda.device_count()
    except Exception as exc:
        unavailable = str(exc)
    for device in requested or [f"cuda:{index}" for index in range(count)]:
        item = {"device": device, "name": None, "uuid": None, "bf16": None,
                "state": "unavailable", "issues": []}
        result["checked_devices"].append(item)
        try:
            if unavailable:
                raise ValueError(unavailable)
            if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
                raise ValueError("TTS 任务须选择 NVIDIA CUDA 显卡。")
            index = int(device.split(":", 1)[1])
            if index >= count:
                raise ValueError(f"显卡 {device} 不在当前 CUDA 可见设备中。")
            with torch.cuda.device(index):
                item["name"] = str(torch.cuda.get_device_name(index))
                identity = getattr(torch.cuda.get_device_properties(index), "uuid", None)
                item["uuid"] = str(identity) if identity is not None else None
                if require_bf16:
                    try:
                        item["bf16"] = bool(torch.cuda.is_bf16_supported())
                    except Exception as exc:
                        item["issues"].append(_issue(device, "tts.device.bf16", f"无法检查 {device} 的 BF16 支持：{exc}"))
                    if item["bf16"] is False:
                        item["issues"].append(_issue(device, "tts.device.bf16", f"显卡 {device} 不支持 VoxCPM 1.5 训练所需的 BF16。"))
                if not item["issues"]:
                    item["state"] = "available"
                    result["eligible_devices"].append(device)
        except Exception as exc:
            item["issues"].append(_issue(device, "tts.device.unavailable", str(exc)))
    return result


def selection_issues(devices: dict) -> list[dict]:
    if devices["eligible_devices"]:
        return []
    return [issue for item in devices["checked_devices"] for issue in item["issues"]] or [
        TtsIssue(code="tts.device.unavailable", loc=["gpu_devices"],
                 message="当前 TTS 环境没有可用的 NVIDIA CUDA 显卡。").model_dump()
    ]
