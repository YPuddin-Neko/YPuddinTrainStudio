"""Align task GPU driver readings with the recorded training steps."""

from __future__ import annotations

import math
import re
from typing import Any

DEVICE_ID = re.compile(r"^(mps|cuda:(0|[1-9][0-9]*))$")
READINGS = ("power_w", "temp_c", "util_pct", "mem_used_mb", "mem_total_mb")


def device_metric_series(events: list[dict[str, Any]], since_step: int = 0) -> list[dict[str, Any]]:
    series: dict[str, dict[str, Any]] = {}
    count = 0
    for event in events:
        if event.get("type") != "step":
            continue
        included = event["step"] > since_step
        if included:
            count += 1
            for device in series.values():
                for key in READINGS:
                    device[key].append(None)
        readings = event.get("gpu_devices")
        if not isinstance(readings, list):
            continue
        seen = set()
        for reading in readings:
            if not isinstance(reading, dict):
                continue
            identity = reading.get("id")
            if not isinstance(identity, str) or len(identity) > 24 or not DEVICE_ID.fullmatch(identity) or identity in seen:
                continue
            seen.add(identity)
            if identity not in series:
                series[identity] = {
                    "id": identity,
                    "index": 0 if identity == "mps" else int(identity.partition(":")[2]),
                    **{key: [None] * count for key in READINGS},
                }
            device = series[identity]
            # Keep the first known identity even when later readings omit or
            # change metadata. Historical identity also survives since_step.
            if "name" not in device and isinstance(reading.get("name"), str) and reading["name"]:
                device["name"] = reading["name"]
            if "kind" not in device and isinstance(reading.get("kind"), str) and reading["kind"] in {"cuda", "mps", "dtk", "rocm"}:
                device["kind"] = reading["kind"]
            if included:
                for key in READINGS:
                    value = reading.get(key)
                    if type(value) not in (int, float):
                        continue
                    try:
                        value = float(value)
                    except OverflowError:
                        continue
                    if math.isfinite(value) and value >= 0:
                        device[key][-1] = value
    for device in series.values():
        device.setdefault("name", device["id"])
        device.setdefault("kind", "mps" if device["id"] == "mps" else "cuda")
    return list(series.values())
