"""Parse Python logging records while preserving unstructured worker output."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_RECORD = re.compile(
    r"^(?P<prefix>\[rank\d+\]:\s*)?"
    r"(?P<time>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"\s+(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\s+(?P<message>.*)$"
)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_log_lines(lines: list[str]) -> list[dict[str, Any]]:
    out = []
    traceback = False
    for raw in lines:
        line = _ANSI.sub("", raw)
        match = _RECORD.match(line)
        if match:
            level = match["level"].lower()
            level = {"warning": "warn", "critical": "error", "fatal": "error"}.get(level, level)
            try:
                timestamp = datetime.fromisoformat(
                    match["time"].replace(",", ".").replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                timestamp = None
            out.append({"ts": timestamp, "level": level, "msg": (match["prefix"] or "") + match["message"]})
            traceback = False
            continue
        if line.lstrip().startswith("Traceback (most recent call last)"):
            traceback = True
        level = "error" if traceback or re.search(r"\b\w*(?:Error|Exception):", line) else "info"
        if level == "info" and re.search(r"(?:^\s*warning\b|\b\w*Warning:)", line, re.IGNORECASE):
            level = "warn"
        out.append({"ts": None, "level": level, "msg": line})
    return out
