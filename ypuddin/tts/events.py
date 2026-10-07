"""Append-only TTS events without importing image training dependencies."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class Events:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.sync()

    def sync(self) -> None:
        if self.path.is_file():
            with self.path.open("rb") as stream:
                stream.seek(max(0, self.path.stat().st_size - 65_536))
                for line in stream:
                    try:
                        self.seq = max(self.seq, int(json.loads(line).get("seq", 0)))
                    except (ValueError, TypeError):
                        continue

    def emit(self, type_: str, **data: Any) -> None:
        self.seq += 1
        record = {"type": type_, "seq": self.seq, "ts": time.time(), **data}
        line = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("无法写入 TTS 事件文件。")
                view = view[written:]
        finally:
            os.close(descriptor)
