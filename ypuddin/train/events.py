"""Structured training events: JSON Lines to a file and/or an inherited pipe fd, plus callbacks."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Listener = Callable[[dict[str, Any]], None]


class Emitter:
    def __init__(self, *, path: str | Path | None = None, fd: int | None = None, listeners: list[Listener] | None = None):
        self._file = open(path, "a", encoding="utf-8", buffering=1) if path else None  # noqa: SIM115
        self._fd = fd
        self._listeners: list[Listener] = list(listeners or [])
        self._seq = 0
        self._lock = threading.Lock()

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def emit(self, type_: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "ts": time.time(), "type": type_, **data}
            line = json.dumps(event, ensure_ascii=False, default=_json_default)
            if self._file is not None:
                self._file.write(line + "\n")
            if self._fd is not None:
                try:
                    os.write(self._fd, (line + "\n").encode("utf-8"))
                except OSError:
                    self._fd = None
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - listeners must never break training
                pass
        return event

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class NullEmitter(Emitter):
    def __init__(self) -> None:
        super().__init__()
