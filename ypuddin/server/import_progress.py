"""Bounded, process-local import progress independent of the database transaction lock."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from .errors import ApiError

ImportPhase = Literal[
    "receiving", "extracting", "validating", "copying", "registering", "completed", "failed"
]
BYTE_PHASES = {"receiving", "extracting", "copying"}
TERMINAL_PHASES = {"completed", "failed"}


class ImportProgress:
    def __init__(self, store: ImportProgressStore, pid: str, id: str, phase: ImportPhase):
        self.store, self.pid, self.id = store, pid, id
        self.started = self.phase_started = store.clock()
        self.finished: float | None = None
        self.phase = phase
        self.bytes_done = self.files_done = 0
        self.bytes_total: int | None = None
        self.files_total: int | None = None
        self.error: str | None = None

    def set_phase(
        self, phase: ImportPhase, *, bytes_total: int | None = None, files_total: int | None = None
    ) -> None:
        with self.store.lock:
            if self.finished is not None:
                return
            self.phase, self.phase_started = phase, self.store.clock()
            self.bytes_done = self.files_done = 0
            self.bytes_total, self.files_total = bytes_total, files_total
            if phase in TERMINAL_PHASES:
                self.finished = self.phase_started

    def advance(self, *, bytes_done: int = 0, files_done: int = 0) -> None:
        if bytes_done < 0 or files_done < 0:
            raise ValueError("import progress increments must not be negative")
        with self.store.lock:
            if self.finished is None:
                self.bytes_done += bytes_done
                self.files_done += files_done

    def complete(self) -> None:
        self.set_phase("completed")

    def fail(self, error: str) -> None:
        with self.store.lock:
            if self.finished is None:
                self.error = error if re.fullmatch(r"[a-z][a-z0-9_.]{0,79}", error) else "import.failed"
                self.set_phase("failed")

    def snapshot(self) -> dict[str, Any]:
        with self.store.lock:
            current = self.store.clock() if self.finished is None else self.finished
            elapsed = max(0.0, current - self.phase_started)
            speed = (
                self.bytes_done / elapsed
                if self.phase in BYTE_PHASES and self.bytes_done and elapsed > 0
                else None
            )
            return {
                "id": self.id,
                "phase": self.phase,
                "bytes_done": self.bytes_done,
                "bytes_total": self.bytes_total,
                "files_done": self.files_done,
                "files_total": self.files_total,
                "elapsed_seconds": max(0.0, current - self.started),
                "phase_elapsed_seconds": elapsed,
                "bytes_per_second": speed,
                "eta_seconds": max(0, self.bytes_total - self.bytes_done) / speed
                if speed and self.bytes_total is not None
                else None,
                "error": self.error,
            }


class ImportProgressStore:
    def __init__(
        self, *, capacity: int = 256, ttl_seconds: float = 900, clock: Callable[[], float] = time.monotonic
    ):
        self.capacity, self.ttl_seconds, self.clock = capacity, ttl_seconds, clock
        self.lock = threading.RLock()
        self.entries: dict[str, ImportProgress] = {}

    @staticmethod
    def validate_id(id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", id):
            raise ApiError("invalid import progress identifier", code="progress.invalid")

    def _expire(self) -> None:
        current = self.clock()
        for id, entry in list(self.entries.items()):
            if entry.finished is not None and current - entry.finished >= self.ttl_seconds:
                del self.entries[id]

    def start(self, pid: str, id: str, phase: ImportPhase = "receiving") -> ImportProgress:
        self.validate_id(id)
        with self.lock:
            self._expire()
            if id in self.entries:
                raise ApiError(
                    "import progress identifier is already in use", code="progress.conflict", status=409
                )
            if len(self.entries) >= self.capacity:
                raise ApiError(
                    "import progress capacity is full; retry later", code="progress.capacity", status=503
                )
            result = self.entries[id] = ImportProgress(self, pid, id, phase)
            return result

    def get(self, pid: str, id: str) -> dict[str, Any]:
        self.validate_id(id)
        with self.lock:
            self._expire()
            entry = self.entries.get(id)
            if entry is None or entry.pid != pid:
                raise ApiError("import progress not found", code="progress.not_found", status=404)
            return entry.snapshot()

    @contextmanager
    def track(self, pid: str, id: str | None, phase: ImportPhase) -> Iterator[ImportProgress | None]:
        progress = self.start(pid, id, phase) if id is not None else None
        try:
            yield progress
        except BaseException as error:
            if progress:
                progress.fail(error.code if isinstance(error, ApiError) else "import.failed")
            raise
        else:
            if progress:
                progress.complete()
