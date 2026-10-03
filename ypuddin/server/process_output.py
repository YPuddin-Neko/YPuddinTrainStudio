"""Collect a worker's merged stdout/stderr without losing native diagnostics."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from .job_logs import captured_output

log = logging.getLogger(__name__)
CHUNK_SIZE = 64 * 1024


def _report_error(message: str, path: Path) -> None:
    try:
        log.exception(message, path)
    except Exception:
        # A service log handler may share the same full volume. Keep draining
        # the worker's output even when reporting the write failure also fails.
        pass


class ProcessOutput:
    def __init__(self, stream: BinaryIO, path: Path):
        self.stream = stream
        self.path = path
        self._switches: dict[bytes, Path] = {}
        self._lock = threading.Lock()
        self._sink: BinaryIO | None = None
        self._write_failed = False
        self._open_sink()
        self._thread = threading.Thread(target=self._run, name="job-output", daemon=True)
        self._thread.start()

    def switch_marker(self, path: Path) -> str:
        """The resident worker writes this boundary before starting its next job."""
        marker = f"\x1eYPUDDIN_LOG_SWITCH:{uuid.uuid4().hex}\x1e"
        with self._lock:
            self._switches[marker.encode("ascii")] = path
        return marker

    @property
    def finished(self) -> bool:
        """Whether the pipe reached its end and every line is in the log."""
        return not self._thread.is_alive()

    def join(self, timeout: float = 2.0) -> None:
        # A surviving descendant may still hold the pipe. Do not let collecting
        # its logs block the supervisor's process-exit or cancellation handling.
        self._thread.join(timeout)

    def _write(self, raw: bytes, observed: datetime, *, continued: bool = False) -> None:
        if self._write_failed:
            return
        try:
            payload = memoryview(raw if continued else captured_output(raw, observed))
            while payload:
                written = self._sink.write(payload)
                if not written:
                    raise OSError("worker output log accepted no bytes")
                payload = payload[written:]
        except OSError:
            # Continue draining so a full/unwritable log volume cannot block the
            # worker on a full stdout pipe or change its exit code.
            self._write_failed = True
            _report_error("could not write captured worker output to %s", self.path)

    def _switch(self, raw: bytes) -> bool:
        with self._lock:
            path = self._switches.pop(raw.rstrip(b"\r\n"), None)
        if path is None:
            return False
        self._close_sink()
        self.path, self._sink, self._write_failed = path, None, False
        self._open_sink()
        return True

    def _open_sink(self) -> None:
        try:
            self._sink = self.path.open("ab", buffering=0)
        except OSError:
            self._write_failed = True
            _report_error("could not open captured worker output for %s", self.path)

    def _close_sink(self) -> None:
        if self._sink is not None:
            try:
                self._sink.close()
            except OSError:
                _report_error("could not close captured worker output for %s", self.path)

    def _run(self) -> None:
        pending = bytearray()
        continued = False
        observed = datetime.now().astimezone()
        try:
            read = getattr(self.stream, "read1", self.stream.read)
            while chunk := read(CHUNK_SIZE):
                received = datetime.now().astimezone()
                if not pending and not continued:
                    observed = received
                pending.extend(chunk)
                while (end := pending.find(b"\n")) >= 0:
                    raw = bytes(pending[:end + 1])
                    del pending[:end + 1]
                    if continued or not self._switch(raw):
                        self._write(raw, observed, continued=continued)
                    continued = False
                    observed = received
                if len(pending) >= CHUNK_SIZE:
                    # Long native output stays bounded and byte-for-byte intact;
                    # subsequent chunks continue the same stamped physical line.
                    self._write(bytes(pending), observed, continued=continued)
                    pending.clear()
                    continued = True
            if pending:
                self._write(bytes(pending), observed, continued=continued)
        except OSError:
            _report_error("could not read worker output for %s", self.path)
        finally:
            self._close_sink()
            self.stream.close()
