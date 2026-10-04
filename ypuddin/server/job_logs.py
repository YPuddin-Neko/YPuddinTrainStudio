"""Read worker logs by byte offset and parse each line's record header.

A ``record`` line carries a Python logging or PyTorch glog header. A ``traceback``
line starts an interpreter traceback printed without a header. Other ``text`` lines
belong to the record above them (traceback frames, multi-line messages, progress
bars); the log view groups them because a record can span two reads.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

_RANK = r"(?P<prefix>\[rank\d+\]:\s*)?"
_RECORD = re.compile(
    _RANK
    + r"(?P<time>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"\s+(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\s+"
    r"(?:(?P<source>[\w.\-<>]+):\s)?(?P<message>.*)$"
)
# PyTorch C++ and elastic-agent records: ``E0923 19:27:48.123000 4821 torch/x.py:874] msg``.
_GLOG = re.compile(
    _RANK
    + r"\[?(?P<level>[IWEF])(?P<month>\d{2})(?P<day>\d{2}) (?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+\d+\s+(?P<source>[^\]\s]+)\]\s?(?P<message>.*)$"
)
_BASIC_RECORD = re.compile(
    _RANK + r"(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL):"
    r"(?P<source>[\w.\-<>]+):(?P<message>.*)$"
)
_BARE_RECORD = re.compile(_RANK + r"(?P<level>WARN(?:ING)?|ERROR|CRITICAL|FATAL):(?P<message>.*)$")
_TRACEBACK = re.compile(_RANK + r"\s*Traceback \(most recent call last\):")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROCESS_RANK = re.compile(r"^\[rank(\d+)\]:\s?")
_WARNING_HEAD = re.compile(r"^(?:\[rank\d+\]:\s*)?(?:[^\n]+:\d+:\s*)?\w*Warning:")
_CHAINED = re.compile(
    r"^(?:During handling of the above exception, another exception occurred:"
    r"|The above exception was the direct cause of the following exception:)$"
)
_ORDER = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_LEVELS = {"warning": "warn", "critical": "error", "fatal": "error"}
_GLOG_LEVELS = {"I": "info", "W": "warn", "E": "error", "F": "error"}
_CAPTURED = re.compile(r"^\[captured (?P<time>[^\]]+)\] (?P<message>.*)$")
_NCCL_CONFIGURATION = re.compile(
    r"^(?:\[rank\d+\]:\s*)?\[PG ID \d+ PG GUID \S+ Rank \d+\] "
    r"ProcessGroupNCCL (?:initialization options|environments):(?:\s|$)"
)

MAX_READ = 512 * 1024
SUPERVISOR_SOURCE = "ypuddin.server.supervisor"
OUTPUT_SOURCE = "process.output"


def captured_output(raw: bytes, observed: datetime) -> bytes:
    """Keep existing record times; otherwise persist when the output was collected."""
    parsed = parse_log_lines([raw.decode("utf-8", errors="replace").rstrip("\n")], now=observed)[0]
    if parsed["ts"] is not None:
        return raw
    stamp = (observed if observed.tzinfo is not None else observed.astimezone()).isoformat(timespec="milliseconds")
    return f"[captured {stamp}] ".encode("ascii") + raw


def failure_record_bytes(record: Mapping[str, Any]) -> bytes:
    """Serialize a supervisor record using the worker log's existing header format."""
    timestamp = record.get("ts")
    if timestamp is None:
        header = f"ERROR:{SUPERVISOR_SOURCE}:"
    else:
        stamp = datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")
        header = f"{stamp} ERROR {SUPERVISOR_SOURCE}: "
    return (header + str(record["msg"]) + "\n").encode("utf-8")


def missing_failure_record(path: Path, job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Use the saved supervisor outcome when its record could not reach run.log."""
    if job.get("status") != "failed" or not job.get("error"):
        return None
    record = {
        "offset": 0,
        "kind": "record",
        "ts": job.get("finished_at"),
        "level": "error",
        "source": SUPERVISOR_SOURCE,
        "msg": job["error"],
    }
    payload = failure_record_bytes(record)
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            record["offset"] = size
            start = max(0, size - len(payload) - 4096)
            stream.seek(start)
            tail = stream.read(len(payload) + 4096)
        # Only our complete record with this outcome's timestamp counts as persisted.
        if (start == 0 and tail.startswith(payload)) or b"\n" + payload in tail:
            return None
    except FileNotFoundError:
        pass
    return record


def append_failure_record(path: Path, job: Mapping[str, Any]) -> None:
    record = missing_failure_record(path, job)
    if record is None:
        return
    separator = b""
    if record["offset"]:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                separator = b"\n"
    with path.open("ab") as stream:
        stream.write(separator + failure_record_bytes(record))


def _clean(raw: str) -> str:
    # A carriage return redraws the terminal line (progress bars); keep the final state.
    line = _ANSI.sub("", raw).rstrip("\r")
    if "\r" in line:
        line = line.rsplit("\r", 1)[-1]
    return line


def _glog_time(match: re.Match[str], now: datetime) -> float | None:
    month, day = int(match["month"]), int(match["day"])
    year = now.year - 1 if (month, day) > (now.month, now.day) else now.year
    try:
        return datetime.fromisoformat(f"{year:04d}-{month:02d}-{day:02d} {match['time']}").timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def parse_log_lines(lines: list[str], *, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now()
    out = []
    for raw in lines:
        # Unwrap before terminal redraw handling, and retain the original line kind:
        # traceback frames and multi-line messages must still join their header.
        if captured := _CAPTURED.match(raw):
            try:
                observed = datetime.fromisoformat(captured["time"])
                timestamp = observed.timestamp()
            except (ValueError, OverflowError, OSError):
                pass
            else:
                # Nested or malformed capture-like text can come from a worker.
                # Only this outer envelope is metadata; never recursively unwrap.
                [line] = _parse_plain_lines([captured["message"]], now=observed)
                if line["ts"] is None:
                    line["ts"] = timestamp
                if line["source"] is None:
                    line["source"] = OUTPUT_SOURCE
                out.append(line)
                continue
        out.extend(_parse_plain_lines([raw], now=now))
    for line in out:
        # Native configuration dumps bypass Python logging; keep their raw file and source intact.
        if line["level"] == "info" and _NCCL_CONFIGURATION.match(line["msg"]):
            line["kind"] = "record"
            line["level"] = "debug"
    return out


def _parse_plain_lines(lines: list[str], *, now: datetime) -> list[dict[str, Any]]:
    out = []
    for raw in lines:
        line = _clean(raw)
        if match := _RECORD.match(line):
            level = match["level"].lower()
            try:
                ts = datetime.fromisoformat(match["time"].replace(",", ".").replace("Z", "+00:00")).timestamp()
            except (ValueError, OverflowError, OSError):
                ts = None
            out.append(
                {
                    "kind": "record",
                    "ts": ts,
                    "level": _LEVELS.get(level, level),
                    "source": match["source"],
                    "msg": (match["prefix"] or "") + match["message"],
                }
            )
            continue
        if match := _GLOG.match(line):
            out.append(
                {
                    "kind": "record",
                    "ts": _glog_time(match, now),
                    "level": _GLOG_LEVELS[match["level"]],
                    "source": match["source"],
                    "msg": (match["prefix"] or "") + match["message"],
                }
            )
            continue
        if match := _BASIC_RECORD.match(line):
            level = match["level"].lower()
            out.append(
                {
                    "kind": "record",
                    "ts": None,
                    "level": _LEVELS.get(level, level),
                    "source": match["source"],
                    "msg": (match["prefix"] or "") + match["message"],
                }
            )
            continue
        if match := _BARE_RECORD.match(line):
            level = match["level"].lower()
            out.append(
                {"kind": "record", "ts": None, "level": _LEVELS.get(level, level), "source": None, "msg": line}
            )
            continue
        if _TRACEBACK.match(line):
            out.append({"kind": "traceback", "ts": None, "level": "error", "source": None, "msg": line})
            continue
        level = "error" if re.search(r"\b\w*(?:Error|Exception):", line) else "info"
        if level == "info" and re.search(r"(?:^\s*warning\b|\b\w*Warning:)", line, re.IGNORECASE):
            level = "warn"
        out.append({"kind": "text", "ts": None, "level": level, "source": None, "msg": line})
    return out


def entry_levels(lines: list[Mapping[str, Any]]) -> list[str]:
    """The level of the log entry each parsed line belongs to, grouped as the job log view groups them.

    A record keeps its own level for every line joined to it, so the traceback a warning was logged
    with (``exc_info``) reads as part of that warning. Any other traceback starts an error entry:
    one after an info or debug record, after plain output, or after a warning whose own traceback
    has ended. Keep in step with ``groupLogLines`` in ``frontend/src/utils/jobLogs.ts``.
    """
    levels: list[str] = []
    level = kind = rank = None
    traceback = "none"
    chained = False
    for line in lines:
        line_kind = line.get("kind") or "text"
        line_level = line.get("level") if line.get("level") in _ORDER else "info"
        message = str(line.get("msg") or "")
        found = _PROCESS_RANK.match(message)
        line_rank = found[1] if found else None
        body = message[found.end() :] if found else message
        different_rank = line_rank is not None and rank is not None and line_rank != rank
        if level is None or different_rank:
            joins = False
        elif line_kind == "traceback":
            joins = (
                kind == "traceback"
                or level == "error"
                or (kind == "record" and level == "warn" and (traceback == "none" or chained))
            )
        elif line_kind == "text":
            warning_head = traceback != "frames" and _WARNING_HEAD.match(message) is not None
            joins = not warning_head and (
                traceback == "frames" or line_level == "info" or _ORDER[line_level] <= _ORDER[level]
            )
        else:
            joins = False
        if not joins:
            level, kind, rank, traceback = line_level, line_kind, line_rank, "none"
        if line_kind == "traceback":
            traceback = "frames"
        elif traceback == "frames" and body.strip() and not body[:1].isspace():
            traceback = "done"  # the unindented exception line ends the frames
        if body.strip():
            chained = _CHAINED.match(body.strip()) is not None
        levels.append(level)
    return levels


def _split(chunk: bytes, base: int) -> list[tuple[int, bytes]]:
    """Split on newline only; ``str.splitlines`` would also split progress-bar redraws."""
    out, start = [], 0
    while start < len(chunk):
        end = chunk.find(b"\n", start)
        if end < 0:
            out.append((base + start, chunk[start:]))
            break
        out.append((base + start, chunk[start : end + 1]))
        start = end + 1
    return out


def _partial(pieces: list[tuple[int, bytes]]) -> bool:
    """Whether the last piece is an unfinished line that a later read can complete."""
    if not pieces or pieces[-1][1].endswith(b"\n"):
        return False
    # A single line longer than one read would otherwise never be returned.
    return not (len(pieces) == 1 and len(pieces[0][1]) >= MAX_READ)


def read_log(
    path: Path,
    *,
    offset: int = 0,
    limit: int = 1000,
    tail: bool = False,
    before: int | None = None,
    complete_only: bool = False,
) -> dict[str, Any]:
    """Return up to ``limit`` lines with the byte range they cover.

    ``tail`` reads the newest lines, ``before`` the lines ending at that offset (for
    loading earlier output), otherwise lines from ``offset`` onward. While the worker
    still writes, ``complete_only`` holds back a line without its newline so the
    next read returns it whole.
    """
    limit = max(1, min(2000, limit))
    if not path.exists():
        return {"lines": [], "start_offset": 0, "next_offset": 0, "has_more": False, "has_earlier": False}
    size = path.stat().st_size
    with path.open("rb") as stream:
        if tail or before is not None:
            end = size if tail else min(size, max(0, before))
            start = max(0, end - MAX_READ)
            stream.seek(start)
            pieces = _split(stream.read(end - start), start)
            if start > 0 and pieces:
                pieces = pieces[1:]  # the first piece may begin inside a line
        else:
            end = start = min(size, max(0, offset))
            stream.seek(start)
            pieces = _split(stream.read(MAX_READ), start)
    resume = end
    if complete_only and _partial(pieces):
        resume = pieces[-1][0]
        pieces = pieces[:-1]
    pieces = pieces[-limit:] if tail or before is not None else pieces[:limit]
    if pieces:
        first, last = pieces[0][0], pieces[-1][0] + len(pieces[-1][1])
    else:
        first = last = resume
    text = [piece.decode("utf-8", errors="replace").rstrip("\n") for _, piece in pieces]
    lines = parse_log_lines(text)
    for line, (position, _) in zip(lines, pieces, strict=True):
        line["offset"] = position
    return {
        "lines": lines,
        "start_offset": first,
        "next_offset": last,
        "has_more": last < size,
        "has_earlier": first > 0,
    }
