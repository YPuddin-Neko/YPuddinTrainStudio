"""Read worker logs by byte offset and parse each line's record header.

A ``record`` line carries a Python logging or PyTorch glog header. A ``traceback``
line starts an interpreter traceback printed without a header. Other ``text`` lines
belong to the record above them (traceback frames, multi-line messages, progress
bars); the log view groups them because a record can span two reads.
"""

from __future__ import annotations

import re
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
_TRACEBACK = re.compile(_RANK + r"\s*Traceback \(most recent call last\):")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_LEVELS = {"warning": "warn", "critical": "error", "fatal": "error"}
_GLOG_LEVELS = {"I": "info", "W": "warn", "E": "error", "F": "error"}

MAX_READ = 512 * 1024


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
    except ValueError:
        return None


def parse_log_lines(lines: list[str], *, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now()
    out = []
    for raw in lines:
        line = _clean(raw)
        if match := _RECORD.match(line):
            level = match["level"].lower()
            try:
                ts = datetime.fromisoformat(match["time"].replace(",", ".").replace("Z", "+00:00")).timestamp()
            except ValueError:
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
        if _TRACEBACK.match(line):
            out.append({"kind": "traceback", "ts": None, "level": "error", "source": None, "msg": line})
            continue
        level = "error" if re.search(r"\b\w*(?:Error|Exception):", line) else "info"
        if level == "info" and re.search(r"(?:^\s*warning\b|\b\w*Warning:)", line, re.IGNORECASE):
            level = "warn"
        out.append({"kind": "text", "ts": None, "level": level, "source": None, "msg": line})
    return out


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
