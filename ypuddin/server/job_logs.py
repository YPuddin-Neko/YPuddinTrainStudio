"""Read worker logs by byte offset and parse each line's record header.

A ``record`` line carries a logging header or independently classified native output.
A ``traceback`` line starts an interpreter traceback printed without a header. Other ``text`` lines
are independent output unless their syntax identifies a continuation. The log view groups
continuations over all loaded lines because a record can span two reads.
"""

from __future__ import annotations

import json
import re
import unicodedata
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
_FATAL_PYTHON = re.compile(r"^\s*Fatal Python error:", re.IGNORECASE)
_THREAD_DUMP = re.compile(
    r"^\s*(?:(?:Current thread|Thread)\s+0x[\da-f]+|Stack \(most recent call first\):|Extension modules:)",
    re.IGNORECASE,
)
_EXCEPTION_END = re.compile(
    r"^(?:[A-Za-z_]\w*\.)*(?:(?:[A-Za-z_]\w*)?(?:Error|Exception|Warning|Failure)"
    r"|KeyboardInterrupt|SystemExit|GeneratorExit|Stop(?:Async)?Iteration)(?::(?:\s|$)|$)"
)
_TRACEBACK_FRAME = re.compile(r'^\s+File ".+", line \d+(?:, in .*)?$')
_TRACEBACK_PROGRESS = re.compile(
    r"^(?:(?:Epoch|Total):\s*\d+(?:\s*/\s*\d+)?\s*|Loading(?:\s.*|:.*)?"
    r"|(?:(?:phoneme_data_len|wav_data_len|skipped_phone|skipped_dur):\s*\d+\s*)"
    r"(?:,\s*(?:phoneme_data_len|wav_data_len|skipped_phone|skipped_dur):\s*\d+\s*)*"
    r"|[^\s:]+\.(?:wav|flac|mp3|ogg|m4a|aac))$",
    re.IGNORECASE,
)
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
    r"ProcessGroupNCCL (?:initialization options|environments):(?:\s+|$)(?P<fields>.*)$"
)
_NCCL_NUMBER_FIELDS = (
    "size", "global rank", "PG Name", "TIMEOUT(ms)", "USE_HIGH_PRIORITY_STREAM", "SPLIT_FROM", "SPLIT_COLOR",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_DUMP_ON_TIMEOUT", "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC",
    "TORCH_NCCL_DESYNC_DEBUG", "TORCH_NCCL_ENABLE_TIMING", "TORCH_NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK", "TORCH_NCCL_ENABLE_MONITORING",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "TORCH_NCCL_TRACE_BUFFER_SIZE", "TORCH_NCCL_COORD_CHECK_MILSEC",
    "TORCH_NCCL_NAN_CHECK", "TORCH_NCCL_CUDA_EVENT_CACHE", "TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN",
)
_NCCL_FIELD = re.compile(
    r"(?:" + "|".join(re.escape(field) for field in _NCCL_NUMBER_FIELDS) + r"):\s*\d+"
    r"|NCCL version:\s*\d+(?:\.\d+){1,3}|TORCH_DISTRIBUTED_DEBUG:\s*(?:OFF|INFO|DETAIL)"
)
_NCCL_INITIALIZATION = re.compile(
    r"(?:\[rank\d+\]:\s*)?\[PG ID \d+ PG GUID \S+ Rank \d+\] "
    r"(?:ProcessGroupNCCL broadcast unique ID through store took \d+(?:\.\d+)?(?:[eE][+-]?\d+)? ms"
    r"|ProcessGroupNCCL created ncclComm_ 0x[\da-fA-F]+ on CUDA device: (?:\d+|[\x00-\x07])"
    r"|NCCL_DEBUG: N/A)\s*"
)
_NCCL_NATIVE = re.compile(
    _RANK + r"\s*(?:[^\s:]+:\d+:\d+\s+)?(?:\[\d+\]\s+)?"
    r"(?:[\w./\\-]+:\d+\s+)?NCCL (?P<level>WARN|ERROR|FATAL)\s+(?P<message>.*)$"
)
_NCCL_DEVICE_FAILURE = re.compile(
    r"\b(?:Cuda|HIP) failure\b|\b(?:cudaErrorMemoryAllocation|hipErrorOutOfMemory|illegal memory access)\b"
    r"|\bunhandled (?:cuda|hip) error\b|^out of memory(?:\W|$)",
    re.IGNORECASE,
)
_NATIVE_CRASH = re.compile(
    _RANK + r"\s*(?:Memory access fault by GPU\b|Fatal Python error:"
    r"|(?:(?:bash|zsh|sh|fish):\s*)?Segmentation fault\b"
    r"|(?:[^\n]+:\s+line \d+:\s+|(?:\[\d+\][+-]?\s+)?)(?:\d+\s+)Segmentation fault\b)",
    re.IGNORECASE,
)
_HYLOG_DIRECTORY_WARNING = re.compile(_RANK + r"\s*Could not open /var/log/hylog/\.\s*")

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


def _raw_event_level(line: str) -> str | None:
    body = _PROCESS_RANK.sub("", line, count=1)
    if not body.startswith("{"):
        return None

    def unique_fields(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("duplicate JSON field")
        return value

    try:
        value = json.loads(body, object_pairs_hook=unique_fields)
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("event"), str):
        return None
    # Extra fields may carry diagnostics; only this complete progress-only schema is quiet.
    if (
        value.keys() == {"event", "step", "total"}
        and value["event"] == "progress"
        and type(value["step"]) is int and type(value["total"]) is int
        and 0 <= value["step"] <= value["total"] and value["total"] > 0
    ):
        return "debug"
    return "info"


def _nccl_configuration(line: str) -> bool:
    header = _NCCL_CONFIGURATION.fullmatch(line)
    if header:
        body = header["fields"].strip()
        if not body:
            return True
    else:
        body = _PROCESS_RANK.sub("", line, count=1)
        if body[:1].isspace():
            return False
    fields = [field.strip() for field in body.split(",")]
    return all(_NCCL_FIELD.fullmatch(field) for field in fields) and (
        header is not None or any(field.partition(":")[0] not in {"size", "global rank", "PG Name"} for field in fields)
    )


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
        # Native initialization notices bypass Python logging; retain their raw file and source.
        if line["level"] == "info" and (
            _nccl_configuration(line["msg"]) or _NCCL_INITIALIZATION.fullmatch(line["msg"])
        ):
            line["kind"] = "record"
            line["level"] = "debug"
            line["standalone"] = True
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
        if match := _NCCL_NATIVE.match(line):
            level = "warn" if match["level"] == "WARN" else "error"
            if _NCCL_DEVICE_FAILURE.search(match["message"]):
                level = "error"
            out.append({
                "kind": "record", "ts": None, "level": level,
                "source": "NCCL", "msg": line, "standalone": True,
            })
            continue
        if _NATIVE_CRASH.match(line):
            out.append({
                "kind": "record", "ts": None, "level": "error", "source": None, "msg": line,
                # Python fatal errors print a thread dump after the header; keep it attached.
                **({} if re.match(_RANK + r"\s*Fatal Python error:", line, re.IGNORECASE) else {"standalone": True}),
            })
            continue
        if _HYLOG_DIRECTORY_WARNING.fullmatch(line):
            out.append({
                "kind": "record", "ts": None, "level": "warn", "source": None,
                "msg": line, "standalone": True,
            })
            continue
        if _TRACEBACK.match(line):
            out.append({"kind": "traceback", "ts": None, "level": "error", "source": None, "msg": line})
            continue
        if level := _raw_event_level(line):
            out.append({
                "kind": "record", "ts": None, "level": level, "source": None,
                "msg": line, "standalone": True,
            })
            continue
        level = "error" if re.search(r"\b\w*(?:Error|Exception):", line) else "info"
        if level == "info" and re.search(r"(?:^\s*warning\b|\b\w*Warning:)", line, re.IGNORECASE):
            level = "warn"
        out.append({"kind": "text", "ts": None, "level": level, "source": None, "msg": line})
    return out


def _exception_terminal(body: str, seen_frame: bool) -> bool:
    if _TRACEBACK_PROGRESS.fullmatch(body):
        return False
    if not seen_frame:
        return _EXCEPTION_END.match(body) is not None
    name, separator, message = body.partition(":")
    if separator and message and not message[:1].isspace():
        return False
    parts = name.split(".")
    return all(
        (part == "<locals>" and index < len(parts) - 1)
        or unicodedata.normalize("NFKC", part).isidentifier()
        for index, part in enumerate(parts)
    )


def entry_levels(lines: list[Mapping[str, Any]]) -> list[str]:
    """The level of the log entry each parsed line belongs to, grouped as the job log view groups them.

    A record keeps its own level for every line joined to it, so the traceback a warning was logged
    with (``exc_info``) reads as part of that warning. Any other traceback starts an error entry:
    one after an info or debug record, after plain output, or after a warning whose own traceback
    has ended. Keep in step with ``groupLogLines`` in ``frontend/src/utils/jobLogs.ts``.
    """
    levels: list[str] = []
    level = kind = rank = source = None
    header = ""
    standalone = False
    traceback = "none"
    seen_frame = False
    chained = False
    for line in lines:
        line_kind = line.get("kind") or "text"
        line_level = line.get("level") if line.get("level") in _ORDER else "info"
        message = str(line.get("msg") or "")
        found = _PROCESS_RANK.match(message)
        line_rank = found[1] if found else None
        body = message[found.end() :] if found else message
        different_rank = line_rank is not None and rank is not None and line_rank != rank
        line_source = line.get("source")
        different_source = bool(line_source and line_source != OUTPUT_SOURCE and line_source != source)
        if level is None or different_rank or different_source or standalone or line.get("standalone"):
            joins = False
        elif line_kind == "traceback":
            joins = (traceback == "none" or chained) and (
                level == "error" or (kind == "record" and level == "warn")
            )
        elif line_kind == "text":
            warning_head = traceback != "frames" and _WARNING_HEAD.match(message) is not None
            warning_detail = _WARNING_HEAD.match(header) is not None
            fatal_python = _FATAL_PYTHON.match(header) is not None
            indented = bool(body[:1].isspace() and body.strip())
            legacy_detail = line.get("ts") is None and not line_source and kind == "record"
            detail_header = kind == "record" or warning_detail or fatal_python or header.rstrip().endswith(":")
            text_detail = line_level == "info" and (
                legacy_detail
                or ((indented or not body.strip()) and detail_header)
                or (fatal_python and _THREAD_DUMP.match(body) is not None)
            )
            joins = not warning_head and (
                (traceback == "frames" and (not body.strip() or indented or _exception_terminal(body, seen_frame)))
                or (traceback == "done" and (not body.strip() or _CHAINED.match(body.strip()) is not None))
                or (traceback == "none" and text_detail)
            )
        else:
            joins = False
        if not joins:
            level, kind, rank, traceback = line_level, line_kind, line_rank, "none"
            source, header = line_source, body
            standalone = bool(line.get("standalone"))
            seen_frame = False
        elif rank is None and line_rank is not None:
            rank = line_rank
        if line_kind == "traceback":
            traceback = "frames"
            seen_frame = False
        elif traceback == "frames" and _TRACEBACK_FRAME.fullmatch(body):
            seen_frame = True
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
