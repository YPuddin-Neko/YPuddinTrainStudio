"""Read-only speech-manifest scans with complete diagnostics and byte identities."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .issues import TtsIssue
from .source_models import TtsSourceSummary

Allowed = Callable[[Path], bool]
SOURCE_ENGINES = {"voxcpm1.5", "gpt-sovits-v5"}
GSV_LANGUAGES = {"zh", "en", "ja", "ko", "yue"}


def _source_engine(engine: str) -> None:
    if engine not in SOURCE_ENGINES:
        raise SourceFileError("tts.engine.unsupported", "不支持这一语音训练引擎。")


class SourceFileError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _linked(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _stat_key(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def absolute_path(path: str | Path, *, base: Path | None = None) -> Path:
    if "\0" in str(path):
        raise SourceFileError("tts.path_invalid", "文件路径不能包含空字符。")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        if base is None:
            raise SourceFileError("tts.path_invalid", "请使用服务器上的绝对路径。")
        candidate = base / candidate
    # Keep lexical identity; resolve() would hide a symbolic link before checking it.
    return Path(os.path.abspath(candidate))


def _parents(path: Path) -> tuple[tuple[Path, tuple[int, int]], ...]:
    result = []
    for parent in reversed(path.parents):
        info = parent.lstat()
        if _linked(info) or not stat.S_ISDIR(info.st_mode):
            raise SourceFileError("tts.path_denied", "音频来源目录不能使用链接或目录重定向。")
        result.append((parent, (info.st_dev, info.st_ino)))
    return tuple(result)


@contextmanager
def open_source(path: Path, allowed: Allowed) -> Iterator[BinaryIO]:
    """Read a regular file through an anchored parent on supported POSIX platforms."""
    path = absolute_path(path)
    if not allowed(path):
        raise SourceFileError("tts.path_denied", "文件不在当前允许访问的目录中。")
    parents = _parents(path)
    before = path.lstat()
    if _linked(before) or not stat.S_ISREG(before.st_mode):
        raise SourceFileError("tts.path_denied", "音频来源必须是普通文件，不能使用链接或特殊文件。")
    parent_fd = None
    try:
        use_fd = os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW")
        if use_fd:
            for parent, identity in parents:
                child = os.open(
                    parent if parent_fd is None else parent.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                if parent_fd is not None:
                    os.close(parent_fd)
                parent_fd = child
                info = os.fstat(child)
                if (info.st_dev, info.st_ino) != identity:
                    raise SourceFileError("tts.source_stale", "文件目录在读取期间发生变化。")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path.name if parent_fd is not None else path, flags, dir_fd=parent_fd)
        with os.fdopen(fd, "rb") as stream:
            current = os.fstat(stream.fileno())
            if not stat.S_ISREG(current.st_mode) or _stat_key(current) != _stat_key(before):
                raise SourceFileError("tts.source_stale", "文件在读取期间发生变化。")
            yield stream
            if _stat_key(os.fstat(stream.fileno())) != _stat_key(before):
                raise SourceFileError("tts.source_stale", "文件在读取期间发生变化。")
        if _parents(path) != parents or not allowed(path):
            raise SourceFileError("tts.path_denied", "文件目录或访问权限在读取期间发生变化。")
        after = path.lstat()
        if _linked(after) or _stat_key(after) != _stat_key(before):
            raise SourceFileError("tts.source_stale", "文件在读取期间发生变化。")
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _digest(stream: BinaryIO) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    while block := stream.read(1024 * 1024):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def file_identity(path: Path, allowed: Allowed) -> dict[str, Any]:
    with open_source(path, allowed) as stream:
        digest, size = _digest(stream)
    return {"path": str(path), "sha256": digest, "size": size}


def _issue(code: str, loc: list[str | int], message: str, *, warning: bool = False) -> TtsIssue:
    return TtsIssue(code=code, loc=loc, message=message, severity="warning" if warning else "error")


def _json(text: str) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"JSON 中不能包含 {value}。")
    return json.loads(text, parse_constant=invalid)


def inspect_audio(path: Path, allowed: Allowed, *, engine: str = "voxcpm1.5") -> tuple[dict, dict, list[tuple[str, str, bool]]]:
    """Return frame metadata, a byte identity and all independent format issues."""
    _source_engine(engine)
    problems: list[tuple[str, str, bool]] = []
    metadata = {"duration_seconds": None, "sample_rate": None, "channels": None}
    with open_source(path, allowed) as stream:
        digest, size = _digest(stream)
        identity = {"path": str(path), "sha256": digest, "size": size}
        stream.seek(0)
        if path.suffix.lower() != ".wav":
            problems.append(("tts.audio.format", "录音须为 PCM WAV。", False))
        try:
            with wave.open(stream, "rb") as audio:
                channels, rate, width, frames = (
                    audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getnframes()
                )
                metadata.update(sample_rate=rate, channels=channels)
                if channels != 1:
                    problems.append(("tts.audio.channels", "录音须为单声道。", False))
                rates = {32_000, 44_100, 48_000} if engine == "gpt-sovits-v5" else {44_100}
                if rate not in rates:
                    message = "录音采样率须为 32000、44100 或 48000 Hz。" if engine == "gpt-sovits-v5" else "录音采样率须为 44100 Hz。"
                    problems.append(("tts.audio.sample_rate", message, False))
                if frames <= 0 or width not in {1, 2, 3, 4}:
                    problems.append(("tts.audio.frames", "录音没有有效 PCM 帧。", False))
                read = 0
                while block := audio.readframes(65_536):
                    read += len(block)
                if read != frames * channels * width:
                    problems.append(("tts.audio.truncated", "录音数据不完整。", False))
                if not problems and rate:
                    metadata["duration_seconds"] = frames / rate
                    if frames / rate < 1:
                        problems.append(("tts.audio.short", "录音短于 1 秒，发音上下文可能不足。", True))
                    elif frames / rate > 30:
                        problems.append(("tts.audio.long", "录音长于 30 秒，训练时会增加显存占用。", True))
        except (wave.Error, EOFError) as exc:
            problems.append(("tts.audio.format", f"无法读取 PCM WAV：{exc}", False))
    return metadata, identity, problems


@dataclass
class ScanResult:
    fingerprint: str
    identities: list[dict]
    rows: list[dict]
    summary: TtsSourceSummary
    issues: list[TtsIssue]
    issues_total: int
    issues_truncated: bool
    engine: str = "voxcpm1.5"


def _manifest_row(text: str, *, native_list: bool) -> Any:
    if native_list:
        parts = text.split("|")
        if len(parts) != 4:
            raise SourceFileError("tts.row.invalid_list", "这一行须为录音路径|说话人|语言|转写文本，字段中不能包含 |。")
        audio, speaker, language, transcript = parts
        return {"audio": audio, "speaker": speaker, "language": language, "text": transcript}
    return _json(text)


def scan_manifest(path: Path, source_id: str, split: str, *, allowed: Allowed, engine: str = "voxcpm1.5") -> ScanResult:
    """Collect every nonblank physical row, including invalid JSON and invalid audio."""
    _source_engine(engine)
    path = absolute_path(path)
    gsv = engine == "gpt-sovits-v5"
    native_list = gsv and path.suffix.lower() == ".list"
    with open_source(path, allowed) as stream:
        raw = stream.read()
    manifest_identity = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    content = raw.decode("utf-8-sig")
    identities = {str(path): manifest_identity}
    rows: list[dict] = []
    all_issues: list[TtsIssue] = []
    valid_count, duration = 0, 0.0
    # One recording can appear on several training rows without duplicating file reads.
    audio_cache: dict[str, tuple[dict, dict, list[tuple[str, str, bool]]]] = {}
    for line, text in enumerate(content.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        if not text.strip():
            continue
        loc: list[str | int] = ["sources", source_id, "rows", line]
        issues: list[TtsIssue] = []
        item: dict[str, Any] = {
            "line": line, "text": None, "audio_name": None, "reference_audio_name": None,
            "dataset_id": None, "duration_seconds": None, "sample_rate": None, "channels": None,
        }
        normalized: dict[str, Any] = {}
        assets: dict[str, dict] = {}
        try:
            value = _manifest_row(text, native_list=native_list)
        except SourceFileError as exc:
            value = None
            issues.append(_issue(exc.code, loc, str(exc)))
        except ValueError:
            value = None
            issues.append(_issue("tts.row.invalid_json", loc, "这一行不是有效 JSON。"))
        if value is not None and not isinstance(value, dict):
            issues.append(_issue("tts.row.object_required", loc, "这一行须为包含 audio、text 的 JSON 对象。"))
        elif value is None and not issues:
            issues.append(_issue("tts.row.object_required", loc, "这一行须为包含 audio、text 的 JSON 对象。"))
        if isinstance(value, dict):
            if isinstance(value.get("text"), str) and value["text"].strip():
                item["text"] = normalized["text"] = value["text"]
                if gsv and any(character in value["text"] for character in "|\r\n\t\0"):
                    issues.append(_issue("tts.text.invalid", loc + ["text"], "转写文本不能包含换行、制表符、空字符或 |。"))
            else:
                issues.append(_issue("tts.text.required", loc + ["text"], "缺少转写文本。"))
            if gsv:
                language = value.get("language")
                if isinstance(language, str) and language.lower() in GSV_LANGUAGES:
                    item["language"] = normalized["language"] = language.lower()
                else:
                    issues.append(_issue("tts.language.invalid", loc + ["language"], "语言须为 zh、en、ja、ko 或 yue。"))
                speaker = value.get("speaker", "speaker")
                if isinstance(speaker, str) and speaker.strip() and not any(character in speaker for character in "|\r\n\t\0"):
                    item["speaker"] = normalized["speaker"] = speaker
                else:
                    issues.append(_issue("tts.speaker.invalid", loc + ["speaker"], "说话人不能为空，也不能包含换行、制表符、空字符或 |。"))
                for field in ("ref_audio", "dataset_id"):
                    if field in value:
                        issues.append(_issue("tts.row.field_unsupported", loc + [field], f"GPT-SoVITS 数据清单不支持 {field}。"))
            else:
                dataset_id = value.get("dataset_id", 0)
                if type(dataset_id) is int and dataset_id >= 0:
                    item["dataset_id"] = normalized["dataset_id"] = dataset_id
                else:
                    issues.append(_issue("tts.dataset_id.invalid", loc + ["dataset_id"], "dataset_id 须为非负整数。"))
            for key, name in (("audio", "audio_name"), ("ref_audio", "reference_audio_name")):
                if gsv and key == "ref_audio":
                    continue
                audio_value = value.get(key)
                if key == "ref_audio" and audio_value is None:
                    continue
                if not isinstance(audio_value, str) or not audio_value.strip():
                    issues.append(_issue("tts.audio.required", loc + [key], "缺少录音路径。"))
                    continue
                item[name] = Path(audio_value).name
                if gsv and any(character in audio_value for character in "|\r\n\t\0"):
                    issues.append(_issue("tts.path_invalid", loc + [key], "录音路径不能包含换行、制表符、空字符或 |。"))
                    continue
                audio = None
                try:
                    audio = absolute_path(audio_value, base=path.parent)
                    if str(audio) not in audio_cache:
                        audio_cache[str(audio)] = inspect_audio(audio, allowed, engine=engine)
                    info, identity, problems = audio_cache[str(audio)]
                    identities[str(audio)] = identity
                    issues.extend(_issue(code, loc + [key], message, warning=warning) for code, message, warning in problems)
                    if key == "audio":
                        item.update(info)
                    if not any(not warning for _, _, warning in problems):
                        assets[key] = identity
                        normalized[key] = str(audio)
                        normalized["duration" if key == "audio" else "ref_duration"] = info["duration_seconds"]
                except (OSError, SourceFileError) as exc:
                    code = exc.code if isinstance(exc, SourceFileError) else (
                        "tts.audio.not_found" if isinstance(exc, FileNotFoundError) else "tts.audio.read_failed"
                    )
                    message = "录音文件不存在。" if isinstance(exc, FileNotFoundError) else str(exc)
                    issues.append(_issue(code, loc + [key], message))
                    if audio is not None:
                        identities[str(audio)] = {"path": str(audio), "sha256": None, "size": None, "error": code}
        issues.sort(key=lambda issue: (issue.code, json.dumps(issue.loc, ensure_ascii=False)))
        valid = not any(issue.severity == "error" for issue in issues)
        if valid:
            valid_count += 1
            duration += item["duration_seconds"]
        rows.append({"row": item, "issues": issues, "normalized": normalized if valid else None, "assets": assets})
        all_issues.extend(issues)
    if not rows:
        all_issues.insert(0, _issue("tts.manifest_empty", ["sources", split, "path"], "数据清单没有非空样本行。"))
    ordered_identities = [identities[key] for key in sorted(identities)]
    identity_payload = {"scanner": 1, "files": ordered_identities}
    if gsv:
        identity_payload["engine"] = engine
    fingerprint = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest()
    return ScanResult(
        fingerprint=fingerprint, identities=ordered_identities, rows=rows,
        summary=TtsSourceSummary(
            clips_count=len(rows), valid_clips_count=valid_count, invalid_count=len(rows) - valid_count,
            duration_seconds=duration,
        ),
        issues=all_issues[:100], issues_total=len(all_issues), issues_truncated=len(all_issues) > 100, engine=engine,
    )


def identity_changes(identities: list[dict], *, allowed: Allowed) -> list[dict]:
    """Rehash files without decoding WAVs; unavailable files remain explicit identities."""
    changes = []
    for expected in identities:
        path = Path(expected["path"])
        try:
            actual = file_identity(path, allowed)
        except (OSError, SourceFileError) as exc:
            code = exc.code if isinstance(exc, SourceFileError) else (
                "tts.audio.not_found" if isinstance(exc, FileNotFoundError) else "tts.audio.read_failed"
            )
            actual = {"path": str(path), "sha256": None, "size": None, "error": code}
        if actual != expected:
            changes.append(actual)
    return changes


def manifest_references(path: Path, *, allowed: Allowed) -> list[Path]:
    """Find dependencies without reading audio; ambiguous manifests cannot prove safety."""
    with open_source(path, allowed) as stream:
        content = stream.read().decode("utf-8-sig")
    paths = {path}
    for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        # Dependency protection must keep working after an engine change and before rescanning.
        try:
            value = _json(line)
        except ValueError:
            if path.suffix.lower() != ".list":
                raise
            value = _manifest_row(line, native_list=True)
        if not isinstance(value, dict):
            raise SourceFileError("tts.references_unresolved", "清单包含无法确定录音引用的行。")
        for key in ("audio", "ref_audio"):
            value_path = value.get(key)
            if value_path is None and key == "ref_audio":
                continue
            if not isinstance(value_path, str) or not value_path.strip():
                raise SourceFileError("tts.references_unresolved", "清单包含无法确定录音引用的行。")
            paths.add(absolute_path(value_path, base=path.parent))
    return sorted(paths)
