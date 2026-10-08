"""Atomic storage and preview-only application of portable speech recipes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ypuddin.tts.gpt_sovits.config import GptSovitsVersionConfig
from ypuddin.tts.issues import TtsIssue
from ypuddin.tts.preset_models import (
    GSV_PARAMETER_FIELDS,
    VOX_PARAMETER_FIELDS,
    TtsPreset,
    TtsPresetConfig,
    TtsPresetCreateBody,
    TtsPresetDocument,
    TtsPresetResolveBody,
    TtsPresetResolveResponse,
    TtsPresetUpdateBody,
)
from ypuddin.tts.version_config import (
    TtsConfigPathError,
    TtsSavedConfig,
    TtsVersionConfig,
    _identity,
    _location,
    _parent_identity,
    default_config,
)

from .db import new_id, now
from .errors import ApiError

_LOCK = threading.RLock()
_MAX_BYTES = 256 * 1024
_PORTABLE = TypeAdapter(TtsPresetConfig)
_SAVED = TypeAdapter(TtsSavedConfig)
_ID = re.compile(r"^tsp_[0-9a-f]{12}$")


@dataclass(frozen=True)
class _FileIdentity:
    stat: tuple[int, int, int, int, int]
    sha256: str


def _stat_identity(value) -> tuple[int, int, int, int, int]:
    return (*_identity(value), value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def validation_issues(errors: list[dict], prefix=()) -> list[dict]:
    result = []
    for item in errors:
        location = [*prefix, *item.get("loc", ())]
        if location and location[0] == "body":
            location.pop(0)
        if len(location) > 1 and location[0] == "config" and location[1] in {"voxcpm1.5", "gpt-sovits-v5"}:
            location.pop(1)
        result.append(TtsIssue(
            code="tts.validation." + item["type"], loc=location, message=item["msg"],
        ).model_dump())
    return result


def _invalid(exc: ValidationError, prefix=()) -> ApiError:
    return ApiError(
        "请检查预设参数。", code="tts.preset_invalid", status=422,
        details={"issues": validation_issues(exc.errors(), prefix)},
    )


def portable_config(config) -> TtsPresetConfig:
    raw = config.model_dump(mode="json")
    fields = VOX_PARAMETER_FIELDS if config.engine == "voxcpm1.5" else GSV_PARAMETER_FIELDS
    return _PORTABLE.validate_python({"engine": config.engine, **{key: raw[key] for key in fields}})


def defaults(engine: str) -> TtsPresetConfig:
    return portable_config(default_config(engine))


def _missing() -> ApiError:
    return ApiError("语音预设不存在。", code="tts.preset_not_found", status=404)


def _directory(c: Any) -> Path:
    current = Path(c.data_root)
    for name in ("tts", "presets"):
        # Open the parent before creating a child, including on an empty installation.
        with _location(current / name) as location:
            try:
                if location.parent_fd is not None:
                    os.mkdir(name, mode=0o700, dir_fd=location.parent_fd)
                else:
                    os.mkdir(current / name, mode=0o700)
            except FileExistsError:
                pass
            location.verify_parents()
        current = current / name
        _parent_identity(current)
    return current


@contextmanager
def _storage(c: Any):
    with _LOCK:
        try:
            yield _directory(c)
        except TtsConfigPathError as exc:
            raise ApiError("预设路径不可安全访问。", code="tts.preset_path", status=403) from exc
        except OSError as exc:
            raise ApiError("无法读写语音预设。", code="tts.preset_io", status=500) from exc


def _path(directory: Path, id_: str) -> Path:
    if not _ID.fullmatch(id_):
        raise _missing()
    return directory / f"{id_}.json"


def _read_bound(path: Path) -> tuple[TtsPreset, _FileIdentity]:
    with _location(path) as location:
        before = location.target_info()
        if before is None:
            raise _missing()
        if before.st_size > _MAX_BYTES:
            raise ApiError("语音预设文件过大。", code="tts.preset_corrupt", status=409)
        descriptor = location.open(path.name, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                raise TtsConfigPathError("预设文件在读取时发生变化。")
            content = stream.read(_MAX_BYTES + 1)
        location.verify_parents()
        after = location.target_info()
        if after is None or _stat_identity(after) != _stat_identity(before):
            raise TtsConfigPathError("预设文件在读取时发生变化。")
        if len(content) > _MAX_BYTES:
            raise ApiError("语音预设文件过大。", code="tts.preset_corrupt", status=409)
        try:
            value = TtsPreset.model_validate_json(content)
        except ValidationError as exc:
            raise ApiError("语音预设文件无法读取。", code="tts.preset_corrupt", status=409) from exc
        if value.id != path.stem:
            raise ApiError("语音预设标识与文件不一致。", code="tts.preset_corrupt", status=409)
        return value, _FileIdentity(_stat_identity(after), hashlib.sha256(content).hexdigest())


def _read(path: Path) -> TtsPreset:
    return _read_bound(path)[0]


def _unchanged(path: Path, expected: _FileIdentity) -> None:
    try:
        _, current = _read_bound(path)
    except ApiError as exc:
        if exc.code not in {"tts.preset_not_found", "tts.preset_corrupt"}:
            raise
        current = None
    if current != expected:
        raise ApiError("预设在操作期间发生变化，请重新读取。", code="tts.preset_conflict", status=409)


def _rows(directory: Path) -> list[TtsPreset]:
    return [_read(path) for path in sorted(directory.glob("tsp_*.json"))]


def _unique(directory: Path, engine: str, name: str, *, except_id: str | None = None) -> None:
    if any(row.id != except_id and row.config.engine == engine and row.name.casefold() == name.casefold()
           for row in _rows(directory)):
        raise ApiError("此引擎已有同名预设。", code="tts.preset_duplicate", status=409)


def _write(path: Path, row: TtsPreset, *, create: bool, expected: _FileIdentity | None = None) -> None:
    contents = (json.dumps(row.model_dump(mode="json"), ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    if len(contents) > _MAX_BYTES:
        raise ApiError("语音预设内容过大。", code="tts.preset_invalid", status=422)
    with _location(path) as location:
        present = location.target_info()
        if create and present is not None:
            raise ApiError("预设标识已存在，请重新保存。", code="tts.preset_conflict", status=409)
        if not create:
            if expected is None:
                raise ValueError("preset update needs its read identity")
            _unchanged(path, expected)
        temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
        descriptor = location.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            location.verify_parents()
            latest = location.target_info()
            if (present is None) != (latest is None) or (
                present is not None and latest is not None and
                _stat_identity(present) != _stat_identity(latest)
            ):
                raise ApiError("预设在保存时发生变化，请重新读取。", code="tts.preset_conflict", status=409)
            if expected is not None:
                _unchanged(path, expected)
            if location.parent_fd is not None:
                os.replace(temporary, path.name, src_dir_fd=location.parent_fd, dst_dir_fd=location.parent_fd)
            else:
                os.replace(path.parent / temporary, path)
            location.verify_parents()
        finally:
            try:
                location.remove(temporary)
            except FileNotFoundError:
                pass


def list_presets(c: Any, engine: str | None = None) -> list[TtsPreset]:
    with _storage(c) as directory:
        return sorted(
            (row for row in _rows(directory) if engine is None or row.config.engine == engine),
            key=lambda row: (row.name.casefold(), row.id),
        )


def get_preset(c: Any, id_: str) -> TtsPreset:
    with _storage(c) as directory:
        return _read(_path(directory, id_))


def _create(c: Any, document: TtsPresetDocument) -> TtsPreset:
    with _storage(c) as directory:
        _unique(directory, document.config.engine, document.name)
        timestamp = now()
        row = TtsPreset(
            **document.model_dump(mode="json"), id=new_id("tsp"), revision=1,
            created_at=timestamp, updated_at=timestamp,
        )
        _write(_path(directory, row.id), row, create=True)
        return row


def create_preset(c: Any, body: TtsPresetCreateBody) -> TtsPreset:
    return _create(c, TtsPresetDocument(
        format="ypuddin-tts-preset", schema_version=1, name=body.name,
        description=body.description, config=portable_config(body.config),
    ))


def import_preset(c: Any, body: TtsPresetDocument) -> TtsPreset:
    return _create(c, body)


def export_preset(c: Any, id_: str) -> TtsPresetDocument:
    row = get_preset(c, id_)
    return TtsPresetDocument.model_validate(row.model_dump(exclude={"id", "revision", "created_at", "updated_at"}))


def _revision(row: TtsPreset, expected: int) -> None:
    if row.revision != expected:
        raise ApiError(
            "预设已更新，请重新读取后操作。", code="tts.preset_conflict", status=409,
            details={"current_revision": row.revision},
        )


def update_preset(c: Any, id_: str, body: TtsPresetUpdateBody) -> TtsPreset:
    with _storage(c) as directory:
        path = _path(directory, id_)
        row, identity = _read_bound(path)
        _revision(row, body.expected_revision)
        if row.config.engine != body.config.engine:
            raise ApiError("不能更改已有预设的训练引擎。", code="tts.preset_engine_mismatch", status=409)
        _unique(directory, body.config.engine, body.name, except_id=id_)
        changed = {
            "name": body.name, "description": body.description,
            "config": portable_config(body.config).model_dump(mode="json"),
        }
        if all(row.model_dump(mode="json")[key] == value for key, value in changed.items()):
            _unchanged(path, identity)
            return row
        replacement = TtsPreset.model_validate({
            **row.model_dump(mode="json"), **changed, "revision": row.revision + 1, "updated_at": now(),
        })
        _write(path, replacement, create=False, expected=identity)
        return replacement


def delete_preset(c: Any, id_: str, expected_revision: int) -> None:
    with _storage(c) as directory:
        path = _path(directory, id_)
        row, identity = _read_bound(path)
        _revision(row, expected_revision)
        with _location(path) as location:
            location.target_info()
            _unchanged(path, identity)
            location.remove(path.name)
            location.verify_parents()


def _overlay(base: dict, patch: dict) -> dict:
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = _overlay(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _changed(before: dict, after: dict, prefix: str = "") -> list[str]:
    paths = []
    for key, value in after.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(before.get(key), dict):
            paths.extend(_changed(before[key], value, path))
        elif key not in before or before[key] != value:
            paths.append(path)
    return paths


def resolve_preset(c: Any, id_: str, body: TtsPresetResolveBody) -> TtsPresetResolveResponse:
    row = get_preset(c, id_)
    current = body.config
    engine = current.get("engine")
    if not isinstance(engine, str) or engine not in {"voxcpm1.5", "gpt-sovits-v5"}:
        raise ApiError(
            "请选择语音训练引擎。", code="tts.preset_invalid", status=422,
            details={"issues": [TtsIssue(code="tts.validation.engine", loc=["config", "engine"], message="请选择语音训练引擎。").model_dump()]},
        )
    if engine != row.config.engine:
        raise ApiError("预设与当前版本的训练引擎不一致。", code="tts.preset_engine_mismatch", status=409)
    model = TtsVersionConfig if engine == "voxcpm1.5" else GptSovitsVersionConfig
    unknown = set(current) - set(model.model_fields)
    if unknown:
        raise ApiError(
            "语音配置包含不支持的字段。", code="tts.preset_invalid", status=422,
            details={"issues": [TtsIssue(code="tts.validation.extra_forbidden", loc=["config", key], message="不支持此配置字段。").model_dump() for key in sorted(unknown)]},
        )
    merged = _overlay(current, row.config.model_dump(mode="json"))
    try:
        config = _SAVED.validate_python(merged)
    except ValidationError as exc:
        raise _invalid(exc, ("config",)) from exc
    return TtsPresetResolveResponse(
        preset_id=row.id, preset_revision=row.revision, config=config,
        changed_fields=_changed(current, config.model_dump(mode="json")),
    )
