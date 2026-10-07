"""Version-owned speech configuration and compare-and-swap persistence."""

from __future__ import annotations

from typing import Any

from ypuddin.tts.version_config import (
    TtsConfigEnvelope,
    TtsConfigPathError,
    TtsConfigResponse,
    TtsConfigSaveBody,
    TtsConfigScope,
    read_envelope,
    write_envelope,
)

from .db import now
from .errors import ApiError
from .project_deletion import deleting
from .versions import assert_version_writable


def empty_audio_stats() -> dict:
    return {"train": {"state": "missing", "clips_count": None, "duration_seconds": None}, "validation": None}


def _read(c: Any, pid: str, vid: str) -> TtsConfigEnvelope:
    try:
        return read_envelope(c.tts_config_path(pid, vid))
    except TtsConfigPathError as exc:
        raise ApiError("语音配置路径不可安全访问。", code="tts.path_denied", status=403) from exc
    except FileNotFoundError as exc:
        raise ApiError("语音版本配置不存在。", code="tts.config_missing", status=409) from exc
    except ValueError as exc:
        raise ApiError("语音版本配置无法读取，请检查配置文件。", code="tts.config_invalid", status=409) from exc
    except OSError as exc:
        raise ApiError("无法读取语音版本配置。", code="tts.config_io", status=500) from exc


def _response(pid: str, version: dict, saved: TtsConfigEnvelope) -> TtsConfigResponse:
    return TtsConfigResponse(
        scope=TtsConfigScope(project_id=pid, version_id=version["id"]),
        revision=saved.revision,
        data_revision=version["data_revision"],
        config=saved.config,
    )


def get_config(c: Any, pid: str, vid: str) -> TtsConfigResponse:
    with c.db.lock:
        c.require_project_type(pid, "tts")
        version = c.resolve_version(pid, vid)
        if deleting(c, pid):
            raise ApiError("项目正在删除。", code="project.deleting", status=409)
        if version["status"] != "ready":
            raise ApiError("版本配置尚未就绪。", code="version.busy", status=409)
        return _response(pid, version, _read(c, pid, vid))


def save_config(c: Any, pid: str, vid: str, body: TtsConfigSaveBody) -> TtsConfigResponse:
    with c.db.lock:
        c.require_project_type(pid, "tts")
        if deleting(c, pid):
            raise ApiError("项目正在删除。", code="project.deleting", status=409)
        version = assert_version_writable(c, pid, vid)
        saved = _read(c, pid, vid)
        if saved.revision != body.expected_revision:
            raise ApiError(
                "配置已更新，请重新读取后保存。",
                code="tts.config_conflict",
                status=409,
                details={"current_revision": saved.revision},
            )
        if saved.config != body.config:
            saved = TtsConfigEnvelope(revision=saved.revision + 1, config=body.config)
            try:
                write_envelope(c.tts_config_path(pid, vid), saved)
            except TtsConfigPathError as exc:
                raise ApiError("语音配置路径不可安全访问。", code="tts.path_denied", status=403) from exc
            except OSError as exc:
                raise ApiError("无法保存语音版本配置。", code="tts.config_io", status=500) from exc
            c.db.update("project_versions", vid, {"updated_at": now()})
            c.bus.publish("version.changed", {"project_id": pid, "version_id": vid})
        return _response(pid, version, saved)
