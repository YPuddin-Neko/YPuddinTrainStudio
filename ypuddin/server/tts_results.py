"""Register speech products and read them without requiring writable project owners."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ypuddin.tts.issues import TtsIssue
from ypuddin.tts.result_models import (
    GptSovitsCheckpointInfo,
    TtsAudio,
    TtsCheckpoint,
    TtsCheckpointPage,
    TtsSampleJob,
    TtsSampleJobPage,
    TtsSampleRequestSnapshot,
    TtsSampleSource,
)
from ypuddin.tts.source_scan import SourceFileError, file_identity, open_source

from .db import now
from .errors import ApiError
from .job_paths import deleting_jobs, event_file, output_directory
from .tts_references import reject_deleting_references

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tts_checkpoints (
  id TEXT PRIMARY KEY, source_job_id TEXT NOT NULL, relative_path TEXT NOT NULL,
  revision TEXT NOT NULL, discovered_at REAL NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(source_job_id, relative_path)
);
CREATE INDEX IF NOT EXISTS idx_tts_checkpoints_source ON tts_checkpoints(source_job_id);
CREATE TABLE IF NOT EXISTS tts_sample_audio (
  id TEXT PRIMARY KEY, sample_job_id TEXT NOT NULL, event_key TEXT NOT NULL,
  record_json TEXT NOT NULL, UNIQUE(sample_job_id,event_key)
);
CREATE INDEX IF NOT EXISTS idx_tts_audio_job ON tts_sample_audio(sample_job_id);
"""
CHECKPOINT_FILES = ("lora_config.json", "lora_weights.safetensors", "lora_weights.ckpt")
GSV_CHECKPOINT_FILES = ("checkpoint.json", "gpt.ckpt", "sovits.pth")
STAT_FIELDS = ("device", "inode", "mtime_ns", "ctime_ns")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(prefix: str, value: Any) -> str:
    return prefix + "_" + hashlib.sha256(_json(value).encode()).hexdigest()[:32]


def _issue(code: str, message: str) -> TtsIssue:
    return TtsIssue(code=code, loc=[], message=message)


def _fail(code: str, message: str, status: int = 409) -> None:
    issue = _issue(code, message)
    raise ApiError(message, code=code, status=status, details={"issues": [issue.model_dump()]})


def _job(c: Any, jid: str, kind: str) -> dict:
    job = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    if job is None:
        code = "tts.sample_not_found" if kind == "tts_sample" else "tts.source_not_found"
        _fail(code, "语音任务不存在。", 404)
    if job["type"] != kind:
        _fail("tts.job_type_mismatch", "任务类型与请求不匹配。")
    if jid in deleting_jobs:
        _fail("job.deleting", "这个任务的文件正在删除。")
    return job


def _scope(c: Any, pid: str, vid: str) -> None:
    c.require_project_type(pid, "tts")
    c.resolve_version(pid, vid)


def _payload(job: dict) -> dict:
    try:
        value = json.loads(job.get("config_json") or "{}")
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value):
        return None
    return value


def _directory(c: Any, path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts or not c.is_allowed(path):
        raise SourceFileError("tts.path_denied", "结果不在允许访问的目录中。")
    for item in (path, *path.parents):
        info = item.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or (getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        ):
            raise SourceFileError("tts.path_denied", "结果目录不能使用链接或目录重定向。")
    return path


def _events(c: Any, job: dict) -> list[dict]:
    try:
        with open_source(event_file(job), c.is_allowed) as stream:
            lines = stream.read().decode("utf-8", errors="replace").split("\n")
    except (OSError, ValueError):
        return []
    result = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            result.append(event)
    return result


def _checkpoint_files(c: Any, root: Path, relative: str, *, verify: bool = True) -> list[dict]:
    _directory(c, root)
    path = Path(relative)
    if path.is_absolute() or not relative or ".." in path.parts or "\\" in relative:
        raise SourceFileError("tts.path_denied", "检查点路径无效。")
    target = _directory(c, root / path)
    if target == root:
        raise SourceFileError("tts.path_denied", "请选择来源任务中的检查点。")
    assets = []
    gsv = (target / "checkpoint.json").exists() or (target / "checkpoint.json").is_symlink()
    names_to_read = GSV_CHECKPOINT_FILES if gsv else CHECKPOINT_FILES
    for name in names_to_read:
        asset = target / name
        if not asset.exists() and not asset.is_symlink():
            continue
        with open_source(asset, c.is_allowed) as stream:
            info = os.fstat(stream.fileno())
            identity = {
                "path": str(asset),
                "size": info.st_size,
                "device": info.st_dev,
                "inode": info.st_ino,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }
            if verify:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                identity["sha256"] = digest.hexdigest()
                if name == "lora_config.json":
                    stream.seek(0)
                    if not isinstance(json.load(stream), dict):
                        raise ValueError("检查点 LoRA 配置不是有效 JSON 对象。")
                elif name == "checkpoint.json":
                    stream.seek(0)
                    metadata = json.load(stream)
                    if not isinstance(metadata, dict) or metadata.get("engine") != "gpt-sovits-v5":
                        raise ValueError("GPT-SoVITS 检查点配置无效。")
                    GptSovitsCheckpointInfo.model_validate({key: metadata.get(key) for key in GptSovitsCheckpointInfo.model_fields})
        identity["name"] = name
        assets.append(identity)
    names = {item["name"] for item in assets if item["size"] > 0}
    if gsv and names != set(GSV_CHECKPOINT_FILES):
        raise ValueError("检查点缺少配套 GPT、SoVITS 权重或配置。")
    if gsv and verify:
        expected = metadata.get("files", {})
        if not isinstance(expected, dict):
            raise ValueError("GPT-SoVITS 检查点缺少权重摘要。")
        for asset in assets:
            if asset["name"] == "checkpoint.json":
                continue
            identity = expected.get(asset["name"], {})
            if not isinstance(identity, dict) or asset["size"] != identity.get("size") or asset["sha256"] != identity.get("sha256"):
                raise ValueError("GPT-SoVITS 配对检查点的权重内容已变化。")
    if not gsv and ("lora_config.json" not in names or not names.intersection(CHECKPOINT_FILES[1:])):
        raise ValueError("检查点缺少 LoRA 配置或权重。")
    return assets


def _resource_issue(
    c: Any, root: str, relative: str, expected: list[dict] | None, *, verify=True
) -> TtsIssue | None:
    try:
        assets = _checkpoint_files(c, Path(root), relative, verify=verify)
        if expected is not None:
            fields = ("path", "size", "sha256") if verify else ("path", "size")
            found = [{key: item[key] for key in fields} for item in assets]
            wanted = [{key: item[key] for key in fields} for item in expected]
            if sorted(found, key=lambda item: item["path"]) != sorted(wanted, key=lambda item: item["path"]):
                return _issue("tts.checkpoint_stale", "检查点内容已变化，请重新选择。")
            if not verify:
                indexed = {item["path"]: item for item in assets}
                if any(
                    item[key] != indexed[item["path"]][key]
                    for item in expected
                    for key in STAT_FIELDS
                    if key in item
                ):
                    return _issue("tts.checkpoint_stale", "检查点文件身份已变化，请重新检查。")
        with c.db.lock:
            reject_deleting_references(c, [Path(item["path"]) for item in assets])
    except ApiError as exc:
        return _issue(exc.code, exc.message)
    except SourceFileError as exc:
        return _issue(exc.code, str(exc))
    except (OSError, ValueError, KeyError, TypeError):
        return _issue("tts.source_unavailable", "检查点配置或权重已不可读取。")
    return None


def frozen_checkpoint_issue(c: Any, sample_snapshot: dict, *, verify: bool = True) -> TtsIssue | None:
    root, value = sample_snapshot.get("source_output_dir"), sample_snapshot.get("checkpoint")
    if not isinstance(root, str) or not isinstance(value, str):
        return _issue("tts.source_unavailable", "试听记录缺少来源检查点。")
    path = Path(value)
    if path.name in CHECKPOINT_FILES[1:]:
        path = path.parent
    try:
        relative = path.relative_to(Path(root)).as_posix() if path.is_absolute() else path.as_posix()
    except ValueError:
        return _issue("tts.path_denied", "检查点不在来源任务的目录中。")
    return _resource_issue(c, root, relative, sample_snapshot.get("checkpoint_fingerprints"), verify=verify)


def frozen_checkpoint(c: Any, sample_snapshot: dict) -> bool:
    return frozen_checkpoint_issue(c, sample_snapshot) is None


def preview_issue(
    c: Any, source_job: dict | None, checkpoint: TtsCheckpoint, *, verify: bool = True
) -> TtsIssue | None:
    if source_job is None:
        return _issue("tts.source_not_found", "来源训练任务不存在。")
    current = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (source_job["id"],))
    if current is None:
        return _issue("tts.source_not_found", "来源训练任务不存在。")
    if current["type"] != "tts_train" or checkpoint.source_job_id != current["id"]:
        return _issue("tts.job_type_mismatch", "检查点不属于这个训练任务。")
    if current["id"] in deleting_jobs:
        return _issue("job.deleting", "来源训练任务的文件正在删除。")
    if current.get("archived_at") is not None:
        return _issue("job.archived", "来源训练任务已归档。")
    from .tts_job_actions import owner_issue

    if issue := owner_issue(c, current):
        return issue
    record = c.db.fetchone(
        "SELECT * FROM tts_checkpoints WHERE id=? AND source_job_id=?", (checkpoint.id, current["id"])
    )
    if record is None or record["revision"] != checkpoint.revision:
        return _issue("tts.checkpoint_stale", "检查点登记已变化，请重新选择。")
    saved = json.loads(record["record_json"])
    if str(output_directory(current)) != saved["root"]:
        return _issue("tts.checkpoint_stale", "来源任务的输出目录已变化。")
    return _resource_issue(c, saved["root"], checkpoint.relative_path, saved["fingerprints"], verify=verify)


def _checkpoint_event(root: Path, events: list[dict]) -> dict[str, dict]:
    result = {}
    for event in events:
        if event.get("type") != "checkpoint.saved" or not isinstance(event.get("path"), str):
            continue
        path = Path(event["path"])
        if path.name in CHECKPOINT_FILES[1:]:
            path = path.parent
        try:
            relative = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        except ValueError:
            continue
        if not relative or relative == "." or ".." in Path(relative).parts or "\\" in relative:
            continue
        result[relative] = event
    return result


def _register_checkpoints(c: Any, job: dict) -> None:
    root = output_directory(job)
    matches = _checkpoint_event(root, _events(c, job))
    relative_paths = set(matches)
    try:
        _directory(c, root)
        engine = _payload(job).get("tts", {}).get("engine", "voxcpm1.5")
        if engine == "gpt-sovits-v5":
            relative_paths.update(path.parent.relative_to(root).as_posix() for path in (root / "gpt_sovits").glob("*/checkpoint.json"))
        else:
            relative_paths.update(path.relative_to(root).as_posix() for path in (root / "voxcpm").glob("step_*"))
    except (OSError, ValueError):
        return
    for relative in sorted(relative_paths):
        try:
            assets = _checkpoint_files(c, root, relative)
        except (OSError, ValueError):
            continue
        event = matches.get(relative, {})
        gsv_info = None
        if any(item["name"] == "checkpoint.json" for item in assets):
            if engine != "gpt-sovits-v5":
                continue
            try:
                with open_source(root / relative / "checkpoint.json", c.is_allowed) as stream:
                    metadata = json.load(stream)
                gsv_info = GptSovitsCheckpointInfo.model_validate({key: metadata.get(key) for key in GptSovitsCheckpointInfo.model_fields})
                if gsv_info.variant != _payload(job)["tts"].get("variant") or gsv_info.stage != _payload(job)["tts"].get("stage"):
                    continue
            except (OSError, ValueError):
                continue
        elif engine == "gpt-sovits-v5":
            continue
        cid = _id("ckpt", [job["id"], relative])
        fingerprints = [
            {key: item[key] for key in ("path", "size", "sha256", *STAT_FIELDS)} for item in assets
        ]
        created = _number(event.get("ts"))
        metadata = {
            "step": _number(event.get("step"), integer=True),
            "upstream_step": _number(event.get("upstream_step"), integer=True),
            "created_at": created,
        }
        try:
            modified = max(Path(item["path"]).stat().st_mtime for item in assets)
        except OSError:
            continue
        with c.db.lock:
            previous = c.db.fetchone("SELECT * FROM tts_checkpoints WHERE id=?", (cid,))
            discovered = previous["discovered_at"] if previous else now()
            if previous and not event:
                saved = json.loads(previous["record_json"])["dto"]
                metadata = {key: saved[key] for key in metadata}
            revision_files = [{key: item[key] for key in ("path", "size", "sha256")} for item in fingerprints]
            revision = _id("rev", {"id": cid, "files": revision_files, "event": metadata})
            dto = TtsCheckpoint(
                id=cid,
                revision=revision,
                source_job_id=job["id"],
                project_id=job.get("project_id"),
                version_id=job.get("version_id"),
                name=Path(relative).name,
                path=relative,
                relative_path=relative,
                **metadata,
                gpt_sovits=gsv_info,
                size=sum(item["size"] for item in assets),
                discovered_at=discovered,
                modified_at=modified,
                can_preview=False,
                unavailable_reason=None,
                files=[
                    {
                        "id": _id("file", [cid, revision, item["name"]]),
                        "name": item["name"],
                        "role": "config" if item["name"] in {"lora_config.json", "checkpoint.json"} else "gpt_weights" if item["name"] == "gpt.ckpt" else "sovits_weights" if item["name"] == "sovits.pth" else "weights",
                        "size": item["size"],
                        "download_url": f"/api/tts/jobs/{job['id']}/checkpoints/{cid}/files/{_id('file', [cid, revision, item['name']])}",
                    }
                    for item in assets
                ],
            )
            record = _json({"dto": dto.model_dump(), "root": str(root), "fingerprints": fingerprints})
            if previous:
                c.db.update("tts_checkpoints", cid, {"revision": revision, "record_json": record})
            else:
                c.db.insert(
                    "tts_checkpoints",
                    {
                        "id": cid,
                        "source_job_id": job["id"],
                        "relative_path": relative,
                        "revision": revision,
                        "discovered_at": discovered,
                        "record_json": record,
                    },
                )


def _project_checkpoint(c: Any, job: dict, row: dict) -> TtsCheckpoint:
    item = TtsCheckpoint.model_validate(json.loads(row["record_json"])["dto"])
    issue = preview_issue(c, job, item)
    return item.model_copy(update={"can_preview": issue is None, "unavailable_reason": issue})


def _checkpoint_key(item: TtsCheckpoint) -> tuple:
    return item.created_at is not None, item.created_at or 0, item.id


def checkpoints(c: Any, jid: str) -> list[TtsCheckpoint]:
    job = _job(c, jid, "tts_train")
    _register_checkpoints(c, job)
    rows = c.db.fetchall("SELECT * FROM tts_checkpoints WHERE source_job_id=?", (jid,))
    return sorted((_project_checkpoint(c, job, row) for row in rows), key=_checkpoint_key, reverse=True)


def checkpoint(c: Any, jid: str, cid: str) -> TtsCheckpoint:
    for item in checkpoints(c, jid):
        if item.id == cid:
            return item
    _fail("tts.checkpoint_not_found", "检查点不存在。", 404)


def checkpoint_snapshot(c: Any, jid: str, cid: str) -> dict:
    item = checkpoint(c, jid, cid)
    if issue := preview_issue(c, _job(c, jid, "tts_train"), item):
        _fail(issue.code, issue.message)
    row = c.db.fetchone("SELECT record_json FROM tts_checkpoints WHERE id=?", (cid,))
    record = json.loads(row["record_json"])
    return {
        "source_output_dir": record["root"],
        "checkpoint": str(Path(record["root"]) / item.relative_path),
        "checkpoint_id": item.id,
        "checkpoint_revision": item.revision,
        "checkpoint_fingerprints": record["fingerprints"],
    }


def checkpoint_asset(c: Any, jid: str, cid: str, fid: str) -> dict:
    _job(c, jid, "tts_train")
    row = c.db.fetchone("SELECT * FROM tts_checkpoints WHERE id=? AND source_job_id=?", (cid, jid))
    if row is None:
        _fail("tts.checkpoint_not_found", "检查点尚未登记。", 404)
    record = json.loads(row["record_json"])
    dto = TtsCheckpoint.model_validate(record["dto"])
    item = next((item for item in dto.files if item.id == fid), None)
    if item is None:
        _fail("tts.checkpoint_file_not_found", "检查点文件不存在。", 404)
    if issue := _resource_issue(c, record["root"], dto.relative_path, record["fingerprints"]):
        _fail(issue.code, issue.message, 403 if issue.code == "tts.path_denied" else 409)
    return next(dict(asset) for asset in record["fingerprints"] if Path(asset["path"]).name == item.name)


def _cursor(c: Any, cursor: str | None, filters: dict, limit: int) -> tuple[float | None, tuple | None]:
    if type(limit) is not int or not 1 <= limit <= 200:
        _fail("tts.invalid", "每页数量须为 1 至 200。", 422)
    if cursor is None:
        return None, None
    try:
        encoded, signature = cursor.split(".")
        payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        if not hmac.compare_digest(
            hmac.new(_cursor_secret(c), payload, hashlib.sha256).hexdigest(), signature
        ):
            raise ValueError
        value = json.loads(payload)
        if (
            value["filters"] != filters
            or _number(value["as_of"]) is None
            or not 0 <= value["as_of"] <= now()
            or not isinstance(value["last"], list)
        ):
            raise ValueError
        return value["as_of"], tuple(value["last"])
    except (ValueError, KeyError, TypeError):
        _fail("tts.cursor_invalid", "分页位置无效，请重新读取列表。", 422)


def _cursor_secret(c: Any) -> bytes:
    with c.db.lock:
        secret = c.db.get_kv("tts.results.cursor_secret")
        if secret is None:
            secret = secrets.token_hex(32)
            c.db.set_kv("tts.results.cursor_secret", secret)
    return bytes.fromhex(secret)


def _next_cursor(c: Any, filters: dict, as_of: float, last: tuple) -> str:
    payload = _json({"filters": filters, "as_of": as_of, "last": last}).encode()
    return (
        base64.urlsafe_b64encode(payload).decode().rstrip("=")
        + "."
        + hmac.new(_cursor_secret(c), payload, hashlib.sha256).hexdigest()
    )


def version_checkpoints(
    c: Any, pid: str, vid: str, cursor: str | None = None, limit: int = 50
) -> TtsCheckpointPage:
    _scope(c, pid, vid)
    filters = {"kind": "checkpoints", "project_id": pid, "version_id": vid}
    as_of, last = _cursor(c, cursor, filters, limit)
    items = []
    for row in c.db.fetchall(
        "SELECT * FROM jobs WHERE project_id=? AND version_id=? AND type='tts_train'", (pid, vid)
    ):
        if row["id"] not in deleting_jobs:
            items.extend(checkpoints(c, row["id"]))
    as_of = now() if as_of is None else as_of
    items = sorted(
        (
            item
            for item in items
            if item.discovered_at <= as_of and (last is None or _checkpoint_key(item) < last)
        ),
        key=_checkpoint_key,
        reverse=True,
    )
    page = items[:limit]
    return TtsCheckpointPage(
        items=page,
        as_of=as_of,
        next_cursor=_next_cursor(c, filters, as_of, _checkpoint_key(page[-1]))
        if len(items) > limit
        else None,
    )


def _sample_options(job: dict) -> dict:
    sample = _payload(job).get("tts_sample", {})
    return sample if isinstance(sample, dict) else {}


def _source_summary(job: dict) -> dict:
    payload = _payload(job)
    summary = payload.get("source_summary") or _sample_options(job).get("source_summary")
    return summary if isinstance(summary, dict) else {}


def _sample_source_id(job: dict) -> str:
    return (
        _sample_options(job).get("source_job_id")
        or _payload(job).get("source_job_id")
        or _source_summary(job).get("job_id")
        or ""
    )


def _audio_candidate(c: Any, job: dict, value: str) -> Path:
    root_value = job.get("samples_dir") or _payload(job).get("sampling", {}).get("output_dir")
    if not root_value:
        raise ValueError("试听任务未记录音频目录。")
    root = _directory(c, Path(root_value))
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.parent != root or path.suffix.lower() != ".wav" or "\\" in path.name:
        raise SourceFileError("tts.path_denied", "录音不属于这个试听任务。")
    return path


def _audio_records(c: Any, job: dict) -> list[dict]:
    options = _sample_options(job)
    for event in _events(c, job):
        if event.get("type") != "tts.sample.saved" or not isinstance(event.get("path"), str):
            continue
        try:
            path = _audio_candidate(c, job, event["path"])
        except FileNotFoundError:
            root_value = job.get("samples_dir") or _payload(job).get("sampling", {}).get("output_dir")
            path = Path(event["path"])
            if (
                not root_value
                or not path.is_absolute()
                or path.parent != Path(root_value)
                or path.suffix.lower() != ".wav"
            ):
                continue
        except (OSError, ValueError):
            continue
        key = _id("event", event)
        aid = _id("audio", [job["id"], key])
        if c.db.fetchone("SELECT id FROM tts_sample_audio WHERE id=?", (aid,)):
            continue
        identity = None
        try:
            identity = file_identity(path, c.is_allowed)
        except (OSError, ValueError):
            pass
        source_id = _sample_source_id(job) or event.get("source_job_id") or ""
        dto = TtsAudio(
            id=aid,
            sample_job_id=job["id"],
            source_job_id=source_id,
            project_id=job.get("project_id"),
            version_id=job.get("version_id"),
            checkpoint_id=options.get("checkpoint_id"),
            filename=path.name,
            text=event.get("text") if isinstance(event.get("text"), str) else options.get("text"),
            requested_seed=_number(
                event.get("requested_seed", event.get("seed", options.get("seed"))), integer=True
            ),
            seed=_number(event.get("seed"), integer=True) if "requested_seed" in event else None,
            cfg_value=_number(event.get("cfg_value", options.get("cfg_value"))),
            inference_timesteps=_number(
                event.get("inference_timesteps", options.get("inference_timesteps")), integer=True
            ),
            duration_seconds=_number(event.get("duration_seconds")),
            sample_rate=_number(event.get("sample_rate"), integer=True),
            created_at=_number(event.get("ts")),
            size=identity["size"] if identity else None,
            available=False,
            url=None,
            gpt_sovits=event.get("gpt_sovits"),
        )
        record = _json({"dto": dto.model_dump(), "path": str(path), "fingerprint": identity})
        with c.db.lock:
            c.db.execute(
                "INSERT OR IGNORE INTO tts_sample_audio(id,sample_job_id,event_key,record_json) VALUES (?,?,?,?)",
                (aid, job["id"], key, record),
            )
    return [
        json.loads(row["record_json"])
        for row in c.db.fetchall("SELECT * FROM tts_sample_audio WHERE sample_job_id=?", (job["id"],))
    ]


def _audio_identity(c: Any, job: dict, record: dict) -> dict:
    path = _audio_candidate(c, job, record["path"])
    actual = file_identity(path, c.is_allowed)
    if record["fingerprint"] is None or actual != record["fingerprint"]:
        _fail("tts.source_stale", "试听录音内容已变化。")
    return actual


def _audio(c: Any, job: dict, record: dict) -> TtsAudio:
    dto = TtsAudio.model_validate(record["dto"])
    available = False
    try:
        _audio_identity(c, job, record)
        available = True
    except (OSError, ValueError, ApiError):
        pass
    return dto.model_copy(
        update={
            "available": available,
            "url": f"/api/tts/jobs/{job['id']}/audio/{quote(dto.filename, safe='')}" if available else None,
        }
    )


def _audio_key(record: dict) -> tuple:
    return (
        _number(record["dto"].get("created_at")) is not None,
        record["dto"].get("created_at") or 0,
        record["dto"]["id"],
    )


def sample_job(c: Any, sid: str) -> TtsSampleJob:
    job = _job(c, sid, "tts_sample")
    options = _sample_options(job)
    source_id = _sample_source_id(job)
    source = c.db.fetchone("SELECT * FROM jobs WHERE id=? AND type='tts_train'", (source_id,))
    summary = _source_summary(job) or c.db.get_kv("tts.source_summary:" + source_id, {})
    records = _audio_records(c, job)
    records.sort(key=_audio_key, reverse=True)
    from .routes_work import _job_row

    return TtsSampleJob(
        job=_job_row(job, c),
        source=TtsSampleSource(
            job_id=source_id,
            name=source["name"] if source else summary.get("name", options.get("source_name")),
            record_exists=source is not None,
            checkpoint_available=frozen_checkpoint(c, options),
        ),
        checkpoint_id=options.get("checkpoint_id"),
        checkpoint_revision=options.get("checkpoint_revision"),
        request=TtsSampleRequestSnapshot(
            **{field: options.get(field) for field in TtsSampleRequestSnapshot.model_fields}
        ),
        audio=_audio(c, job, records[0]) if records else None,
    )


def sample_jobs(
    c: Any,
    source_jid: str | None = None,
    pid: str | None = None,
    vid: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
    include_archived: bool = False,
) -> TtsSampleJobPage:
    if source_jid is None:
        if pid is None or vid is None:
            _fail("tts.invalid", "请指定来源任务或项目版本。", 422)
        _scope(c, pid, vid)
    else:
        source = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (source_jid,))
        if source is not None:
            _job(c, source_jid, "tts_train")
    filters = {
        "kind": "samples",
        "source_job_id": source_jid,
        "project_id": pid,
        "version_id": vid,
        "include_archived": include_archived,
    }
    as_of, last = _cursor(c, cursor, filters, limit)
    as_of = now() if as_of is None else as_of
    conditions, params = ["type='tts_sample'", "created_at<=?"], [as_of]
    if source_jid is not None:
        conditions.append(
            "coalesce(json_extract(config_json,'$.tts_sample.source_job_id'),"
            "json_extract(config_json,'$.source_job_id'),"
            "json_extract(config_json,'$.source_summary.job_id'),"
            "json_extract(config_json,'$.tts_sample.source_summary.job_id'))=?"
        )
        params.append(source_jid)
    if pid is not None:
        conditions.append("project_id=? AND version_id=?")
        params.extend((pid, vid))
    if not include_archived:
        conditions.append("archived_at IS NULL")
    if last is not None:
        if len(last) != 2 or _number(last[0]) is None or not isinstance(last[1], str):
            _fail("tts.cursor_invalid", "分页位置无效，请重新读取列表。", 422)
        conditions.append("(created_at<? OR (created_at=? AND id<?))")
        params.extend((last[0], last[0], last[1]))
    rows = c.db.fetchall(
        "SELECT * FROM jobs WHERE " + " AND ".join(conditions) + " ORDER BY created_at DESC,id DESC",
        tuple(params),
    )
    rows = [row for row in rows if row["id"] not in deleting_jobs]
    page = rows[:limit]
    return TtsSampleJobPage(
        items=[sample_job(c, row["id"]) for row in page],
        as_of=as_of,
        next_cursor=_next_cursor(c, filters, as_of, (page[-1]["created_at"], page[-1]["id"]))
        if len(rows) > limit
        else None,
    )


def audio_results(c: Any, jid: str) -> list[TtsAudio]:
    row = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    if row is None or row["type"] not in {"tts_train", "tts_sample"}:
        _fail("tts.not_found", "语音任务不存在。", 404)
    job = _job(c, jid, row["type"])
    return [_audio(c, job, record) for record in sorted(_audio_records(c, job), key=_audio_key, reverse=True)]


def sample_audio_asset(c: Any, sid: str, filename: str) -> dict:
    job = _job(c, sid, "tts_sample")
    if (
        not filename
        or filename != Path(filename).name
        or "\\" in filename
        or Path(filename).suffix.lower() != ".wav"
    ):
        _fail("tts.audio_not_found", "试听录音不存在。", 404)
    records = [record for record in _audio_records(c, job) if record["dto"]["filename"] == filename]
    if not records:
        _fail("tts.audio_not_found", "试听录音不存在。", 404)
    try:
        return _audio_identity(c, job, max(records, key=_audio_key))
    except FileNotFoundError:
        _fail("tts.audio_not_found", "试听录音文件不存在。", 404)
    except SourceFileError as exc:
        _fail(exc.code, str(exc), 403 if exc.code == "tts.path_denied" else 409)
    except (OSError, ValueError):
        _fail("tts.path_denied", "无法安全读取试听录音。", 403)
