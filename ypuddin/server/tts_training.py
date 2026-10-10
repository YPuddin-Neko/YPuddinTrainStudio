"""Saved-version training admission with durable request replay and immutable manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from . import tts_requests, tts_validation
from .db import new_id, now
from .errors import ApiError
from .gpu_selection import GpuSelection
from .routes_tts import _devices, _verify_environment
from .routes_work import _job_row
from .tts_references import reject_deleting_references
from .tts_source_copy import DirectoryGuard


class TtsTrainingBody(GpuSelection):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=1)
    data_revision: int = Field(ge=1)
    name: str = Field("语音训练", min_length=1, max_length=200, pattern=r".*\S.*")


def _directory(path: Path) -> None:
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or (getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        ):
            raise ApiError("语音任务目录不能使用链接或非目录文件。", code="tts.path_denied", status=403)


def _snapshot(
    c: Any, pid: str, vid: str, jid: str, captured, report, inputs: dict
) -> tuple[Path, dict, tuple[int, int], DirectoryGuard]:
    from ypuddin.tts.source_scan import file_identity

    run = c.job_records_dir(pid, vid, jid)
    output = c.job_output_dir(pid, vid, jid)
    samples = c.job_storage_dir(pid, vid, jid, "samples_dir", run)
    logs = c.job_storage_dir(pid, vid, jid, "logs_dir", run)
    state = c.job_storage_dir(pid, vid, jid, "state_dir", run)
    for path in (run, output, samples, logs, state):
        _directory(path)
        if not c.is_allowed(path):
            raise ApiError("语音任务目录不在允许访问的范围内。", code="tts.path_denied", status=403)
    pending = []
    parent = run.parent
    while not parent.exists():
        pending.append(parent.name)
        parent = parent.parent
    guard = DirectoryGuard.capture(parent, c.is_allowed)
    for name in reversed(pending):
        try:
            guard = guard.child(name, c.is_allowed)
        except FileExistsError:
            guard.check(c.is_allowed)
            guard = DirectoryGuard.capture(guard.path / name, c.is_allowed)
    guard = guard.child(run.name, c.is_allowed)
    identity = guard.chain[-1][1]
    try:
        execution = report._runtime_config or captured.config
        recipe = execution.model_dump(mode="json")
        source_paths = {
            str(source["source"].path if hasattr(source["source"], "path") else source["source"]["path"])
            for source in (inputs["train"], inputs["validation"])
            if source is not None
        }
        files = {
            item["path"]: dict(item)
            for item in report._runtime_fingerprints
            if item["path"] not in source_paths
        }
        source_snapshots = {}
        for split, field in (("train", "train_manifest"), ("validation", "val_manifest")):
            source = inputs[split]
            recipe[field] = ""
            if source is None:
                continue
            model = source["source"]
            model = model.model_dump(mode="json") if hasattr(model, "model_dump") else model
            target = run / f"{field}.jsonl"
            digest, size = hashlib.sha256(), 0
            with guard.new_file(target.name, c.is_allowed) as stream:
                for row in source["rows"]:
                    content = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                    stream.write(content)
                    digest.update(content)
                    size += len(content)
            recipe[field] = str(target)
            actual = file_identity(target, c.is_allowed)
            if (actual["size"], actual["sha256"]) != (size, digest.hexdigest()):
                raise ApiError("数据清单在保存时发生变化，请重新检查。", code="tts.source_stale", status=409)
            files[str(target)] = actual
            for item in source["fingerprints"]:
                if item["path"] != model["path"]:
                    files[item["path"]] = {key: item[key] for key in ("path", "size", "sha256")}
            source_snapshots[split] = {
                "source_id": model["id"],
                "source_revision": model["revision"],
                "snapshot_id": model["snapshot_id"],
                "fingerprint": source["fingerprint"],
            }
        payload = {
            "scope": {"project_id": pid, "version_id": vid},
            "revision": captured.revision,
            "data_revision": captured.data_revision,
            "input_fingerprint": report.input_fingerprint,
            "validation_id": report.validation_id,
            "tts": recipe,
            **({"tts_environment": dict(report._runtime_environment)}
               if report._runtime_environment is not None else {}),
            "tts_inputs": {
                "fingerprints": sorted(files.values(), key=lambda item: item["path"]),
                **(
                    {"model_identity": report._runtime_model_identity}
                    if report._runtime_model_identity is not None
                    else {}
                ),
                "sources": source_snapshots,
            },
            "checkpoint": {"output_dir": str(output), "state_dir": str(state), "resume": None},
            "sampling": {"output_dir": str(samples)},
            "logging": {"events_path": str(logs / "events.jsonl"), "output_dir": str(logs)},
        }
        return run, payload, identity, guard
    except BaseException:
        _remove_owned(run, identity)
        raise


def _remove_owned(path: Path | None, identity: tuple[int, int] | None) -> None:
    if path is None or identity is None:
        return
    try:
        info = path.lstat()
    except OSError:
        return
    if stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == identity:
        shutil.rmtree(path, ignore_errors=True)


def _sources_digest(inputs: dict) -> str:
    def source_identity(value):
        if value is None:
            return None
        source = value["source"]
        source = source.model_dump(mode="json") if hasattr(source, "model_dump") else source
        return {
            "id": source["id"],
            "revision": source["revision"],
            "snapshot_id": source["snapshot_id"],
            "state": source["state"],
            "fingerprint": value["fingerprint"],
            "fingerprints": value["fingerprints"],
        }

    payload = {
        "data_revision": inputs["data_revision"],
        **{split: source_identity(inputs[split]) for split in ("train", "validation")},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def create_training(c: Any, pid: str, vid: str, body: TtsTrainingBody, key: str) -> tuple[dict, int]:
    request_id, replay = tts_requests.reserve(c, pid, vid, "train", key, body.model_dump(mode="json"))
    if replay is not None:
        return _job_row(replay, c), 200
    run = identity = None
    committed = False
    try:
        captured = tts_validation.capture_validation_state(c, pid, vid, body.revision, body.data_revision)
        initial = c.tts_sources.snapshot_identity(pid, vid)
        devices = _devices(body.gpu_devices)
        report = tts_validation.validate_version(c, pid, vid, body.revision, body.data_revision, gpu_devices=devices)
        if not report.valid:
            raise ApiError(
                "请先解决语音训练配置中的问题。",
                code="tts.invalid",
                status=422,
                details={
                    "issues": [issue.model_dump() for issue in report.errors],
                    "validation_id": report.validation_id,
                },
            )
        inputs = c.tts_sources.validation_inputs(pid, vid, recheck=True)
        with c.db.lock:
            tts_validation.assert_validation_current(c, pid, vid, captured)
            if c.tts_sources.snapshot_identity(pid, vid) != initial:
                raise ApiError(
                    "数据检查结果已变化，请重新检查。",
                    code="tts.data_conflict",
                    status=409,
                    details={"current_data_revision": c.resolve_version(pid, vid)["data_revision"]},
                )
        expected_sources = _sources_digest(inputs)
        jid = new_id("j")
        run, payload, identity, guard = _snapshot(c, pid, vid, jid, captured, report, inputs)
        # Recheck actual bytes after manifest persistence; metadata admission stays short and locked.
        refreshed = c.tts_sources.validation_inputs(pid, vid, recheck=True)
        from ypuddin.tts.runtime import verify_input_snapshot

        try:
            verify_input_snapshot(payload["tts_inputs"], model_path=payload["tts"].get("model_path") or None)
        except (OSError, ValueError) as exc:
            raise ApiError("训练输入已变化，请重新检查。", code="tts.source_stale", status=409) from exc
        _verify_environment(c, payload)
        with c.db.transaction():
            guard.check(c.is_allowed)
            tts_validation.assert_validation_current(c, pid, vid, captured)
            if (
                c.tts_sources.snapshot_identity(pid, vid) != initial
                or _sources_digest(refreshed) != expected_sources
            ):
                raise ApiError(
                    "训练输入已变化，请重新检查。",
                    code="tts.data_conflict",
                    status=409,
                    details={"current_data_revision": c.resolve_version(pid, vid)["data_revision"]},
                )
            reject_deleting_references(c, [Path(item["path"]) for item in payload["tts_inputs"]["fingerprints"]])
            c.db.insert(
                "jobs",
                {
                    "id": jid,
                    "project_id": pid,
                    "version_id": vid,
                    "type": "tts_train",
                    "name": body.name,
                    "status": "queued",
                    "priority": 0,
                    "created_at": now(),
                    "run_dir": str(run),
                    "samples_dir": payload["sampling"]["output_dir"],
                    "gpu_devices_json": json.dumps(devices),
                    "config_json": json.dumps(payload, ensure_ascii=False),
                    "progress_json": "{}",
                    "latest_json": "{}",
                },
            )
            tts_requests.complete(c, request_id, jid)
        committed = True
        c.bus.publish("queue.changed", {})
        c.bus.publish("job.state", {"job_id": jid, "status": "queued"})
        return _job_row(c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,)), c), 201
    except OSError as exc:
        raise ApiError("无法保存语音任务文件。", code="tts.snapshot_io", status=500) from exc
    except ValueError as exc:
        raise ApiError("无法保存或校验语音训练输入。", code="tts.source_stale", status=409) from exc
    finally:
        if not committed:
            _remove_owned(run, identity)
            tts_requests.release(c, request_id)
