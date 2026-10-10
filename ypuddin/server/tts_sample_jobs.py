"""Create listening jobs and retries from frozen speech inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field, model_serializer, model_validator

from ypuddin.tts.execution_config import parse_execution_config
from ypuddin.tts.issues import TtsIssue
from ypuddin.tts.runtime import _model_identity, verify_input_snapshot
from ypuddin.tts.sample_config import GptSovitsSampleOptions
from ypuddin.tts.source_scan import SourceFileError, file_identity, inspect_audio, open_source

from . import tts_requests, tts_results
from .db import new_id, now
from .errors import ApiError
from .gpu_selection import GpuSelection
from .job_paths import deleting_jobs
from .tts_job_actions import assert_action
from .tts_references import reject_deleting_references
from .tts_source_copy import DirectoryGuard
from .tts_training import _directory, _remove_owned


class TtsSampleBody(GpuSelection):
    model_config = ConfigDict(extra="forbid", strict=True)
    checkpoint_id: str | None = Field(None, min_length=1, max_length=100)
    checkpoint_revision: str | None = Field(None, min_length=1, max_length=100)
    checkpoint: str | None = Field(None, min_length=1, max_length=2000)
    text: str = Field(min_length=1, max_length=4000, pattern=r".*\S.*")
    reference_audio: str = ""
    reference_text: str = ""
    seed: int = Field(42, ge=0, lt=2**32)
    cfg_value: float = Field(2.0, ge=0, le=20, allow_inf_nan=False)
    inference_timesteps: int = Field(10, ge=1, le=100)
    gpt_sovits: GptSovitsSampleOptions | None = None

    @model_serializer(mode="wrap")
    def engine_options(self, handler):
        value = handler(self)
        if value.get("gpt_sovits") is None:
            value.pop("gpt_sovits", None)
        return value

    @model_validator(mode="after")
    def identity(self):
        if self.checkpoint is not None:
            if self.checkpoint_id is not None or self.checkpoint_revision is not None:
                raise ValueError("检查点路径与检查点身份不能同时填写。")
            path = Path(self.checkpoint)
            if path.is_absolute() or ".." in path.parts or "\\" in self.checkpoint:
                raise ValueError("请选择此训练任务中的检查点相对路径。")
        elif self.checkpoint_id is None or self.checkpoint_revision is None:
            raise ValueError("请同时提供检查点 ID 和修订。")
        if bool(self.reference_audio.strip()) != bool(self.reference_text.strip()):
            raise ValueError("参考音频和对应文本需要一起填写。")
        return self


def _fail(code: str, message: str, *, status=409, loc=()) -> None:
    issue = TtsIssue(code=code, loc=list(loc), message=message)
    raise ApiError(message, code=code, status=status, details={"issues": [issue.model_dump()]})


def _raise(issue: TtsIssue | None) -> None:
    if issue:
        raise ApiError(issue.message, code=issue.code, status=409, details={"issues": [issue.model_dump()]})


def _row(c: Any, jid: str, kind: str | None = None) -> dict:
    row = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    if row is None:
        _fail("tts.source_not_found", "语音任务不存在。", status=404)
    if row["type"] not in ({kind} if kind else {"tts_train", "tts_sample"}):
        _fail("tts.job_type_mismatch", "任务类型与请求不匹配。")
    return row


def _reserve(c: Any, jid: str, action: str, key: str | None, payload: dict):
    source = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    pid, vid = (source.get("project_id"), source.get("version_id")) if source else (None, None)
    if source is None:
        try:
            normalized = str(UUID(key))
        except (TypeError, AttributeError, ValueError):
            normalized = ""
        saved = c.db.fetchone("SELECT scope FROM tts_requests WHERE action=? AND request_key=? AND target_job_id=?", (action, normalized, jid))
        if saved:
            pid, vid = json.loads(saved["scope"])
    return tts_requests.reserve(c, pid, vid, action, key, {"target_job_id": jid, **payload}, target_job_id=jid)


def _check_environment(c: Any, recipe: dict, *, sample: bool, gpu_devices: list[str] | None = None,
                       environment: dict | None = None):
    from . import routes_tts
    from .tts_gpu import invalidate

    invalidate()
    config = parse_execution_config(recipe)
    if not config.model_path.strip():
        if config.engine == "gpt-sovits-v5":
            from ypuddin.tts.gpt_sovits.core import MODEL_REQUIRED_MESSAGE
        else:
            from ypuddin.tts.core import MODEL_REQUIRED_MESSAGE

        _fail("tts.model_required", MODEL_REQUIRED_MESSAGE, status=422, loc=["model_path"])
    allowed = c.is_allowed
    if environment is not None:
        routes_tts._verify_environment(c, {"tts": recipe, "tts_environment": environment})
        manager = getattr(c, "tts_environments", None)
        if manager is not None:
            allowed = manager.runtime_allowed(environment)
    report = routes_tts.preflight(config, allowed=allowed, mode="sample" if sample else "train", gpu_devices=gpu_devices)
    if not report["ok"]:
        from ypuddin.tts.devices import selection_issues

        selected = report.get("details", {}).get("runtime", {}).get("devices")
        issues = selection_issues(selected) if selected is not None else []
        device_messages = {item["message"] for item in issues}
        issues.extend(TtsIssue(code="tts.environment.invalid", loc=["environment"], message=str(value)).model_dump()
                      for value in report["errors"] if str(value) not in device_messages)
        raise ApiError("请先解决语音运行环境中的问题。", code="tts.invalid", status=422, details={"issues": issues})
    return config


def _preflight_inputs(c: Any, recipe: dict, *, sample: dict | None, previous: dict | None = None,
                      gpu_devices: list[str] | None = None, environment: dict | None = None) -> dict:
    config = _check_environment(c, recipe, sample=sample is not None, gpu_devices=gpu_devices, environment=environment)
    if previous is not None:
        model = previous.get("model_identity")
        if model:
            assets = set(model["assets"].values())
            frozen_model = {"model_identity": model, "fingerprints": [item for item in previous["fingerprints"] if item["path"] in assets]}
            verify_input_snapshot(frozen_model, model_path=config.model_path)
    if config.engine == "gpt-sovits-v5":
        from ypuddin.tts.gpt_sovits.core import model_identity, require_text_assets

        if sample:
            options = sample.get("gpt_sovits", {})
            try:
                require_text_assets(config, [options.get("text_language", "zh"), options.get("reference_language", "zh")], inference=True)
            except (OSError, ValueError) as exc:
                _fail("tts.sample_language_assets", str(exc), status=422, loc=["gpt_sovits"])
        identity = model_identity(config)
    else:
        identity = _model_identity(config.model_path)
    files = {path: file_identity(Path(path), c.is_allowed) for path in identity["assets"].values()}
    if sample:
        _raise(tts_results.frozen_checkpoint_issue(c, sample))
        for item in sample["checkpoint_fingerprints"]:
            files[item["path"]] = {key: item[key] for key in ("path", "size", "sha256")}
        if sample.get("reference_audio"):
            path = Path(sample["reference_audio"])
            metadata, audio_identity, issues = inspect_audio(path, c.is_allowed, engine=config.engine)
            if error := next((item for item in issues if not item[2]), None):
                issue = TtsIssue(code=error[0], loc=["reference_audio"], message=error[1])
                raise ApiError(error[1], code="tts.reference_invalid", status=422, details={"issues": [issue.model_dump()]})
            if config.engine == "gpt-sovits-v5" and not 3 <= metadata["duration_seconds"] <= 10:
                _fail("tts.reference_duration", "GPT-SoVITS 参考录音须为 3 至 10 秒。", status=422, loc=["reference_audio"])
            files[str(path)] = audio_identity
    else:
        # Historical recipes have no byte identities; capture their current valid manifests once.
        from ypuddin.tts.core import validate_manifest

        for field in ("train_manifest", "val_manifest"):
            if not recipe.get(field):
                continue
            path = Path(recipe[field])
            if config.engine == "gpt-sovits-v5":
                from ypuddin.tts.source_scan import scan_manifest

                scanned = scan_manifest(path, "retry", field, allowed=c.is_allowed, engine=config.engine)
                if scanned.summary.invalid_count or not scanned.summary.valid_clips_count:
                    raise ValueError("GPT-SoVITS 数据清单没有通过检查。")
            else:
                validate_manifest(path, allowed=c.is_allowed)
            files[str(path)] = file_identity(path, c.is_allowed)
            with open_source(path, c.is_allowed) as stream:
                lines = stream.read().decode("utf-8-sig").split("\n")
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("audio", "ref_audio"):
                    if row.get(key):
                        audio = Path(row[key]).expanduser()
                        audio = audio if audio.is_absolute() else path.parent / audio
                        files[str(audio)] = file_identity(audio, c.is_allowed)
    return {"model_identity": identity, "fingerprints": sorted(files.values(), key=lambda item: item["path"])}


def _new_snapshot(c: Any, job: dict, jid: str, payload: dict, *, sample: bool):
    pid, vid = job.get("project_id"), job.get("version_id")
    run = c.job_records_dir(pid, vid, jid) if pid else c.data_root / "tts" / "jobs" / jid
    output = c.job_output_dir(pid, vid, jid)
    paths = {name: c.job_storage_dir(pid, vid, jid, name + "_dir", run) for name in ("samples", "logs", "state")}
    for path in (run, output, *paths.values()):
        _directory(path)
        if not c.is_allowed(path):
            _fail("tts.path_denied", "语音任务目录不在允许访问的范围内。", status=403)
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
    own = guard.chain[-1][1]
    try:
        result = copy.deepcopy(payload)
        result.pop("device", None)
        result.pop("control_dir", None)
        result["checkpoint"] = {"output_dir": str(output), "state_dir": str(paths["state"]), "resume": None}
        result["logging"] = {"events_path": str(paths["logs"] / "events.jsonl"), "output_dir": str(paths["logs"])}
        result["sampling"] = {"output_dir": str(paths["samples"])}
        result["scope"] = {"project_id": pid, "version_id": vid}
        if not sample:
            files = {item["path"]: item for item in result["tts_inputs"]["fingerprints"]}
            for field in ("train_manifest", "val_manifest"):
                original = result["tts"].get(field)
                if not original:
                    continue
                target = run / (field + ".jsonl")
                with open_source(Path(original), c.is_allowed) as src, guard.new_file(target.name, c.is_allowed) as dst:
                    dst.write(src.read())
                actual = file_identity(target, c.is_allowed)
                expected = files.pop(original)
                if (actual["size"], actual["sha256"]) != (expected["size"], expected["sha256"]):
                    _fail("tts.source_stale", "冻结数据清单在复制时发生变化。")
                files[str(target)] = actual
                result["tts"][field] = str(target)
            result["tts_inputs"]["fingerprints"] = sorted(files.values(), key=lambda item: item["path"])
        return run, result, own, guard
    except BaseException:
        _remove_owned(run, own)
        raise


def _verify(c: Any, payload: dict, *, sample: bool) -> None:
    from .routes_tts import _verify_environment

    _verify_environment(c, payload)
    inputs = payload["tts_inputs"]
    for item in inputs["fingerprints"]:
        if file_identity(Path(item["path"]), c.is_allowed) != {key: item[key] for key in ("path", "size", "sha256")}:
            _fail("tts.source_stale", "语音任务输入内容已变化。")
    manifests = () if sample else tuple(payload["tts"][key] for key in ("train_manifest", "val_manifest") if payload["tts"].get(key))
    verify_input_snapshot(inputs, manifests=manifests, model_path=payload["tts"].get("model_path") or None)
    if sample:
        _raise(tts_results.frozen_checkpoint_issue(c, payload["tts_sample"]))


def _insert(c: Any, rid: str, jid: str, owner: dict, payload: dict, run: Path, name: str, kind: str, devices: list[str]):
    reject_deleting_references(c, [Path(item["path"]) for item in payload["tts_inputs"]["fingerprints"]])
    c.db.insert("jobs", {
        "id": jid, "type": kind, "name": name, "project_id": owner.get("project_id"), "version_id": owner.get("version_id"),
        "status": "queued", "priority": owner.get("priority", 0), "created_at": now(), "run_dir": str(run),
        "samples_dir": payload["sampling"]["output_dir"], "gpu_devices_json": json.dumps(devices),
        "config_json": json.dumps(payload, ensure_ascii=False), "progress_json": "{}", "latest_json": "{}",
    })
    tts_requests.complete(c, rid, jid)


def _result(c: Any, jid: str, status: int):
    from .routes_work import _job_row

    row = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    if status == 201:
        payload = json.loads(row["config_json"])
        event = {"job_id": jid, "project_id": row.get("project_id"), "version_id": row.get("version_id"), "source_job_id": (payload.get("tts_sample") or {}).get("source_job_id")}
        c.bus.publish("queue.changed", event)
        c.bus.publish("job.state", {**event, "status": "queued"})
    return _job_row(row, c), status


def create_sample(c: Any, jid: str, body: TtsSampleBody, key: str) -> tuple[dict, int]:
    from .routes_tts import _devices

    rid, replay = _reserve(c, jid, "sample", key, body.model_dump(mode="json"))
    if replay is not None:
        return _result(c, replay["id"], 200)
    run = own = None
    committed = False
    try:
        source = _row(c, jid, "tts_train")
        if body.checkpoint is not None:
            item = next((item for item in tts_results.checkpoints(c, jid) if item.relative_path == body.checkpoint), None)
            if item is None:
                _fail("tts.checkpoint_not_found", "检查点不存在。", status=404)
        else:
            item = tts_results.checkpoint(c, jid, body.checkpoint_id)
            if item.revision != body.checkpoint_revision:
                _fail("tts.checkpoint_stale", "检查点已变化，请重新选择。", loc=["checkpoint_revision"])
        _raise(tts_results.preview_issue(c, source, item))
        sample = tts_results.checkpoint_snapshot(c, jid, item.id)
        if sample["checkpoint_revision"] != item.revision:
            _fail("tts.checkpoint_stale", "检查点在检查时发生变化。")
        sample.update(body.model_dump(exclude={"gpu_devices", "checkpoint", "checkpoint_id", "checkpoint_revision"}))
        sample["reference_audio"] = str(Path(body.reference_audio).expanduser().absolute()) if body.reference_audio else ""
        sample["source_job_id"] = jid
        original = json.loads(source["config_json"])
        recipe = original["tts"]
        if recipe.get("engine") == "gpt-sovits-v5":
            if not body.reference_audio.strip() or not body.reference_text.strip():
                _fail("tts.reference_required", "GPT-SoVITS 试听需要参考录音及对应文本。", status=422, loc=["reference_audio"])
            if {"cfg_value", "inference_timesteps"} & body.model_fields_set:
                _fail("tts.sample_options", "请使用 GPT-SoVITS 的 CFG 和采样步数。", status=422, loc=["gpt_sovits"])
            options = (body.gpt_sovits or GptSovitsSampleOptions()).resolved(recipe["variant"])
            sample["gpt_sovits"] = options.model_dump()
            sample["cfg_value"] = options.cfg_scale
            sample["inference_timesteps"] = options.sample_steps
        elif body.gpt_sovits is not None:
            _fail("tts.sample_options", "此训练任务不使用 GPT-SoVITS 试听参数。", status=422, loc=["gpt_sovits"])
        payload = {"tts": copy.deepcopy(original["tts"]), "tts_sample": sample, "source_job_id": jid,
                   "source_summary": {"job_id": jid, "name": source["name"]}}
        if "tts_environment" in original:
            payload["tts_environment"] = copy.deepcopy(original["tts_environment"])
        devices = _devices(body.gpu_devices)
        payload["tts_inputs"] = _preflight_inputs(c, payload["tts"], sample=sample, previous=original.get("tts_inputs"),
                                                  gpu_devices=devices, environment=payload.get("tts_environment"))
        new = new_id("j")
        run, payload, own, guard = _new_snapshot(c, source, new, payload, sample=True)
        _verify(c, payload, sample=True)
        with c.db.transaction():
            guard.check(c.is_allowed)
            current = _row(c, jid, "tts_train")
            if current["config_json"] != source["config_json"]:
                _fail("tts.source_stale", "来源任务快照已变化。")
            _raise(tts_results.preview_issue(c, current, item, verify=False))
            _insert(c, rid, new, current, payload, run, f"{source['name']} · 试听", "tts_sample", devices)
        committed = True
        return _result(c, new, 201)
    except SourceFileError as exc:
        _fail(exc.code, str(exc), status=403 if exc.code == "tts.path_denied" else 409)
    except (OSError, ValueError) as exc:
        _fail("tts.source_unavailable", f"无法读取或保存试听输入：{exc}")
    finally:
        if not committed:
            _remove_owned(run, own)
            tts_requests.release(c, rid)


def retry_job(c: Any, jid: str, key: str | None) -> tuple[dict, int]:
    from .routes_tts import _devices

    rid, replay = _reserve(c, jid, "retry", key, {})
    if replay is not None:
        return _result(c, replay["id"], 200)
    run = own = None
    committed = False
    try:
        with c.db.lock:
            job = _row(c, jid)
            assert_action(c, job, "retry")
        sample = job["type"] == "tts_sample"
        original = json.loads(job["config_json"])
        payload = copy.deepcopy(original)
        payload["retry_of_job_id"] = jid
        devices = _devices(json.loads(job.get("gpu_devices_json") or "[]"))
        if "tts_inputs" in payload:
            _verify(c, payload, sample=sample)
            _check_environment(c, payload["tts"], sample=sample, gpu_devices=devices, environment=payload.get("tts_environment"))
        else:
            if sample:
                options = payload["tts_sample"]
                _raise(tts_results.frozen_checkpoint_issue(c, options))
                # Legacy snapshots recorded paths, so capture their readable resources for this new retry.
                root = Path(options["source_output_dir"])
                checkpoint = Path(options["checkpoint"])
                checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
                options["checkpoint_fingerprints"] = [file_identity(checkpoint / name, c.is_allowed) for name in tts_results.CHECKPOINT_FILES if (checkpoint / name).is_file()]
            payload["tts_inputs"] = _preflight_inputs(c, payload["tts"], sample=payload.get("tts_sample") if sample else None,
                                                      gpu_devices=devices, environment=payload.get("tts_environment"))
        new = new_id("j")
        run, payload, own, guard = _new_snapshot(c, job, new, payload, sample=sample)
        _verify(c, payload, sample=sample)
        with c.db.transaction():
            guard.check(c.is_allowed)
            current = _row(c, jid)
            assert_action(c, current, "retry")
            if current["config_json"] != job["config_json"] or current.get("gpu_devices_json") != job.get("gpu_devices_json"):
                _fail("tts.source_stale", "任务快照已变化。")
            if jid in deleting_jobs:
                _fail("job.deleting", "这个任务正在删除。")
            _insert(c, rid, new, current, payload, run, job["name"], job["type"], devices)
        committed = True
        return _result(c, new, 201)
    except SourceFileError as exc:
        _fail(exc.code, str(exc), status=403 if exc.code == "tts.path_denied" else 409)
    except (OSError, ValueError) as exc:
        _fail("tts.source_unavailable", f"无法读取或保存重试输入：{exc}")
    finally:
        if not committed:
            _remove_owned(run, own)
            tts_requests.release(c, rid)
