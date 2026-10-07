"""Speech configuration, training and listening jobs in the shared queue."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, ValidationError

from ypuddin.tts.config import TtsConfig
from ypuddin.tts.core import preflight, safe_checkpoint
from ypuddin.tts.execution_config import parse_execution_config

from . import models as m
from .context import ServiceContext
from .db import now
from .errors import ApiError, NotFound
from .gpu_selection import GpuSelection, selection_error
from .hardware import gpu_info
from .job_paths import deleting_jobs
from .routes_work import _job_row
from .supervisor import tts_device_error

router = APIRouter()
TTS_JOBS = frozenset({"tts_train", "tts_sample"})
PATH_FIELDS = ("python_path", "trainer_path", "model_path", "train_manifest", "val_manifest")
CONFIG_KEY = "tts.config"


def ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


class ValidationBody(BaseModel):
    config: dict[str, Any]


class TrainingBody(GpuSelection):
    config: dict[str, Any]
    name: str = Field("语音训练", min_length=1, max_length=200)


class SampleBody(GpuSelection):
    checkpoint: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1, max_length=4000, pattern=r".*\S.*")
    reference_audio: str = ""
    reference_text: str = ""
    seed: int = Field(42, ge=0, lt=2**32)
    cfg_value: float = Field(2.0, ge=0, le=20)
    inference_timesteps: int = Field(10, ge=1, le=100)


def _issues(error: ValidationError) -> list[dict[str, str]]:
    return [{"loc": ".".join(map(str, item["loc"])), "msg": item["msg"]} for item in error.errors()]


def _normalize(config: dict[str, Any]) -> TtsConfig:
    typed = TtsConfig.model_validate(config)
    values = typed.model_dump()
    for key in PATH_FIELDS:
        if values.get(key):
            values[key] = str(Path(values[key]).expanduser().absolute())
    return TtsConfig.model_validate(values)


def _messages(values: list[Any]) -> list[dict[str, str]]:
    return [
        {"loc": str(value.get("loc", "")), "msg": str(value.get("msg", value.get("message", "")))}
        if isinstance(value, dict) else {"loc": "", "msg": str(value)}
        for value in values
    ]


def _inspect(config: dict[str, Any], c: ServiceContext, *, mode: str = "train") -> tuple[TtsConfig | None, dict]:
    result: dict[str, Any] = {"valid": False, "errors": [], "warnings": [], "dataset": None, "environment": None}
    try:
        typed = _normalize(config)
    except ValidationError as error:
        result["errors"] = _issues(error)
        return None, result
    except (OSError, ValueError) as error:
        result["errors"] = [{"loc": "", "msg": str(error)}]
        return None, result
    fields = PATH_FIELDS if mode == "train" else PATH_FIELDS[:3]
    for key in fields:
        value = getattr(typed, key, None)
        if value and not c.is_allowed(Path(value)):
            result["errors"].append({"loc": key, "msg": "路径不在允许访问的目录中。"})
    if result["errors"]:
        return typed, result
    checked = preflight(typed, mode=mode, allowed=c.is_allowed)
    result["errors"] = _messages(checked.get("errors", []))
    result["warnings"] = _messages(checked.get("warnings", []))
    details = checked.get("details", {})
    result["dataset"] = details.get("dataset")
    result["environment"] = {key: value for key, value in details.items() if key != "dataset"}
    if error := tts_device_error(gpu_info()):
        result["errors"].append({"loc": "device", "msg": error})
    result["valid"] = bool(checked.get("ok")) and not result["errors"]
    return typed, result


def _require_valid(config: dict[str, Any], c: ServiceContext, *, mode: str = "train") -> TtsConfig:
    typed, checked = _inspect(config, c, mode=mode)
    if not checked["valid"]:
        raise ApiError("请先解决语音训练配置中的问题。", code="tts.invalid", status=422, details=checked)
    assert typed is not None
    return typed


@router.get("/tts/config", response_model=TtsConfig)
def get_config(c: ServiceContext = Depends(ctx)) -> dict:
    return c.db.get_kv(CONFIG_KEY, TtsConfig().model_dump())


@router.put("/tts/config", response_model=TtsConfig)
def put_config(config: TtsConfig, c: ServiceContext = Depends(ctx)) -> dict:
    values = config.model_dump()
    c.db.set_kv(CONFIG_KEY, values)
    return values


@router.post("/tts/validate")
def validate_config(body: ValidationBody, c: ServiceContext = Depends(ctx)) -> dict:
    return _inspect(body.config, c)[1]


def _devices(requested: list[str]) -> list[str]:
    from ypuddin.tts.issues import TtsIssue

    def reject(message: str, code: str):
        issue = TtsIssue(code=code, loc=["gpu_devices"], message=message)
        raise ApiError(message, code=code, status=422, details={"issues": [issue.model_dump()]})

    inventory = gpu_info()
    if error := tts_device_error(inventory):
        reject(error, "tts.device")
    if error := selection_error(requested, 1, inventory):
        reject(error, "job.gpu_selection")
    if any(not device.startswith("cuda:") for device in requested):
        reject("此语音训练器需要 CUDA 显卡。", "tts.device")
    return requested


def _snapshot(c: ServiceContext, jid: str, typed: TtsConfig, *, sample: bool = False) -> tuple[Path, dict]:
    run_dir = c.data_root / "tts" / "jobs" / jid
    for path in (run_dir.parent.parent, run_dir.parent, run_dir):
        if path.is_symlink():
            raise ApiError("语音任务目录不能使用符号链接。", code="tts.path", status=409)
    products = run_dir if sample else c.job_output_dir(None, None, jid)
    samples = c.job_storage_dir(None, None, jid, "samples_dir", run_dir)
    logs = c.job_storage_dir(None, None, jid, "logs_dir", run_dir)
    return run_dir, {
        "tts": typed.model_dump(),
        "checkpoint": {"output_dir": str(products), "state_dir": str(run_dir / "resume"), "resume": None},
        "logging": {"events_path": str(logs / "events.jsonl"), "output_dir": str(logs)},
        "sampling": {"output_dir": str(samples)},
    }


def _enqueue(c: ServiceContext, jid: str, name: str, kind: str, run_dir: Path, payload: dict, devices: list[str]) -> dict:
    c.db.insert("jobs", {
        "id": jid, "type": kind, "name": name, "status": "queued", "priority": 0,
        "created_at": now(), "run_dir": str(run_dir), "samples_dir": payload["sampling"]["output_dir"],
        "gpu_devices_json": json.dumps(devices), "config_json": json.dumps(payload, ensure_ascii=False),
        "progress_json": "{}", "latest_json": "{}",
    })
    c.bus.publish("queue.changed", {})
    c.bus.publish("job.state", {"job_id": jid, "status": "queued"})
    return _job_row(c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,)), c)


@router.post("/tts/jobs", responses={409: {"model": m.ApiErrorResponse}})
def create_training(body: TrainingBody, c: ServiceContext = Depends(ctx)) -> dict:
    raise ApiError("请在语音项目的版本中创建训练任务。", code="tts.scope_required", status=409)


def _job(c: ServiceContext, jid: str, *, training: bool = False) -> dict:
    row = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    if not row or row["type"] not in ({"tts_train"} if training else TTS_JOBS):
        raise NotFound("语音任务不存在。", code="tts.not_found")
    if jid in deleting_jobs:
        raise ApiError("这个任务的文件正在删除。", code="job.deleting", status=409)
    return row


def _relative_checkpoint(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not value:
        raise ApiError("请选择此训练任务中的检查点。", code="tts.checkpoint", status=422)
    try:
        return safe_checkpoint(root, path)
    except (OSError, ValueError) as error:
        raise ApiError(str(error), code="tts.checkpoint", status=422) from error


def _stable_root(path: Path) -> Path:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ApiError("任务输出目录不能通过符号链接重定向。", code="tts.path", status=409)
    return path




def launch_payload(job: dict, device: str | None = None) -> dict:
    """Recheck immutable requests just before launch or retry, without loading a model."""
    payload = json.loads(job["config_json"])
    if "tts_inputs" in payload:
        payload["tts"] = parse_execution_config(payload["tts"]).model_dump()
    else:
        payload["tts"] = _normalize(payload["tts"]).model_dump()
    if device is not None:
        if device != "cuda:0":
            raise ValueError("此语音训练器需要分配一张 NVIDIA CUDA 显卡。")
        payload["device"] = device
    if job["type"] == "tts_sample":
        sample = payload["tts_sample"]
        _stable_root(Path(sample["source_output_dir"]))
        if payload["tts"].get("engine") == "gpt-sovits-v5":
            from ypuddin.tts.gpt_sovits.core import safe_checkpoint as gsv_checkpoint

            sample["checkpoint"] = str(gsv_checkpoint(sample["source_output_dir"], sample["checkpoint"]))
        else:
            sample["checkpoint"] = str(safe_checkpoint(sample["source_output_dir"], sample["checkpoint"]))
    return payload


@contextmanager
def cloned_manifests(payload: dict, run_dir: Path):
    """A retry owns its manifests even after the original job and files are removed."""
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        for key in ("train_manifest", "val_manifest"):
            if source := payload["tts"].get(key):
                target = run_dir / f"{key}.jsonl"
                shutil.copyfile(source, target)
                payload["tts"][key] = str(target)
        yield
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _audio_path(job: dict, filename: str) -> Path:
    if not filename or filename != Path(filename).name or "\\" in filename or Path(filename).suffix.lower() != ".wav":
        raise NotFound("音频不存在。", code="tts.audio_not_found")
    root = _stable_root(Path(job["samples_dir"]))
    path = root / filename
    if root.is_symlink() or path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        raise NotFound("音频不存在。", code="tts.audio_not_found")
    return path
