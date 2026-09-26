"""Version-owned XYZ sampling requests and durable jobs in the shared GPU queue."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ypuddin.config import MemoryConfig, ModelConfig, SamplingConfig
from ypuddin.models import get_family

from .db import new_id, now
from .errors import ApiError, NotFound
from .gpu_selection import GpuSelection, selection_error
from .hardware import gpu_info

AXES = {
    "steps": "Steps",
    "cfg": "CFG",
    "seed": "Seed",
    "sampler": "Sampler",
    "scheduler": "Scheduler",
    "shift": "Shift",
    "adapter_scale": "Adapter strength",
    "checkpoint": "Checkpoint",
}
# Cells are separate images; the bound only catches runaway grids such as a mistyped range.
MAX_CELLS = 1000


class XyzAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: Literal["steps", "cfg", "seed", "sampler", "scheduler", "shift", "adapter_scale", "checkpoint"]
    values: list[Any] = Field(min_length=1)

    @model_validator(mode="after")
    def typed_values(self):
        for value in self.values:
            if self.key in {"sampler", "scheduler", "checkpoint"}:
                valid = isinstance(value, str) and bool(value.strip()) and len(value) <= 200
            elif self.key in {"steps", "seed"}:
                valid = type(value) is int and (
                    1 <= value <= 100 if self.key == "steps" else 0 <= value < 2**63
                )
            else:
                valid = type(value) in {int, float} and math.isfinite(value)
                if valid:
                    valid = {
                        "cfg": 0 <= value <= 30,
                        "shift": 0 < value <= 100,
                        "adapter_scale": -4 <= value <= 4,
                    }[self.key]
            if not valid:
                raise ValueError(f"invalid {self.key} axis value: {value!r}")
        if len(set(self.values)) != len(self.values):
            raise ValueError("axis values must be distinct")
        return self


class XyzRequest(GpuSelection):
    model_config = ConfigDict(extra="forbid")
    name: str = Field("模型测试", min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=8000)
    negative: str = Field("", max_length=8000)
    width: int = Field(512, ge=32, le=2048)
    height: int = Field(512, ge=32, le=2048)
    steps: int = Field(20, ge=1, le=100)
    cfg: float = Field(4, ge=0, le=30, allow_inf_nan=False)
    seed: int = Field(1, ge=0, lt=2**63)
    sampler: str = "euler"
    scheduler: str = "uniform"
    shift: float | None = Field(None, gt=0, le=100, allow_inf_nan=False)
    guidance: float | None = Field(None, ge=0, le=30, allow_inf_nan=False)
    adapter_scale: float = Field(1, ge=-4, le=4, allow_inf_nan=False)
    checkpoint_id: str | None = None
    sampling_model_id: str | None = None
    x: XyzAxis
    y: XyzAxis | None = None
    z: XyzAxis | None = None

    @model_validator(mode="after")
    def bounded_grid(self):
        axes = [axis for axis in (self.x, self.y, self.z) if axis]
        if len({axis.key for axis in axes}) != len(axes):
            raise ValueError("X, Y and Z must use different parameters")
        count = math.prod(len(axis.values) for axis in axes)
        if count > MAX_CELLS:
            raise ValueError(f"模型测试一次最多 {MAX_CELLS} 张，请减少参数值。")
        if any(axis.key == "adapter_scale" for axis in axes) and not (
            self.checkpoint_id or any(axis.key == "checkpoint" for axis in axes)
        ):
            raise ValueError("Adapter strength requires a saved checkpoint")
        return self


class XyzManifest(BaseModel):
    cells: list[dict[str, Any]] = Field(default_factory=list)
    grids: list[dict[str, Any]] = Field(default_factory=list)
    axes: dict[str, Any] = Field(default_factory=dict)
    complete: bool = False


class XyzTask(BaseModel):
    id: str
    job_id: str
    source_job_id: str
    status: str
    phase: str
    done: int
    total: int
    cell_index: int | None = None
    sample_step: int | None = None
    sample_steps: int | None = None
    error: str | None
    created_at: float
    finished_at: float | None
    request: dict[str, Any]
    manifest: XyzManifest
    can_cancel: bool


class XyzOptions(BaseModel):
    family: str
    training_mode: Literal["adapter", "full"] = "adapter"
    defaults: dict[str, Any]
    axes: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]
    sampling_models: list[dict[str, Any]]
    limits: dict[str, int]


def expand_cells(request: XyzRequest) -> list[dict[str, Any]]:
    cells = []
    axes = [request.x, request.y, request.z]
    ranges = [list(enumerate(axis.values)) if axis else [(0, None)] for axis in axes]
    for (zi, zv), (yi, yv), (xi, xv) in itertools.product(ranges[2], ranges[1], ranges[0]):
        cell = request.model_dump(exclude={"name", "prompt", "negative", "x", "y", "z", "gpu_devices"})
        for axis, value in zip(axes, (xv, yv, zv), strict=True):
            if axis:
                cell["checkpoint_id" if axis.key == "checkpoint" else axis.key] = value
        cells.append(
            cell
            | {"index": len(cells), "x": xi, "y": yi, "z": zi, "x_value": xv, "y_value": yv, "z_value": zv}
        )
    return cells


def file_signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def checkpoint_signature(path: Path):
    """Pin all native component/config files, not just the directory inode."""
    if path.is_file():
        return file_signature(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Checkpoint is missing or not a regular model directory")
    signature = {}
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            raise ValueError("Model checkpoint must not contain symbolic links")
        if entry.is_file():
            signature[entry.relative_to(path).as_posix()] = file_signature(entry)
    if not signature:
        raise ValueError("Model checkpoint is empty")
    return signature


def full_checkpoint_model(path: Path, family: str) -> ModelConfig:
    from ypuddin.config import load_config

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "ypuddin-full-model-v1" or manifest.get("family") != family:
        raise ValueError("Invalid full-model checkpoint format or family")
    if not manifest.get("components"):
        raise ValueError("Full-model checkpoint contains no trained components")
    config = load_config(path / "config.toml")
    if config.training.mode != "full" or config.model.family != family:
        raise ValueError("Full-model checkpoint configuration differs from its manifest")
    return config.model


def _source(context, source_id):
    row = context.db.fetchone("SELECT * FROM jobs WHERE id=?", (source_id,))
    if not row or row["type"] != "train":
        raise NotFound("请为模型测试选择一个来源训练任务。", code="xyz.source")
    return row


def _checkpoints(context, source_id, *, full=False):
    records = []
    for row in context.db.fetchall(
        "SELECT * FROM artifacts WHERE job_id=? AND kind=? ORDER BY step DESC, created_at DESC",
        (source_id, "model" if full else "weights"),
    ):
        path = Path(row["path"]).expanduser().resolve()
        exists = (
            path.is_dir() and (path / "manifest.json").is_file()
            if full
            else (path.is_file() and path.suffix.lower() == ".safetensors")
        )
        if exists and context.is_allowed(path):
            records.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "step": row["step"],
                    "kind": "model" if full else "weights",
                    "path": str(path),
                }
            )
    return records


def options(context, source_id):
    source = _source(context, source_id)
    config = json.loads(source["config_json"])
    model = ModelConfig.model_validate(config["model"])
    family = get_family(model.family)
    sampling = SamplingConfig.model_validate(config.get("sampling", {}))
    prompt = sampling.prompts[0] if sampling.prompts else None
    full = config.get("training", {}).get("mode") == "full"
    checkpoints = _checkpoints(context, source_id, full=full)
    defaults = {
        key: getattr(sampling, key)
        for key in ("width", "height", "steps", "cfg", "sampler", "scheduler", "shift", "guidance")
    }
    defaults.update(
        prompt=prompt.prompt if prompt else "",
        negative=prompt.negative if prompt else "",
        seed=prompt.seed if prompt and prompt.seed is not None else 1,
        adapter_scale=1,
        checkpoint_id=checkpoints[0]["id"] if checkpoints else None,
        sampling_model_id=None,
    )
    if prompt:
        for key in ("width", "height", "steps", "cfg"):
            if getattr(prompt, key) is not None:
                defaults[key] = getattr(prompt, key)
    models = []
    from .routes_core import _model_row

    for row in context.db.fetchall(
        "SELECT * FROM models WHERE family=? AND kind='dit' ORDER BY created_at DESC", (model.family,)
    ):
        projected = _model_row(row, context)
        if model.family == "krea2" and projected.get("variant") not in {"raw", "turbo"}:
            continue
        if projected["exists"] and not projected.get("unsupported_reason"):
            models.append(
                {
                    "id": row["id"],
                    "name": Path(row["path"]).name,
                    "variant": projected.get("variant"),
                    "purpose": projected.get("purpose"),
                }
            )
    axes = [
        {"key": key, "label": label} for key, label in AXES.items() if not (full and key == "adapter_scale")
    ]
    for axis in axes:
        if axis["key"] in {"sampler", "scheduler"}:
            axis["values"] = list(getattr(family.spec, "sampling_" + axis["key"] + "s"))
    return {
        "family": model.family,
        "training_mode": "full" if full else "adapter",
        "defaults": defaults,
        "axes": axes,
        "checkpoints": [{k: v for k, v in row.items() if k != "path"} for row in checkpoints],
        "sampling_models": [] if full else models,
        "limits": {"max_cells": MAX_CELLS},
    }


def start(context, source_id: str, request: XyzRequest):
    if error := selection_error(request.gpu_devices, 1, gpu_info()):
        raise ApiError(error, code="job.gpu_selection", status=422)
    source = _source(context, source_id)
    context.supervisor._check_job_version(source)
    config = json.loads(source["config_json"])
    model = ModelConfig.model_validate(config["model"])
    full = config.get("training", {}).get("mode") == "full"
    cells = expand_cells(request)
    checkpoints = {row["id"]: row for row in _checkpoints(context, source_id, full=full)}
    if full:
        if (
            request.sampling_model_id
            or request.adapter_scale != 1
            or any(axis and axis.key == "adapter_scale" for axis in (request.x, request.y, request.z))
        ):
            raise ApiError(
                "Full-model results use their exported components; adapter strength and replacement backbones are unavailable",
                code="xyz.full_model",
                status=422,
            )
        if any(not cell["checkpoint_id"] or cell["checkpoint_id"] not in checkpoints for cell in cells):
            raise ApiError(
                "请为模型测试中的每张图片选择已导出的完整模型，不能使用原始底模代替。",
                code="xyz.checkpoint",
                status=422,
            )
        try:
            first_path = Path(checkpoints[cells[0]["checkpoint_id"]]["path"])
            checkpoint_signature(first_path)
            model = full_checkpoint_model(first_path, model.family)
        except (OSError, ValueError) as exc:
            raise ApiError(
                f"Cannot read full-model checkpoint: {exc}", code="xyz.checkpoint", status=422
            ) from exc
    if request.sampling_model_id:
        row = context.db.fetchone("SELECT * FROM models WHERE id=?", (request.sampling_model_id,))
        if not row or row["family"] != model.family or row["kind"] != "dit":
            raise ApiError(
                "Select a registered sampling backbone from the same model family",
                code="xyz.model",
                status=422,
            )
        from .routes_core import _model_row

        projected = _model_row(row, context)
        if projected.get("unsupported_reason"):
            raise ApiError(projected["unsupported_reason"], code="xyz.model", status=422)
        if model.family == "krea2" and projected.get("variant") not in {"raw", "turbo"}:
            raise ApiError(
                "Confirm this Krea 2 model as Raw or Turbo in the model library before selecting it for sampling",
                code="xyz.variant",
                status=422,
            )
        model.dit_path = row["path"]
        if model.family == "krea2" and projected.get("variant"):
            model.krea2_variant = projected["variant"]
    family = get_family(model.family)
    errors = ([family.spec.retired_reason] if family.spec.retired_reason else []) + family.validate_config(
        model
    )
    if errors:
        raise ApiError("; ".join(errors), code="xyz.model", status=422)
    if model.family == "krea2" and model.krea2_variant == "turbo":
        # The API's generic defaults describe a non-distilled model. Resolve only
        # omitted fixed values after the selected model's identity is validated;
        # explicit values and every axis remain authoritative.
        request = request.model_copy(
            update={
                key: value
                for key, value in {"steps": 8, "cfg": 0.0}.items()
                if key not in request.model_fields_set
            }
        )
    for field in ("dit_path", "text_encoder_path", "text_encoder_2_path", "vae_path", "tokenizer_path"):
        value = getattr(model, field)
        if value and not context.is_allowed(Path(value).expanduser().resolve()):
            raise ApiError("Model is outside allowed roots", code="xyz.path", status=403)
    if request.width % family.spec.latent.align or request.height % family.spec.latent.align:
        raise ApiError(
            f"Image dimensions must be multiples of {family.spec.latent.align}",
            code="xyz.dimensions",
            status=422,
        )
    resolved = {}
    cells = expand_cells(request)
    for cell in cells:
        try:
            sampling = SamplingConfig.model_validate(
                {
                    key: cell[key]
                    for key in (
                        "width",
                        "height",
                        "steps",
                        "cfg",
                        "sampler",
                        "scheduler",
                        "shift",
                        "guidance",
                    )
                }
            )
        except ValidationError as exc:
            raise ApiError(str(exc), code="xyz.axis", status=422) from exc
        problems = family.sampling_errors(sampling)
        if problems:
            raise ApiError("; ".join(item["msg"] for item in problems), code="xyz.axis", status=422)
        checkpoint = cell["checkpoint_id"]
        if checkpoint and checkpoint not in resolved:
            if checkpoint not in checkpoints:
                raise ApiError(
                    "Checkpoint is missing or belongs to a different training job",
                    code="xyz.checkpoint",
                    status=422,
                )
            path = Path(checkpoints[checkpoint]["path"])
            try:
                if full:
                    checkpoint_model = full_checkpoint_model(path, model.family)
                    problems = family.validate_config(checkpoint_model)
                    if problems:
                        raise ValueError("; ".join(problems))
                    for field in (
                        "dit_path",
                        "text_encoder_path",
                        "text_encoder_2_path",
                        "vae_path",
                        "tokenizer_path",
                    ):
                        value = getattr(checkpoint_model, field)
                        if value and not context.is_allowed(Path(value).expanduser().resolve()):
                            raise ApiError("Model is outside allowed roots", code="xyz.path", status=403)
                    resolved[checkpoint] = checkpoints[checkpoint] | {
                        "signature": checkpoint_signature(path),
                        "model": checkpoint_model.model_dump(mode="json"),
                    }
                else:
                    from safetensors import safe_open

                    with safe_open(path, framework="pt", device="cpu") as handle:
                        metadata = handle.metadata() or {}
                        if metadata.get("ypuddin.family", model.family) != model.family:
                            raise ValueError("Checkpoint model family does not match the sampling model")
                    resolved[checkpoint] = checkpoints[checkpoint] | {"signature": checkpoint_signature(path)}
            except ApiError:
                raise
            except Exception as exc:
                raise ApiError(f"Cannot read checkpoint: {exc}", code="xyz.checkpoint", status=422) from exc
    memory = MemoryConfig.model_validate(config.get("memory", {}))
    # Klein's loader requires block checkpointing when swap is enabled. Keeping
    # that configuration satisfies its memory contract; eval/inference_mode below
    # means no gradient checkpoint is actually evaluated during sampling.
    memory.activation_checkpointing = "block" if model.family == "flux2" and memory.blocks_to_swap else "none"
    memory.compile = False
    if memory.blocks_to_swap and "block_swap" not in family.spec.capabilities:
        raise ApiError("Selected family does not support block swapping", code="xyz.memory", status=422)
    jid = new_id("j")
    # A model test makes no products, so all of its files stay in its records folder.
    run_dir = (
        context.job_records_dir(source["project_id"], source.get("version_id"), jid)
        if source["project_id"]
        else context.job_output_dir(None, None, jid)
    )
    samples_dir = (
        context.job_storage_dir(source["project_id"], source.get("version_id"), jid, "samples_dir", run_dir)
        if context.settings()["paths"].get("samples_dir")
        else run_dir / "samples"
    )
    logs_dir = context.job_storage_dir(
        source["project_id"], source.get("version_id"), jid, "logs_dir", run_dir
    )
    payload = {
        "model": model.model_dump(mode="json"),
        "memory": memory.model_dump(mode="json"),
        "training": {"mode": "full" if full else "adapter"},
        "xyz": {
            "source_job_id": source_id,
            "request": request.model_dump(mode="json"),
            "checkpoints": resolved,
        },
        "checkpoint": {"output_dir": str(run_dir)},
        "sampling": {"output_dir": str(samples_dir)},
        "logging": {"events_path": str(logs_dir / "events.jsonl"), "output_dir": str(logs_dir)},
    }
    with context.db.lock:
        # Header validation happens outside the queue lock. A deletion may have
        # completed in that interval, so publish references only while their
        # records and checkpoint files still exist under the same lock used by
        # the deletion endpoints.
        source = _source(context, source_id)
        context.supervisor._check_job_version(source)
        current_checkpoints = {row["id"]: row for row in _checkpoints(context, source_id, full=full)}
        for checkpoint_id, checkpoint in resolved.items():
            current = current_checkpoints.get(checkpoint_id)
            try:
                unchanged = (
                    current is not None
                    and current["path"] == checkpoint["path"]
                    and checkpoint_signature(Path(current["path"])) == checkpoint["signature"]
                )
            except (OSError, ValueError):
                unchanged = False
            if not unchanged:
                raise ApiError(
                    "A selected checkpoint changed or was removed; refresh the checkpoint list and try again",
                    code="xyz.checkpoint",
                    status=422,
                )
        context.db.insert(
            "jobs",
            {
                "id": jid,
                "type": "xyz",
                "name": request.name,
                "project_id": source["project_id"],
                "version_id": source.get("version_id"),
                "status": "queued",
                "priority": 0,
                "gpu_devices_json": json.dumps(request.gpu_devices),
                "created_at": now(),
                "run_dir": str(run_dir),
                "samples_dir": str(samples_dir),
                "config_json": json.dumps(payload),
                "progress_json": json.dumps({"done": 0, "total": len(cells)}),
                "latest_json": "{}",
            },
        )
    context.bus.publish("queue.changed", {})
    return task(context, jid)


def _row(context, jid):
    row = context.db.fetchone("SELECT * FROM jobs WHERE id=? AND type='xyz'", (jid,))
    if not row:
        raise NotFound("模型测试任务不存在。", code="xyz.not_found")
    return row


def result_root(context, row):
    root = (
        Path(row["samples_dir"]).expanduser()
        if row.get("samples_dir")
        else Path(row["run_dir"]).expanduser() / "samples"
    )
    if (
        root.is_symlink()
        or any(parent.is_symlink() for parent in root.parents)
        or not context.is_allowed(root.resolve())
    ):
        raise ApiError("模型测试结果目录不在允许访问的范围内。", code="xyz.path", status=403)
    return root.resolve()


def task(context, jid):
    row = _row(context, jid)
    snapshot = json.loads(row["config_json"])["xyz"]
    progress = json.loads(row["progress_json"] or "{}")
    path = result_root(context, row) / "manifest.json"
    manifest = {"cells": [], "grids": [], "axes": {}, "complete": False}
    if path.is_file() and not path.is_symlink():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest.get("cells", []) + manifest.get("grids", []):
        entry["url"] = f"/api/xyz/{jid}/file?name={entry['file']}"
    return {
        "id": jid,
        "job_id": jid,
        "source_job_id": snapshot["source_job_id"],
        "status": row["status"],
        "phase": progress.get("phase", row["status"]),
        "done": progress.get("done", len(manifest["cells"])),
        "total": progress.get("total", len(expand_cells(XyzRequest.model_validate(snapshot["request"])))),
        "cell_index": progress.get("cell_index"),
        "sample_step": progress.get("sample_step"),
        "sample_steps": progress.get("sample_steps"),
        "error": row["error"],
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
        "request": snapshot["request"],
        "manifest": manifest,
        "can_cancel": row["status"] in {"queued", "scheduled", "running", "pausing"},
    }


def history(context, source_id):
    _source(context, source_id)
    return [
        task(context, row["id"])
        for row in context.db.fetchall(
            "SELECT id FROM jobs WHERE type='xyz' AND json_extract(config_json,'$.xyz.source_job_id')=? ORDER BY created_at DESC",
            (source_id,),
        )
    ]
