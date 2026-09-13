"""System / settings / fs / schema / plan / presets / models / events endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

import psutil
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import ypuddin
from ypuddin.config import TrainConfig, deep_merge, dump_toml, read_config_file
from ypuddin.models import available as available_families
from ypuddin.train.plan import plan as make_plan

from . import models as m
from .context import ServiceContext
from .db import new_id, now
from .errors import ApiError, NotFound
from .hardware import gpu_info

router = APIRouter()


def ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


# --------------------------------------------------------------------------- system
def system_stats(data_root: Path) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disks = []
    try:
        du = psutil.disk_usage(str(data_root))
        disks.append(
            {
                "path": str(data_root),
                "used_gb": round(du.used / 2**30, 1),
                "total_gb": round(du.total / 2**30, 1),
            }
        )
    except Exception:  # noqa: BLE001
        pass
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        # Available memory accounts for reclaimable caches consistently across platforms.
        # MPS reports this same system pool, so both tiles use one snapshot and definition.
        "ram": {"used_mb": round((vm.total - vm.available) / 2**20), "total_mb": round(vm.total / 2**20)},
        "disks": disks,
        "gpus": gpu_info(include_unavailable=True, system_memory=vm),
    }


@router.get("/health", response_model=m.Health, response_model_exclude_unset=True)
def health(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    try:
        import torch

        torch_v, cuda = torch.__version__, torch.version.cuda
    except Exception:  # noqa: BLE001
        torch_v, cuda = None, None
    devices = gpu_info()
    return {
        "version": ypuddin.__version__,
        "api_version": ypuddin.API_VERSION,
        "torch": torch_v,
        "cuda": cuda,
        "mps": any(g["kind"] == "mps" for g in devices),
        "gpus": [
            {"index": g["index"], "kind": g["kind"], "name": g["name"], "total_mb": g["mem_total_mb"]}
            for g in devices
        ],
        "families": available_families(),
    }


@router.get("/system/stats", response_model=m.SystemStats, response_model_exclude_unset=True)
def stats(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return system_stats(c.data_root)


@router.get("/system/info", response_model=m.SystemInfo, response_model_exclude_unset=True)
def info() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    import torch

    mods = {}
    for name in ("torch", "transformers", "safetensors", "pydantic", "fastapi"):
        try:
            mods[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            mods[name] = None
    try:
        mods["nvidia-ml-py"] = version("nvidia-ml-py")
    except PackageNotFoundError:
        mods["nvidia-ml-py"] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": mods,
        "ypuddin": ypuddin.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


# --------------------------------------------------------------------------- families
_FAMILY_INFO: dict[str, dict[str, Any]] = {}


def family_info(name: str) -> dict[str, Any]:
    """Everything a UI needs to render a family: presets (with matched-layer counts on the official geometry),
    capabilities, valid text modes, sampling defaults, latent alignment and the weight files it expects."""
    if name in _FAMILY_INFO:
        return _FAMILY_INFO[name]
    from ypuddin.adapters.rules import resolve_targets
    from ypuddin.config import AdapterConfig
    from ypuddin.models import get_family

    fam = get_family(name)
    spec = fam.spec
    names: list[str] = []
    if hasattr(fam, "linear_module_names"):
        try:
            names = fam.linear_module_names()
        except Exception:  # noqa: BLE001
            names = []
    probe = AdapterConfig(algo="lora", rank=4, alpha=4)
    presets = []
    for pname, preset in fam.presets().items():
        layers = len(resolve_targets(names, probe, preset)) if names else 0
        presets.append(
            {
                "name": pname,
                "description": preset.description,
                "include": list(preset.include),
                "exclude": list(preset.exclude),
                "layers": layers,
            }
        )
    text_modes = ["auto", "cached"] + (["online"] if "online_text" in spec.capabilities else [])
    info = {
        "name": spec.name,
        "label": spec.label or spec.name,
        "architecture": spec.architecture,
        "objective": spec.objective,
        "attention_backends": list(spec.attention_backends),
        "sampling_samplers": list(spec.sampling_samplers),
        "sampling_schedulers": list(spec.sampling_schedulers),
        "objective_timestep_sampling": list(spec.objective_timestep_sampling),
        "objective_weighting": list(spec.objective_weighting),
        "adapter_prefix": spec.adapter_prefix,
        "capabilities": sorted(spec.capabilities),
        "text_modes": text_modes,
        "presets": presets,
        "default_preset": fam.default_preset(),
        "sampling": {
            "steps": spec.sampling.steps,
            "cfg": spec.sampling.cfg,
            "shift": spec.sampling.shift,
            "sampler": spec.sampling.sampler,
            "guidance": spec.sampling.guidance,
        },
        "latent": {
            "channels": spec.latent.channels,
            "stride": spec.latent.stride,
            "patch": spec.latent.patch,
            "align": spec.latent.align,
        },
        "text_max_len": spec.text.max_len,
        "weights": [
            {
                "field": f,
                "label": lbl,
                "hint": hint,
                "kind": f.removesuffix("_path"),
                "required": f not in spec.optional_weights,
                "downloadable": f not in spec.directory_only_weights,
            }
            for f, lbl, hint in spec.weights
        ],
        "linear_modules": len(names),
    }
    _FAMILY_INFO[name] = info
    return info


@router.get("/families", response_model=list[m.FamilyInfo], response_model_exclude_unset=True)
def list_families() -> list[dict[str, Any]]:
    return [family_info(n) for n in available_families()]


@router.get("/families/{name}", response_model=m.FamilyInfo, response_model_exclude_unset=True)
def get_family_info(name: str) -> dict[str, Any]:
    if name not in available_families():
        raise NotFound(f"unknown model family {name!r}")
    return family_info(name)


# --------------------------------------------------------------------------- settings / fs
@router.get("/settings", response_model=m.Settings, response_model_exclude_unset=True)
def get_settings(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return c.settings()


@router.put("/settings", response_model=m.Settings, response_model_exclude_unset=True)
def put_settings(patch: dict[str, Any], c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    try:
        return c.save_settings(patch)
    except (ValueError, TypeError, KeyError) as exc:
        raise ApiError(str(exc), code="settings.invalid") from exc


@router.get("/fs/list", response_model=m.FsList, response_model_exclude_unset=True)
def fs_list(path: str = "", c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    p = Path(path).expanduser() if path else c.data_root
    if not p.exists() or not p.is_dir():
        raise NotFound(f"directory not found: {p}", code="fs.not_found")
    entries = []
    try:
        for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if child.name.startswith("."):
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": child.name,
                    "is_dir": child.is_dir(),
                    "size": st.st_size if child.is_file() else None,
                    "mtime": st.st_mtime,
                }
            )
    except PermissionError as e:
        raise ApiError(str(e), code="fs.permission", status=403) from e
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "entries": entries[:2000]}


# --------------------------------------------------------------------------- schema / config / plan
@router.get("/schema/train")
def schema_train() -> dict[str, Any]:
    return TrainConfig.json_schema()


@router.get("/config/defaults", response_model=TrainConfig)
def config_defaults(
    family: Literal["anima", "krea2", "sdxl", "flux", "flux2", "toy"] | None = None,
    c: ServiceContext = Depends(ctx),
) -> dict[str, Any]:
    from .environment import environment_attention_default

    if family is not None:
        from .family_config import initial_family_config

        return initial_family_config(c, family)
    defaults = TrainConfig().to_dict()
    defaults["model"]["attention"] = environment_attention_default(c)
    return defaults


class ConfigImportBody(BaseModel):
    text: str
    format: Literal["toml", "json"] = "toml"


class ConfigExportBody(BaseModel):
    config: dict[str, Any]
    format: Literal["toml", "json"] = "toml"


class ConfigText(BaseModel):
    text: str


def _validated_or_error(raw: dict[str, Any]) -> TrainConfig:
    cfg, errors = _validate(raw)
    if cfg is None:
        raise ApiError("invalid config", code="config.invalid", details={"errors": errors})
    return cfg


@router.post("/config/import", response_model=TrainConfig)
def config_import(body: ConfigImportBody) -> dict[str, Any]:
    from ypuddin.config.io import parse_toml

    try:
        raw = json.loads(body.text) if body.format == "json" else parse_toml(body.text)
    except ValueError as exc:
        raise ApiError(str(exc), code="config.parse") from exc
    if not isinstance(raw, dict):
        raise ApiError("config must be an object", code="config.invalid")
    return _validated_or_error(raw).to_dict()


@router.post("/config/export", response_model=ConfigText)
def config_export(body: ConfigExportBody) -> dict[str, str]:
    cfg = _validated_or_error(body.config)
    return {
        "text": dump_toml(cfg)
        if body.format == "toml"
        else json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)
    }


class ConfigBody(BaseModel):
    config: dict[str, Any]
    dataset_ids: list[str] | None = None
    project_id: str | None = None
    version_id: str | None = None


def _scoped_config(body: ConfigBody, c: ServiceContext) -> dict[str, Any]:
    if body.version_id and not body.project_id:
        raise ApiError("version_id requires project_id", code="version.project_required", status=422)
    if body.project_id:
        from .output_binding import bind_output_name
        from .source_roles import normalize_source_roles

        config = normalize_source_roles(c, body.project_id, body.config, body.version_id)
        return bind_output_name(c, body.project_id, config, body.version_id)
    return body.config


def _validate(config: dict[str, Any]) -> tuple[TrainConfig | None, list[dict[str, str]]]:
    from pydantic import ValidationError

    try:
        return TrainConfig.model_validate(config), []
    except ValidationError as e:
        return None, [{"loc": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]} for err in e.errors()]


@router.post("/config/validate", response_model=m.ValidateResult, response_model_exclude_unset=True)
def config_validate(body: ConfigBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    cfg, errors = _validate(_scoped_config(body, c))
    return {"ok": cfg is not None, "errors": errors, "warnings": []}


@router.post("/plan", response_model=m.Plan, response_model_exclude_unset=True)
def config_plan(body: ConfigBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    # plan keeps full validation errors while previewing independently valid data fields.
    cfg = _scoped_config(body, c)
    gpus = gpu_info()
    return make_plan(
        cfg,
        gpu_total_mb=gpus[0]["mem_total_mb"] if gpus else None,
        device=gpus[0]["device"] if gpus else "cpu",
    )


# --------------------------------------------------------------------------- presets
def _preset_dir(c: ServiceContext) -> Path:
    d = c.data_root / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _preset_row(name: str, data: dict[str, Any], builtin: bool, updated_at: float | None) -> dict[str, Any]:
    return {
        "name": name,
        "description": data.get("description", ""),
        "config": data.get("config", {}),
        "builtin": builtin,
        "updated_at": updated_at,
    }


@router.get("/presets", response_model=list[m.Preset], response_model_exclude_unset=True)
def list_presets(c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    out = []
    for f in sorted(_preset_dir(c).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        out.append(_preset_row(f.stem, data, False, f.stat().st_mtime))
    return out


class PresetBody(BaseModel):
    name: str
    description: str = ""
    config: dict[str, Any]


_preset_lock = threading.RLock()


def _preset_path(name: str, c: ServiceContext) -> Path:
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1, 10)
    }
    if len(name) > 128 or not name.replace("-", "").replace("_", "").isalnum() or name.casefold() in reserved:
        raise ApiError("preset name must be 1–128 letters or numbers with - or _", code="preset.bad_name")
    return _preset_dir(c) / f"{name}.json"


def _write_preset(name: str, body: PresetBody, c: ServiceContext, *, create: bool) -> dict[str, Any]:
    path = _preset_path(name, c)
    # Validate partial presets against the appropriate family without persisting
    # expanded defaults or requiring any project data/model files to exist.
    from .family_config import initial_family_config

    family = (
        body.config.get("model", {}).get("family", "anima")
        if isinstance(body.config.get("model", {}), dict)
        else "anima"
    )
    if not isinstance(family, str):
        raise ApiError(
            "invalid config",
            code="config.invalid",
            details={"errors": [{"loc": "model.family", "msg": "family must be a string"}]},
        )
    _validated_or_error(deep_merge(initial_family_config(c, family), body.config))
    with _preset_lock:
        collision = next(
            (f for f in path.parent.glob("*.json") if f.stem.casefold() == name.casefold()), None
        )
        if create and collision is not None:
            raise ApiError("a preset with this name already exists", code="preset.duplicate", status=409)
        if not create and (collision is None or collision.name != path.name):
            raise NotFound(f"preset {name} not found", code="preset.not_found")
        data = {"description": body.description, "config": body.config}
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=path.parent, prefix=".preset-", suffix=".tmp", encoding="utf-8", delete=False
            ) as handle:
                temporary = handle.name
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return _preset_row(name, data, False, path.stat().st_mtime)


@router.post("/presets", response_model=m.Preset, response_model_exclude_unset=True)
def create_preset(body: PresetBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return _write_preset(body.name, body, c, create=True)


@router.get("/presets/{name}", response_model=m.Preset, response_model_exclude_unset=True)
def get_preset(name: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    f = _preset_path(name, c)
    if not f.exists():
        raise NotFound(f"preset {name} not found", code="preset.not_found")
    return _preset_row(name, json.loads(f.read_text(encoding="utf-8")), False, f.stat().st_mtime)


@router.put("/presets/{name}", response_model=m.Preset, response_model_exclude_unset=True)
def put_preset(name: str, body: PresetBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return _write_preset(name, body, c, create=False)


@router.delete("/presets/{name}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_preset(name: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    f = _preset_path(name, c)
    with _preset_lock:
        if not f.exists():
            raise NotFound(f"preset {name} not found", code="preset.not_found")
        f.unlink()
    return {"ok": True}


@router.post("/presets/{name}/resolve")
def resolve_preset(name: str, body: ConfigBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    """Overlay ``body.config`` on the preset and return the fully resolved config (+ TOML)."""
    preset = get_preset(name, c)
    merged = deep_merge(TrainConfig().to_dict(), deep_merge(preset["config"], body.config))
    cfg, errors = _validate(merged)
    return {
        "ok": cfg is not None,
        "errors": errors,
        "config": cfg.to_dict() if cfg else merged,
        "toml": dump_toml(cfg) if cfg else None,
    }


@router.post("/config/import-toml", response_model=m.ValidateResult, response_model_exclude_unset=True)
def import_toml(body: dict[str, str]) -> dict[str, Any]:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as f:
        f.write(body.get("toml", ""))
        path = f.name
    try:
        data = read_config_file(path)
    finally:
        Path(path).unlink(missing_ok=True)
    cfg, errors = _validate(data)
    return {"ok": cfg is not None, "errors": errors, "warnings": [], "config": cfg.to_dict() if cfg else data}


# --------------------------------------------------------------------------- models
class ModelBody(BaseModel):
    family: Literal["anima", "krea2", "sdxl", "flux", "flux2", "toy", "tagger"]
    kind: Literal["dit", "text_encoder", "text_encoder_2", "vae", "tokenizer", "tagger"]
    path: str
    dtype: str | None = None
    is_default: bool = False


def _model_row(r: dict[str, Any]) -> dict[str, Any]:
    p = Path(r["path"])
    return {**r, "exists": p.exists(), "is_default": bool(r["is_default"])}


@router.get("/models", response_model=list[m.ModelAsset], response_model_exclude_unset=True)
def list_models(c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    return [_model_row(r) for r in c.db.fetchall("SELECT * FROM models ORDER BY family, kind, created_at")]


@router.post("/models", response_model=m.ModelAsset, response_model_exclude_unset=True)
def add_model(body: ModelBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if (body.family == "tagger") != (body.kind == "tagger") or body.kind == "tagger" and body.is_default:
        raise ApiError(
            "tagger assets have their own family/role and cannot be training defaults", code="model.role"
        )
    p = Path(body.path).expanduser().resolve()
    if not p.exists():
        raise NotFound(f"model path not found: {p}", code="model.not_found")
    if not c.is_allowed(p):
        raise ApiError("model path is outside allowed storage roots", code="model.path", status=403)
    if body.kind in {"dit", "vae"} and not p.is_file() and body.family not in {"sdxl", "flux", "flux2"}:
        raise ApiError("DiT and VAE paths must point to a weight file", code="model.path")
    if body.kind == "tokenizer" and not p.is_dir():
        raise ApiError("tokenizer path must point to a complete tokenizer directory", code="model.path")
    if body.kind == "tagger" and not (
        p.is_dir() and (p / "model.onnx").is_file() and (p / "selected_tags.csv").is_file()
    ):
        raise ApiError("tagger directory needs model.onnx and selected_tags.csv together", code="model.path")
    size = (
        p.stat().st_size
        if p.is_file()
        else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir()
        else 0
    )
    with c.db.lock:
        existing = c.db.fetchone(
            "SELECT * FROM models WHERE family=? AND kind=? AND path=?", (body.family, body.kind, str(p))
        )
        mid = existing["id"] if existing else new_id("m")
        if body.is_default:
            c.db.execute("UPDATE models SET is_default=0 WHERE family=? AND kind=?", (body.family, body.kind))
        if existing:
            c.db.update(
                "models",
                mid,
                {
                    "size": size,
                    "dtype": body.dtype,
                    "is_default": int(body.is_default or existing["is_default"]),
                },
            )
        else:
            c.db.insert(
                "models",
                {
                    "id": mid,
                    "family": body.family,
                    "kind": body.kind,
                    "path": str(p),
                    "size": size,
                    "dtype": body.dtype,
                    "is_default": int(body.is_default),
                    "created_at": now(),
                },
            )
    return _model_row(c.db.fetchone("SELECT * FROM models WHERE id=?", (mid,)))


class ModelInspectionBody(BaseModel):
    path: str


class ModelInspection(BaseModel):
    path: str
    family: str | None
    family_candidates: list[str]
    kind: str | None
    dtype: str | None
    dtypes: dict[str, int]
    confidence: Literal["high", "partial", "unknown"]
    evidence: list[str]
    warnings: list[str]
    files_inspected: int


@router.post("/models/inspect", response_model=ModelInspection)
def inspect_local_model(body: ModelInspectionBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    from .model_inspection import inspect_model

    path = Path(body.path).expanduser().resolve()
    if not c.is_allowed(path):
        raise ApiError("model path is outside allowed storage roots", code="model.path", status=403)
    try:
        return inspect_model(path, allowed=c.is_allowed)
    except (ValueError, OSError, OverflowError) as error:
        raise ApiError(f"Cannot inspect model: {error}", code="model.inspect", status=422) from error


class ModelDefaultPatch(BaseModel):
    is_default: bool


@router.patch("/models/{model_id}", response_model=m.ModelAsset)
def patch_model(model_id: str, body: ModelDefaultPatch, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        row = c.db.fetchone("SELECT * FROM models WHERE id=?", (model_id,))
        if row is None:
            raise NotFound("model not found", code="model.not_found")
        if row["kind"] == "tagger" and body.is_default:
            raise ApiError("tagger assets cannot be training defaults", code="model.role")
        if body.is_default and not Path(row["path"]).exists():
            raise NotFound(
                "the model file is missing; register its new location first", code="model.not_found"
            )
        if body.is_default:
            c.db.execute(
                "UPDATE models SET is_default=0 WHERE family=? AND kind=?", (row["family"], row["kind"])
            )
        c.db.update("models", model_id, {"is_default": int(body.is_default)})
        return _model_row(c.db.fetchone("SELECT * FROM models WHERE id=?", (model_id,)))


@router.delete("/models/{model_id}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_model(model_id: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    c.db.delete("models", model_id)
    return {"ok": True}


class ScanBody(BaseModel):
    path: str | None = Field(
        None, description="directory to scan recursively for *.safetensors; default settings.paths.models_dir"
    )
    family: str = Field("anima", description="family assigned to newly registered files")


@router.post("/models/scan", response_model=list[m.ModelAsset], response_model_exclude_unset=True)
def scan_models(body: ScanBody, c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    root = Path(body.path or c.settings()["paths"]["models_dir"]).expanduser().resolve()
    if not c.is_allowed(root):
        raise ApiError("model path is outside allowed storage roots", code="model.path", status=403)
    if not root.is_dir():
        raise NotFound(f"directory not found: {root}", code="fs.not_found")
    from .model_inspection import inspect_model

    found = []
    known = {str(Path(r["path"]).resolve()) for r in c.db.fetchall("SELECT path FROM models")}

    def visible(path: Path) -> bool:
        return not any(part.startswith(".") for part in path.relative_to(root).parts)

    # A pipeline or sharded model is a single asset. Never fall back to registering
    # its internal files if its metadata is incomplete or incompatible.
    directory_markers = (
        "model_index.json",
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    )
    directories = {file.parent for name in directory_markers for file in root.rglob(name) if visible(file)}
    grouped = {
        folder
        for folder in directories
        if not any(folder != parent and folder.is_relative_to(parent) for parent in directories)
    }
    # A plain config may be a component sidecar or unrelated folder metadata.
    # Only suppress its direct weight files after successfully recognizing the
    # component; nested independent models are never owned by that plain config.
    configured = {
        file.parent
        for file in root.rglob("config.json")
        if visible(file) and not any(file.is_relative_to(folder) for folder in grouped)
    }
    recognized_components: set[Path] = set()
    candidates = sorted(grouped | configured, key=str) + [
        file
        for file in sorted(root.rglob("*.safetensors"))
        if visible(file) and not any(file.is_relative_to(folder) for folder in grouped)
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.parent in recognized_components:
            continue
        if str(candidate.resolve()) in known:
            if candidate in configured:
                recognized_components.add(candidate)
            continue
        try:
            detected = inspect_model(candidate, allowed=c.is_allowed)
        except (ValueError, OSError, OverflowError):
            continue
        kind = detected["kind"]
        family = detected["family"]
        if family is None and body.family in detected["family_candidates"]:
            family = body.family  # Explicit target for recognized shared components.
        if not kind or not family:
            continue  # Unknown assets need an explicit review in the local-file dialog.
        if candidate in configured:
            recognized_components.add(candidate)
        path = Path(detected["path"]).resolve()
        if str(path) in known:
            continue
        size = (
            path.stat().st_size
            if path.is_file()
            else sum(
                file.stat().st_size
                for file in path.rglob("*")
                if file.is_file() and file.resolve().is_relative_to(path) and c.is_allowed(file.resolve())
            )
        )
        mid = new_id("m")
        c.db.insert(
            "models",
            {
                "id": mid,
                "family": family,
                "kind": kind,
                "path": str(path),
                "size": size,
                "dtype": detected["dtype"],
                "is_default": 0,
                "created_at": now(),
            },
        )
        known.add(str(path))
        found.append(_model_row(c.db.fetchone("SELECT * FROM models WHERE id=?", (mid,))))
    return found


# --------------------------------------------------------------------------- events (SSE)
@router.get("/events")
async def events(
    request: Request, last_event_id: int | None = None, c: ServiceContext = Depends(ctx)
) -> StreamingResponse:
    header = request.headers.get("Last-Event-ID")
    after = int(header) if header and header.isdigit() else (last_event_id or 0)
    bus = c.bus
    queue = bus.subscribe()

    async def gen():
        try:
            for ev in bus.replay(after):
                yield bus.format_sse(ev)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield bus.format_sse(ev)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def stats_publisher(c: ServiceContext, interval: float = 2.5) -> None:
    while True:
        try:
            c.bus.publish("system.stats", await asyncio.to_thread(system_stats, c.data_root))
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(interval)


_ = time
