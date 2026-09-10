"""System / settings / fs / schema / plan / presets / models / events endpoints."""

from __future__ import annotations

import asyncio
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import ypuddin
from ypuddin.config import TrainConfig, deep_merge, dump_toml, read_config_file
from ypuddin.models import available as available_families
from ypuddin.train.plan import plan as make_plan

from . import models as m
from .context import ServiceContext
from .db import new_id, now
from .errors import ApiError, NotFound

router = APIRouter()


def ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


# --------------------------------------------------------------------------- system
def gpu_info() -> list[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            entry: dict[str, Any] = {
                "index": i,
                "name": props.name,
                "mem_total_mb": round(props.total_memory / 2**20),
            }
            try:
                free, total = torch.cuda.mem_get_info(i)
                entry["mem_used_mb"] = round((total - free) / 2**20)
            except Exception:  # noqa: BLE001
                pass
            out.append(entry)
        try:
            import pynvml

            pynvml.nvmlInit()
            for e in out:
                h = pynvml.nvmlDeviceGetHandleByIndex(e["index"])
                e["util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                e["temp_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception:  # noqa: BLE001
        return []


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
        "ram": {"used_mb": round(vm.used / 2**20), "total_mb": round(vm.total / 2**20)},
        "disks": disks,
        "gpus": gpu_info(),
    }


@router.get("/health", response_model=m.Health, response_model_exclude_unset=True)
def health(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    try:
        import torch

        torch_v, cuda = torch.__version__, torch.version.cuda
    except Exception:  # noqa: BLE001
        torch_v, cuda = None, None
    return {
        "version": ypuddin.__version__,
        "api_version": ypuddin.API_VERSION,
        "torch": torch_v,
        "cuda": cuda,
        "gpus": [{"index": g["index"], "name": g["name"], "total_mb": g["mem_total_mb"]} for g in gpu_info()],
        "families": available_families(),
    }


@router.get("/system/stats", response_model=m.SystemStats, response_model_exclude_unset=True)
def stats(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return system_stats(c.data_root)


@router.get("/system/info", response_model=m.SystemInfo, response_model_exclude_unset=True)
def info() -> dict[str, Any]:
    mods = {}
    for name in ("torch", "transformers", "safetensors", "pydantic", "fastapi"):
        try:
            mods[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            mods[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": mods,
        "ypuddin": ypuddin.__version__,
    }


# --------------------------------------------------------------------------- settings / fs
@router.get("/settings", response_model=m.Settings, response_model_exclude_unset=True)
def get_settings(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return c.settings()


@router.put("/settings", response_model=m.Settings, response_model_exclude_unset=True)
def put_settings(patch: dict[str, Any], c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return c.save_settings(patch)


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


class ConfigBody(BaseModel):
    config: dict[str, Any]
    dataset_ids: list[str] | None = None


def _validate(config: dict[str, Any]) -> tuple[TrainConfig | None, list[dict[str, str]]]:
    from pydantic import ValidationError

    try:
        return TrainConfig.model_validate(config), []
    except ValidationError as e:
        return None, [{"loc": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]} for err in e.errors()]


@router.post("/config/validate", response_model=m.ValidateResult, response_model_exclude_unset=True)
def config_validate(body: ConfigBody) -> dict[str, Any]:
    cfg, errors = _validate(body.config)
    return {"ok": cfg is not None, "errors": errors, "warnings": []}


@router.post("/plan", response_model=m.Plan, response_model_exclude_unset=True)
def config_plan(body: ConfigBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    cfg, errors = _validate(body.config)
    if cfg is None:
        return {"ok": False, "errors": errors, "warnings": []}
    gpus = gpu_info()
    return make_plan(cfg, gpu_total_mb=gpus[0]["mem_total_mb"] if gpus else None)


# --------------------------------------------------------------------------- presets
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "anima-lokr-default": {
        "description": "Anima LoKr：全矩阵 W2、factor 8、注意力+MLP",
        "config": {
            "model": {"family": "anima"},
            "adapter": {"algo": "lokr", "rank": "full", "alpha": 1.0, "factor": 8, "preset": "attn-mlp"},
            "optimizer": {"type": "adamw", "lr": 1e-4},
            "scheduler": {"type": "cosine", "warmup_steps": 0.05},
        },
    },
    "anima-lora-16": {
        "description": "Anima LoRA rank 16 / alpha 16",
        "config": {
            "model": {"family": "anima"},
            "adapter": {"algo": "lora", "rank": 16, "alpha": 16.0, "preset": "attn-mlp"},
            "optimizer": {"type": "adamw", "lr": 2e-4},
        },
    },
    "toy-smoke": {
        "description": "CPU 玩具模型冒烟测试",
        "config": {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {"resolutions": [64], "bucket_step": 16, "batch_size": 2, "num_workers": 0},
            "loop": {"epochs": 1, "mixed_precision": "no"},
        },
    },
}


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
    out = [_preset_row(n, d, True, None) for n, d in BUILTIN_PRESETS.items()]
    for f in sorted(_preset_dir(c).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        out.append(_preset_row(f.stem, data, False, f.stat().st_mtime))
    return out


class PresetBody(BaseModel):
    name: str
    description: str = ""
    config: dict[str, Any]


@router.post("/presets", response_model=m.Preset, response_model_exclude_unset=True)
def create_preset(body: PresetBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if not body.name.replace("-", "").replace("_", "").isalnum():
        raise ApiError("preset name must be alphanumeric with - or _", code="preset.bad_name")
    f = _preset_dir(c) / f"{body.name}.json"
    f.write_text(
        json.dumps({"description": body.description, "config": body.config}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return _preset_row(
        body.name, {"description": body.description, "config": body.config}, False, f.stat().st_mtime
    )


@router.get("/presets/{name}", response_model=m.Preset, response_model_exclude_unset=True)
def get_preset(name: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if name in BUILTIN_PRESETS:
        return _preset_row(name, BUILTIN_PRESETS[name], True, None)
    f = _preset_dir(c) / f"{name}.json"
    if not f.exists():
        raise NotFound(f"preset {name} not found", code="preset.not_found")
    return _preset_row(name, json.loads(f.read_text(encoding="utf-8")), False, f.stat().st_mtime)


@router.put("/presets/{name}", response_model=m.Preset, response_model_exclude_unset=True)
def put_preset(name: str, body: PresetBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if name in BUILTIN_PRESETS:
        raise ApiError("builtin presets are read-only", code="preset.readonly", status=403)
    return create_preset(PresetBody(name=name, description=body.description, config=body.config), c)


@router.delete("/presets/{name}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_preset(name: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if name in BUILTIN_PRESETS:
        raise ApiError("builtin presets are read-only", code="preset.readonly", status=403)
    f = _preset_dir(c) / f"{name}.json"
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
    family: str
    kind: str  # dit | text_encoder | vae | tokenizer
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
    p = Path(body.path).expanduser()
    size = (
        p.stat().st_size
        if p.is_file()
        else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir()
        else 0
    )
    mid = new_id("m")
    if body.is_default:
        c.db.execute("UPDATE models SET is_default=0 WHERE family=? AND kind=?", (body.family, body.kind))
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


@router.delete("/models/{model_id}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_model(model_id: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    c.db.delete("models", model_id)
    return {"ok": True}


@router.post("/models/scan", response_model=list[m.ModelAsset], response_model_exclude_unset=True)
def scan_models(body: dict[str, str], c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    root = Path(body.get("path", str(c.data_root / "models"))).expanduser()
    if not root.is_dir():
        raise NotFound(f"directory not found: {root}", code="fs.not_found")
    found = []
    known = {r["path"] for r in c.db.fetchall("SELECT path FROM models")}
    for f in sorted(root.rglob("*.safetensors")):
        if str(f) in known:
            continue
        name = f.name.lower()
        kind = (
            "vae"
            if "vae" in name
            else "text_encoder"
            if any(k in name for k in ("qwen", "t5", "clip", "text"))
            else "dit"
        )
        family = body.get("family", "anima")
        mid = new_id("m")
        c.db.insert(
            "models",
            {
                "id": mid,
                "family": family,
                "kind": kind,
                "path": str(f),
                "size": f.stat().st_size,
                "dtype": "fp8" if "fp8" in name else None,
                "is_default": 0,
                "created_at": now(),
            },
        )
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
            c.bus.publish("system.stats", system_stats(c.data_root))
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(interval)


_ = time
