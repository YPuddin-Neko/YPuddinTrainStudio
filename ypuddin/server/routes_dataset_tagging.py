"""Read-only local tagger diagnostics. Inference is a version pipeline operation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .context import ServiceContext
from .dataset_tagging import INPUT_SIZE, local_models, read_labels, runtime_info
from .routes_work import ctx

router = APIRouter()


class LocalTaggingModel(BaseModel):
    name: str
    path: str
    model_path: str
    tags_path: str


class TaggingStatus(BaseModel):
    available: bool
    runtime_available: bool
    runtime_version: str | None
    runtime_providers: list[str]
    providers: list[str]
    provider: Literal["cpu", "cuda"]
    model_exists: bool
    tags_exists: bool
    model_path: str | None
    tags_path: str | None
    input_size: int
    models: list[LocalTaggingModel]
    recommended_model_dir: str
    errors: list[str]
    notes: list[str]


@router.get("/dataset-tagging/status", response_model=TaggingStatus)
def tagging_status(
    model_path: str | None = None,
    tags_path: str | None = None,
    provider: Literal["cpu", "cuda"] = "cpu",
    c: ServiceContext = Depends(ctx),
) -> dict:
    root = Path(c.settings()["paths"]["models_dir"])
    models = local_models(root)
    if not model_path and models:
        model_path = models[0]["model_path"]
    if model_path:
        model = Path(model_path).expanduser().resolve()
        if model.is_dir():
            model = model / "model.onnx"
        tags = Path(tags_path).expanduser().resolve() if tags_path else model.with_name("selected_tags.csv")
    else:
        model = None
        tags = Path(tags_path).expanduser().resolve() if tags_path else None
    info = runtime_info()
    errors = []
    if not info["runtime_available"]:
        errors.append(info["runtime_error"])
    elif provider not in info["providers"]:
        errors.append(
            f"{provider} is unavailable in the installed ONNX Runtime; select CPU or install a compatible GPU runtime"
        )
    model_exists = bool(model and model.is_file() and model.suffix.lower() == ".onnx" and c.is_allowed(model))
    tags_exists = bool(tags and tags.is_file() and c.is_allowed(tags))
    if not model_exists:
        errors.append("Select a local WD14 model.onnx file")
    if not tags_exists:
        errors.append("Select the matching selected_tags.csv file")
    elif tags:
        try:
            read_labels(tags)
        except Exception as exc:
            errors.append(str(exc))
    return {
        "available": not errors,
        **{
            key: info[key]
            for key in ("runtime_available", "runtime_version", "runtime_providers", "providers")
        },
        "provider": provider,
        "model_exists": model_exists,
        "tags_exists": tags_exists,
        "model_path": str(model) if model else None,
        "tags_path": str(tags) if tags else None,
        "input_size": INPUT_SIZE,
        "models": models,
        "recommended_model_dir": str(root / "tagger"),
        "errors": errors,
        "notes": [
            "All inference runs locally; image files are never sent to an external service.",
            "Model input dimensions, label count and GPU initialization are validated by the worker when tagging starts.",
            "CPU inference is supported on Windows, Linux and macOS. CUDA requires an explicitly installed compatible ONNX Runtime GPU package.",
        ],
    }
