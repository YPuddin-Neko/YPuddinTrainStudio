"""Tagging and mask-detection models: the runtime they need, downloads and removal."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict

from .vision_downloads import VisionModels

router = APIRouter()


def vision(request: Request) -> VisionModels:
    return request.app.state.vision_models


class VisionRuntime(BaseModel):
    available: bool
    package: str | None = None
    version: str | None = None
    providers: list[str]
    error: str | None = None


class VisionDownload(BaseModel):
    status: str | None = None
    source: str | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    bytes_per_second: float | None = None
    error: str | None = None


class VisionThresholds(BaseModel):
    general: float
    character: float


class VisionModel(BaseModel):
    id: str
    role: Literal["tagger", "head_detector"]
    # Tagger series (wd, pixai, cl) or the kind of region a mask detector finds.
    family: str
    label: str
    repo: str
    revision: str
    license: str
    size: int
    recommended: bool
    # Label categories a tagger can write, and the thresholds its authors recommend.
    categories: list[str] = []
    thresholds: VisionThresholds | None = None
    # Gated on Hugging Face: downloads need a saved access token whose account accepted the terms.
    token_required: bool = False
    token_configured: bool = False
    sources: list[Literal["huggingface", "modelscope"]]
    ready: bool
    path: str
    download: VisionDownload | None = None


class VisionCatalog(BaseModel):
    runtime: VisionRuntime
    models: list[VisionModel]


class VisionDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["huggingface", "modelscope"] = "huggingface"


@router.get("/vision/models", response_model=VisionCatalog)
async def vision_models(service: VisionModels = Depends(vision)) -> dict[str, Any]:
    from .vision_models import runtime_status

    return {"runtime": await run_in_threadpool(runtime_status), "models": service.catalog()}


@router.post("/vision/models/{model_id}/download", response_model=VisionModel, status_code=202)
def download_vision_model(
    model_id: str, body: VisionDownloadRequest, service: VisionModels = Depends(vision)
) -> dict[str, Any]:
    return service.start(model_id, body.source)


@router.post("/vision/models/{model_id}/cancel", response_model=VisionModel)
def cancel_vision_model(model_id: str, service: VisionModels = Depends(vision)) -> dict[str, Any]:
    return service.cancel(model_id)


@router.delete("/vision/models/{model_id}", response_model=VisionModel)
def remove_vision_model(model_id: str, service: VisionModels = Depends(vision)) -> dict[str, Any]:
    return service.remove(model_id)
