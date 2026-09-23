"""Recommended model catalog and download/registration endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from .context import ServiceContext
from .errors import ApiError, NotFound
from .model_credentials import Provider
from .model_downloads import ModelDownload, ModelDownloadRequest, ModelDownloads
from .model_recommendations import RecommendedModel, available_models, find_recommendation, verify_local_model
from .models import ModelAsset
from .routes_core import ctx
from .routes_model_downloads import downloads

router = APIRouter()


class RecommendationDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Provider = "huggingface"
    is_default: bool = True


@router.get("/models/recommendations", response_model=list[RecommendedModel])
def recommendations(context: ServiceContext = Depends(ctx)):
    return available_models(context)


@router.post("/models/recommendations/{model_id}/download", response_model=ModelDownload, status_code=202)
def download_recommendation(
    model_id: str, body: RecommendationDownloadRequest, service: ModelDownloads = Depends(downloads)
):
    entry = find_recommendation(model_id)
    if entry is None:
        raise NotFound("recommended model not found", code="model.recommendation")
    source = next((source for source in entry.sources if source.provider == body.provider), None)
    if source is None:
        raise ApiError("this model is not available from the selected provider", code="model.source")
    return service.start(
        ModelDownloadRequest(
            family=entry.family,
            kind=entry.kind,
            dtype=entry.dtype,
            provider=body.provider,
            repo_id=source.repo_id,
            filename=source.filename,
            revision=source.revision,
            is_default=body.is_default,
            purpose=entry.purpose,
            variant=entry.variant,
        ),
        recommendation=entry,
    )


@router.post("/models/recommendations/{model_id}/use", response_model=ModelAsset)
def use_recommendation(model_id: str, context: ServiceContext = Depends(ctx)):
    entry = next((item for item in available_models(context) if item.id == model_id), None)
    if entry is None or not entry.available_path:
        raise NotFound("download or register this model first", code="model.not_found")
    from .routes_core import ModelBody, add_model

    verify_local_model(context, entry, Path(entry.available_path))
    return add_model(
        ModelBody(
            family=entry.family,
            kind=entry.kind,
            path=entry.available_path,
            dtype=entry.dtype,
            is_default=entry.purpose == "training",
            purpose=entry.purpose,
            variant=entry.variant,
        ),
        context,
    )
