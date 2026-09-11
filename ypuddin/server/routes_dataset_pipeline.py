"""Actual dataset pipeline operations scoped to one project version."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, model_validator

from .dataset_pipeline import DatasetPipeline

router = APIRouter()


def pipeline(request: Request) -> DatasetPipeline:
    return request.app.state.dataset_pipeline


class PipelineImage(BaseModel):
    dataset_id: str
    rel_path: str = Field(min_length=1)


class CropRectangle(BaseModel):
    x: int = Field(ge=0, strict=True)
    y: int = Field(ge=0, strict=True)
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)


class PreprocessOptions(BaseModel):
    mode: Literal["resize", "center_crop", "crop_rect"] = "resize"
    width: int = Field(1024, ge=16, le=8192)
    height: int = Field(1024, ge=16, le=8192)
    allow_upscale: bool = False
    crop: CropRectangle | None = None


class CaptionOptions(BaseModel):
    mode: Literal["fill_missing", "append", "remove", "replace"] = "fill_missing"
    text: str = Field(max_length=32000)


class TaggingOptions(BaseModel):
    model_path: str = Field(min_length=1)
    tags_path: str = Field(min_length=1)
    general_threshold: float = Field(0.35, ge=0, le=1)
    character_threshold: float = Field(0.85, ge=0, le=1)
    provider: Literal["cpu", "cuda"] = "cpu"
    mode: Literal["missing", "append", "overwrite"] = "missing"
    trigger_word: str = Field("", max_length=1000)


class PipelineRequest(BaseModel):
    action: Literal["inspect", "exclude", "restore", "preprocess", "captions", "tag", "prepare"]
    images: list[PipelineImage] = Field(default_factory=list, max_length=20000)
    restore_operation_id: str | None = None
    preprocess: PreprocessOptions | None = None
    captions: CaptionOptions | None = None
    tagging: TaggingOptions | None = None

    @model_validator(mode="after")
    def validate_action(self) -> PipelineRequest:
        if self.action in {"exclude", "preprocess", "captions", "tag"} and not self.images:
            raise ValueError("select at least one image")
        if self.action == "preprocess" and not self.preprocess:
            raise ValueError("preprocess options are required")
        if self.preprocess and self.preprocess.mode == "crop_rect":
            if not self.preprocess.crop or len(self.images) != 1:
                raise ValueError("visual cropping requires one image and a source pixel rectangle")
        if self.action == "captions" and not self.captions:
            raise ValueError("caption options are required")
        if self.action == "tag" and not self.tagging:
            raise ValueError("tagging model and options are required")
        if self.action == "restore" and not self.restore_operation_id:
            raise ValueError("restore_operation_id is required")
        return self


class PipelineOperation(BaseModel):
    id: str
    project_id: str
    version_id: str
    action: str
    status: str
    phase: str
    done: int
    total: int
    error: str | None = None
    job_id: str | None = None
    created_at: float
    updated_at: float
    finished_at: float | None = None
    request: dict[str, Any]
    result: dict[str, Any]
    logs: list[dict[str, Any]]
    can_undo: bool
    can_cancel: bool


class PipelineSnapshot(BaseModel):
    project_id: str
    version_id: str
    signature: str
    inspection: dict[str, Any] | None
    plan: dict[str, Any] | None
    operations: list[PipelineOperation]
    datasets: list[dict[str, Any]]
    busy: bool
    archived: bool
    ready_to_train: bool
    prepared_job_id: str | None
    stale: bool


@router.get("/projects/{pid}/versions/{vid}/pipeline", response_model=PipelineSnapshot)
def get_pipeline(pid: str, vid: str, manager: DatasetPipeline = Depends(pipeline)) -> dict:
    return manager.snapshot(pid, vid)


@router.post(
    "/projects/{pid}/versions/{vid}/pipeline/operations", response_model=PipelineOperation, status_code=202
)
def start_operation(
    pid: str, vid: str, body: PipelineRequest, manager: DatasetPipeline = Depends(pipeline)
) -> dict:
    return manager.start(pid, vid, body.model_dump())


@router.get("/dataset-pipeline/operations/{oid}", response_model=PipelineOperation)
def get_operation(oid: str, manager: DatasetPipeline = Depends(pipeline)) -> dict:
    return manager.operation(oid)


@router.post("/dataset-pipeline/operations/{oid}/cancel", response_model=PipelineOperation)
def cancel_operation(oid: str, manager: DatasetPipeline = Depends(pipeline)) -> dict:
    return manager.cancel(oid)


@router.post("/dataset-pipeline/operations/{oid}/retry", response_model=PipelineOperation, status_code=202)
def retry_operation(oid: str, manager: DatasetPipeline = Depends(pipeline)) -> dict:
    return manager.retry(oid)
