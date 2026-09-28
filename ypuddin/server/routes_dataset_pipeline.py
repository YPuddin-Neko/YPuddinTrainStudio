"""Dataset pipeline operations scoped to one project version."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, model_validator

from .dataset_pipeline import DatasetPipeline
from .model_credentials import VlmProvider

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
    model: str = "wd-eva02-large-tagger-v3"
    general_threshold: float = Field(0.35, ge=0.01, le=0.99)
    character_threshold: float = Field(0.85, ge=0.01, le=0.99)
    exclude_tags: list[str] = Field(default_factory=list, max_length=1000)
    existing: Literal["skip", "overwrite", "append", "prepend"] = "skip"
    trigger_word: str | None = Field(None, max_length=200)
    device: Literal["auto", "cpu"] = "auto"


class HeadSelection(BaseModel):
    """The detected heads of one image to write, by their index in the detection."""

    dataset_id: str
    rel_path: str = Field(min_length=1)
    regions: list[int] = Field(min_length=1, max_length=256)


class AutoMaskOptions(BaseModel):
    model: str = "anime-head-detector-v2"
    confidence: float = Field(0.413, ge=0.01, le=0.99)
    padding: float = Field(0.10, ge=0, le=1)
    feather: float = Field(0.03, ge=0, le=0.5)
    device: Literal["auto", "cpu"] = "auto"
    # Writes the chosen heads of a finished detection instead of detecting again.
    proposal_id: str | None = None
    selections: list[HeadSelection] = Field(default_factory=list, max_length=20000)


class VlmOptions(BaseModel):
    provider: VlmProvider
    base_url: str | None = Field(None, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=20000)
    # What the reply holds and where it goes: a tag list, schema groups (JSON), the file's own tags
    # regrouped without adding or removing any (JSON), or prose.
    output: Literal["tags", "categories", "sort", "description"] = "tags"
    # "refine" sends each image's own caption as the reference instead of tagger output.
    existing: Literal["skip", "overwrite", "refine"] = "skip"
    trigger_word: str | None = Field(None, max_length=200)
    exclude_tags: list[str] = Field(default_factory=list, max_length=1000)
    temperature: float = Field(0.3, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=16, le=65536)
    image_size: int = Field(1024, ge=256, le=4096)
    image_detail: Literal["", "auto", "low", "high"] = ""
    concurrency: int = Field(2, ge=1, le=16)
    interval: float = Field(0, ge=0, le=120)
    timeout: int = Field(120, ge=10, le=900)
    retries: int = Field(2, ge=0, le=5)


VISION_ACTIONS = {"autotag", "automask", "detectheads", "vlmtag", "assisttag"}


class PipelineRequest(BaseModel):
    action: Literal[
        "inspect",
        "exclude",
        "restore",
        "preprocess",
        "captions",
        "prepare",
        "autotag",
        "automask",
        "detectheads",
        "vlmtag",
        "assisttag",
    ]
    images: list[PipelineImage] = Field(default_factory=list, max_length=20000)
    # Automatic tagging and masks may name whole datasets instead of images.
    dataset_ids: list[str] = Field(default_factory=list, max_length=200)
    restore_operation_id: str | None = None
    preprocess: PreprocessOptions | None = None
    captions: CaptionOptions | None = None
    tagging: TaggingOptions | None = None
    automask: AutoMaskOptions | None = None
    vlm: VlmOptions | None = None

    @model_validator(mode="after")
    def validate_action(self) -> PipelineRequest:
        if self.action in {"exclude", "preprocess", "captions"} and not self.images:
            raise ValueError("select at least one image")
        reviewed = self.action == "automask" and self.automask is not None and self.automask.proposal_id
        if reviewed:
            if self.images or self.dataset_ids:
                raise ValueError("reviewed heads name their own images")
            if not self.automask.selections:
                raise ValueError("choose at least one head to write")
        elif self.automask is not None and self.automask.selections:
            raise ValueError("head selections belong to a detection; pass its proposal_id")
        if self.action in VISION_ACTIONS and not reviewed and not (self.images or self.dataset_ids):
            raise ValueError("select at least one image or dataset")
        if self.dataset_ids and self.action not in VISION_ACTIONS:
            raise ValueError("only automatic tagging and masks accept whole datasets")
        if self.action in {"autotag", "assisttag"} and not self.tagging:
            raise ValueError("tagging options are required")
        if self.action in {"vlmtag", "assisttag"} and not self.vlm:
            raise ValueError("vision model options are required")
        if (
            self.action == "vlmtag"
            and self.vlm
            and (self.vlm.output == "sort" or self.vlm.existing == "refine")
        ):
            raise ValueError("regrouping and refining need reference tags; use assisted tagging")
        if self.action in {"automask", "detectheads"} and not self.automask:
            raise ValueError("automatic mask options are required")
        if self.action == "preprocess" and not self.preprocess:
            raise ValueError("preprocess options are required")
        if self.preprocess and self.preprocess.mode == "crop_rect":
            if not self.preprocess.crop or len(self.images) != 1:
                raise ValueError("visual cropping requires one image and a source pixel rectangle")
        if self.action == "captions" and not self.captions:
            raise ValueError("caption options are required")
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


class HeadRegion(BaseModel):
    index: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    feather_x: int
    feather_y: int


class HeadProposalImage(BaseModel):
    dataset_id: str
    rel_path: str
    hash: str | None = None
    width: int | None = None
    height: int | None = None
    regions: list[HeadRegion]
    error: str | None = None


class HeadProposals(BaseModel):
    operation_id: str
    parameters: dict[str, Any]
    images: list[HeadProposalImage]
    applied_by: str | None = None
    dismissed: bool = False


@router.get("/dataset-pipeline/operations/{oid}/proposals", response_model=HeadProposals)
def get_head_proposals(oid: str, manager: DatasetPipeline = Depends(pipeline)) -> dict:
    return manager.proposals(oid)


@router.post("/dataset-pipeline/operations/{oid}/dismiss", response_model=PipelineOperation)
def dismiss_head_proposals(oid: str, manager: DatasetPipeline = Depends(pipeline)) -> dict:
    return manager.dismiss(oid)


@router.post("/dataset-pipeline/operations/{oid}/retry", response_model=PipelineOperation, status_code=202)
def retry_operation(oid: str, manager: DatasetPipeline = Depends(pipeline)) -> dict:
    return manager.retry(oid)
