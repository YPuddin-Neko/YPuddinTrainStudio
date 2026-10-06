"""Version-owned regularization generation and collection, without credential reflection."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ypuddin.config import ModelConfig
from ypuddin.config.issues import plain_context

from .errors import ApiError


class SecretSafeRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                raise ApiError(
                    "Invalid regularization request",
                    status=422,
                    code="regularization.validation",
                    details={
                        "errors": [
                            {k: v for k, v in error.items() if k in {"loc", "msg", "type"}}
                            | ({"ctx": ctx} if (ctx := plain_context(error.get("ctx"))) else {})
                            for error in exc.errors()
                        ]
                    },
                ) from None

        return handler


router = APIRouter(route_class=SecretSafeRoute)


class RegularizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["ai", "danbooru", "gelbooru", "e621", "rule34"] = "ai"
    model: ModelConfig | None = None
    prompt: str = Field("", max_length=8000)
    prompt_source: Literal["manual", "training_tags"] = "manual"
    source_ids: list[str] = Field(default_factory=list, max_length=200)
    generation_scope: Literal["incremental", "all"] = "incremental"
    plan_signature: str | None = Field(None, min_length=64, max_length=64)
    negative: str = Field("", max_length=8000)
    count: int = Field(20, ge=1, le=200)
    width: int = Field(1024, ge=32, le=2048)
    height: int = Field(1024, ge=32, le=2048)
    steps: int = Field(25, ge=1, le=100)
    cfg: float = Field(4.0, ge=0, le=20, allow_inf_nan=False)
    seed: int = Field(0, ge=0, le=2**63 - 201)
    repeats: int = Field(1, ge=1, le=1000)
    prior_weight: float = Field(1.0, ge=0, le=100, allow_inf_nan=False)
    excluded_tags: list[str] = Field(default_factory=list, max_length=100)
    username: str = Field("", max_length=200, exclude=True)
    user_id: str = Field("", max_length=200, exclude=True)
    api_key: SecretStr = Field(default=SecretStr(""), max_length=1000, exclude=True)


class RegularizationTask(BaseModel):
    id: str
    project_id: str
    version_id: str
    source: str
    method: Literal["ai", "web"]
    provider: str | None
    status: Literal["queued", "running", "cancelling", "completed", "failed", "cancelled"]
    phase: str
    done: int
    total: int
    error: str | None
    logs: list[str]
    dataset_id: str | None
    path: str | None
    images: int
    duplicates: int
    created_at: float
    finished_at: float | None
    can_cancel: bool


class RegularizationStatus(BaseModel):
    path: str
    images: int
    operations: list[RegularizationTask]


class RegularizationPlanSource(BaseModel):
    id: str
    path: str
    name: str


class RegularizationTagCount(BaseModel):
    tag: str
    count: int


class RegularizationPlanExample(BaseModel):
    source_id: str
    rel_path: str
    prompt: str


class RegularizationPlan(BaseModel):
    signature: str
    sources: list[RegularizationPlanSource]
    top_tags: list[RegularizationTagCount]
    source_images: int
    existing_images: int
    missing_captions: int
    invalid_captions: int
    empty_after_exclusion: int
    eligible_images: int
    planned_images: int
    remaining_images: int
    max_batch_images: int
    examples: list[RegularizationPlanExample]


class RegularizationEstimate(BaseModel):
    source: str
    # Matching safe posts on the site; None when the site did not say.
    count: int | None
    # What is sent: search tags, as many exclusions as the account's tag limit allows, and the rating.
    terms: list[str]
    # Exclusions beyond the tag limit, checked on each post instead.
    local_exclusions: list[str]
    tag_limit: int | None


class RegularizationShareTag(BaseModel):
    tag: str
    share: float


class RegularizationRange(BaseModel):
    low: float
    median: float
    high: float


class RegularizationSize(BaseModel):
    width: int
    height: int


class RegularizationMatchPlan(BaseModel):
    source: str
    sources: list[RegularizationPlanSource]
    source_images: int
    captioned_images: int
    missing_captions: int
    invalid_captions: int
    top_tags: list[RegularizationTagCount]
    # The tags searched first, with the share of captioned training images that show them.
    search_tags: list[RegularizationShareTag]
    searchable_tags: int
    # Caption words no site searches for, such as quality words and @artist names.
    unsearchable_tags: list[str]
    aspect: RegularizationRange
    size: RegularizationSize
    existing_images: int
    suggested_count: int
    tag_limit: int | None
    tag_limit_known: bool


def manager(request: Request):
    return request.app.state.regularization


@router.get("/projects/{pid}/versions/{vid}/regularization", response_model=RegularizationStatus)
def status(pid: str, vid: str, service=Depends(manager)):
    return service.snapshot(pid, vid)


@router.post("/projects/{pid}/versions/{vid}/regularization/plan", response_model=RegularizationPlan)
def plan(pid: str, vid: str, body: RegularizationRequest, service=Depends(manager)):
    return service.plan(pid, vid, body)


@router.post("/projects/{pid}/versions/{vid}/regularization/estimate", response_model=RegularizationEstimate)
def estimate(pid: str, vid: str, body: RegularizationRequest, service=Depends(manager)):
    return service.estimate(pid, vid, body)


@router.post("/projects/{pid}/versions/{vid}/regularization/match", response_model=RegularizationMatchPlan)
def match(pid: str, vid: str, body: RegularizationRequest, service=Depends(manager)):
    return service.match_plan(pid, vid, body)


@router.post(
    "/projects/{pid}/versions/{vid}/regularization", status_code=202, response_model=RegularizationTask
)
def start(pid: str, vid: str, body: RegularizationRequest, service=Depends(manager)):
    return service.start(pid, vid, body)


@router.get("/regularization/{oid}", response_model=RegularizationTask)
def operation(oid: str, service=Depends(manager)):
    return service.get(oid)


@router.post("/regularization/{oid}/cancel", response_model=RegularizationTask)
def cancel(oid: str, service=Depends(manager)):
    return service.cancel(oid)


@router.post("/regularization/{oid}/release", response_model=RegularizationTask)
def release(oid: str, service=Depends(manager)):
    return service.release(oid)
