"""Training images from Danbooru and Gelbooru: suggestions, match counts and background downloads."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from .routes_regularization import SecretSafeRoute

router = APIRouter(route_class=SecretSafeRoute)

Rating = Literal["general", "sensitive", "questionable", "explicit"]
SiteName = Literal["danbooru", "gelbooru", "e621", "rule34"]


class SiteDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: SiteName = "danbooru"
    # Site tags separated by spaces; ratings, sort order and score have their own fields.
    tags: str = Field("", max_length=2000)
    excluded_tags: list[str] = Field(default_factory=list, max_length=100)
    count: int = Field(100, ge=1, le=1000)
    ratings: list[Rating] = Field(default_factory=lambda: ["general"], min_length=1, max_length=4)
    order: Literal["score", "newest"] = "score"
    min_score: int | None = Field(None, ge=-100_000, le=100_000)
    # The shorter side a picture needs, in pixels.
    min_side: int = Field(512, ge=0, le=8192)
    # Add to this dataset of the version instead of creating one.
    dataset_id: str | None = Field(None, max_length=100)
    name: str = Field("", max_length=100)
    repeats: int = Field(1, ge=1, le=1_000_000)
    caption_ext: str = Field("auto", max_length=32)


class SiteDownloadTask(BaseModel):
    id: str
    project_id: str
    version_id: str
    source: str
    query: str
    target_dataset_id: str | None
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


class SiteDownloadStatus(BaseModel):
    operations: list[SiteDownloadTask]


class SiteDownloadEstimate(BaseModel):
    source: str
    # Matching posts on the site; None when the site did not say.
    count: int | None
    # What is sent: the tags, the sort order when it fits, as many exclusions as fit, and the conditions.
    terms: list[str]
    # Exclusions beyond the tag limit, checked on each post instead.
    local_exclusions: list[str]
    tag_limit: int | None
    # False when sorting by score did not fit the account's tag limit.
    sorted: bool


class SiteTagSuggestion(BaseModel):
    tag: str
    category: str
    posts: int | None
    alias: str | None


def manager(request: Request):
    return request.app.state.site_downloads


@router.get("/projects/{pid}/versions/{vid}/site-downloads", response_model=SiteDownloadStatus)
def status(pid: str, vid: str, service=Depends(manager)):
    return service.snapshot(pid, vid)


@router.post("/projects/{pid}/versions/{vid}/site-downloads/estimate", response_model=SiteDownloadEstimate)
def estimate(pid: str, vid: str, body: SiteDownloadRequest, service=Depends(manager)):
    return service.estimate(pid, vid, body)


@router.post(
    "/projects/{pid}/versions/{vid}/site-downloads", status_code=202, response_model=SiteDownloadTask
)
def start(pid: str, vid: str, body: SiteDownloadRequest, service=Depends(manager)):
    return service.start(pid, vid, body)


@router.get("/site-downloads/suggestions", response_model=list[SiteTagSuggestion])
def suggestions(source: SiteName = "danbooru", q: str = Query("", max_length=200), service=Depends(manager)):
    return service.suggest(source, q)


@router.get("/site-downloads/{oid}", response_model=SiteDownloadTask)
def operation(oid: str, service=Depends(manager)):
    return service.get(oid)


@router.post("/site-downloads/{oid}/cancel", response_model=SiteDownloadTask)
def cancel(oid: str, service=Depends(manager)):
    return service.cancel(oid)
