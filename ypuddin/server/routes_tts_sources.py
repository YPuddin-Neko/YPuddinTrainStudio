"""Version-owned source registration, complete scan results and recording previews."""

from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from starlette.responses import StreamingResponse

from ypuddin.tts.source_models import (
    SourceSplit,
    TtsRowsResponse,
    TtsSource,
    TtsSourceCheckBody,
    TtsSourcePutBody,
    TtsSourcesResponse,
)

from .context import ServiceContext
from .routes_tts_projects import ERRORS, TtsProjectRoute, ctx
from .tts_audio_response import audio_response

router = APIRouter(route_class=TtsProjectRoute)
BASE = "/tts/projects/{pid}/versions/{vid}/sources"


@router.get(BASE, response_model=TtsSourcesResponse, responses=ERRORS)
def list_sources(pid: str, vid: str, c: ServiceContext = Depends(ctx)):
    return c.tts_sources.list(pid, vid)


@router.put(BASE + "/{split}", response_model=TtsSourcesResponse, responses=ERRORS)
def put_source(
    pid: str, vid: str, split: SourceSplit, body: TtsSourcePutBody, c: ServiceContext = Depends(ctx)
):
    return c.tts_sources.put(
        pid, vid, split, expected_data_revision=body.expected_data_revision, path=body.path
    )


@router.delete(BASE + "/{split}", response_model=TtsSourcesResponse, responses=ERRORS)
def remove_source(
    pid: str,
    vid: str,
    split: SourceSplit,
    expected_data_revision: int = Query(ge=1),
    c: ServiceContext = Depends(ctx),
):
    return c.tts_sources.remove(pid, vid, split, expected_data_revision=expected_data_revision)


@router.get(BASE + "/{source_id}", response_model=TtsSource, responses=ERRORS)
def read_source(pid: str, vid: str, source_id: str, c: ServiceContext = Depends(ctx)):
    return c.tts_sources.get(pid, vid, source_id)


@router.post(BASE + "/{source_id}/check", status_code=202, response_model=TtsSource, responses=ERRORS)
def check_source(
    pid: str, vid: str, source_id: str, body: TtsSourceCheckBody, c: ServiceContext = Depends(ctx)
):
    return c.tts_sources.check(pid, vid, source_id, expected_data_revision=body.expected_data_revision)


@router.get(BASE + "/{source_id}/rows", response_model=TtsRowsResponse, responses=ERRORS)
def source_rows(
    pid: str,
    vid: str,
    source_id: str,
    snapshot_id: str = Query(min_length=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    c: ServiceContext = Depends(ctx),
):
    return c.tts_sources.rows(pid, vid, source_id, snapshot_id=snapshot_id, page=page, page_size=page_size)


@router.get(
    BASE + "/{source_id}/rows/{row_id}/audio",
    response_class=StreamingResponse,
    responses={
        **ERRORS,
        200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}},
        206: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}},
    },
)
def source_audio(
    pid: str,
    vid: str,
    source_id: str,
    row_id: str,
    request: Request,
    snapshot_id: str = Query(min_length=1),
    role: Literal["audio", "ref_audio"] = "audio",
    c: ServiceContext = Depends(ctx),
):
    asset = c.tts_sources.audio_asset(pid, vid, source_id, row_id, snapshot_id=snapshot_id, role=role)
    return audio_response(request, asset)
