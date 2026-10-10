"""Registered speech checkpoints, listening jobs and their audio."""

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from starlette.responses import StreamingResponse

from ypuddin.tts.result_models import (
    TtsAudio,
    TtsCheckpoint,
    TtsCheckpointPage,
    TtsSampleJob,
    TtsSampleJobPage,
)

from . import models as m
from . import tts_results
from .context import ServiceContext
from .routes_tts_projects import ERRORS, TtsProjectRoute, ctx
from .tts_audio_response import audio_response
from .tts_sample_jobs import TtsSampleBody, create_sample

router = APIRouter(route_class=TtsProjectRoute)


def _binary_responses(media_type: str) -> dict:
    headers = {name: {"schema": {"type": "string"}} for name in (
        "Accept-Ranges", "ETag", "Cache-Control", "Content-Disposition",
    )}
    headers["Content-Length"] = {"schema": {"type": "integer"}}
    content = {media_type: {"schema": {"type": "string", "format": "binary"}}}
    content_range = {"Content-Range": {"schema": {"type": "string"}}}
    return {
        **ERRORS,
        200: {"content": content, "headers": headers},
        206: {"description": "Partial Content", "content": content, "headers": {**headers, **content_range}},
        416: {**ERRORS[416], "headers": content_range},
    }


@router.get("/tts/jobs/{jid}/checkpoints", response_model=list[TtsCheckpoint], responses=ERRORS)
def checkpoints(jid: str, c: ServiceContext = Depends(ctx)):
    return tts_results.checkpoints(c, jid)


@router.get("/tts/projects/{pid}/versions/{vid}/checkpoints", response_model=TtsCheckpointPage, responses=ERRORS)
def version_checkpoints(pid: str, vid: str, cursor: str | None = None, limit: int = Query(50, ge=1, le=200), c: ServiceContext = Depends(ctx)):
    return tts_results.version_checkpoints(c, pid, vid, cursor=cursor, limit=limit)


@router.get("/tts/jobs/{jid}/checkpoints/{cid}/files/{file_id}", response_model=None,
            response_class=StreamingResponse, responses=_binary_responses("application/octet-stream"))
def checkpoint_file(jid: str, cid: str, file_id: str, request: Request, c: ServiceContext = Depends(ctx)):
    return audio_response(request, tts_results.checkpoint_asset(c, jid, cid, file_id), media_type="application/octet-stream", attachment=True)


@router.post("/tts/jobs/{jid}/samples", status_code=201, response_model=m.Job, response_model_exclude_unset=True, responses={**ERRORS, 200: {"model": m.Job}})
def launch_sample(jid: str, body: TtsSampleBody, response: Response, idempotency_key: str = Header(alias="Idempotency-Key"), c: ServiceContext = Depends(ctx)):
    job, response.status_code = create_sample(c, jid, body, idempotency_key)
    return job


@router.get("/tts/jobs/{source_jid}/sample-jobs", response_model=TtsSampleJobPage, responses=ERRORS)
def source_samples(source_jid: str, cursor: str | None = None, limit: int = Query(50, ge=1, le=200), include_archived: bool = False, c: ServiceContext = Depends(ctx)):
    return tts_results.sample_jobs(c, source_jid=source_jid, cursor=cursor, limit=limit, include_archived=include_archived)


@router.get("/tts/projects/{pid}/versions/{vid}/sample-jobs", response_model=TtsSampleJobPage, responses=ERRORS)
def version_samples(pid: str, vid: str, cursor: str | None = None, limit: int = Query(50, ge=1, le=200), include_archived: bool = False, c: ServiceContext = Depends(ctx)):
    return tts_results.sample_jobs(c, pid=pid, vid=vid, cursor=cursor, limit=limit, include_archived=include_archived)


@router.get("/tts/sample-jobs/{sample_job_id}", response_model=TtsSampleJob, responses=ERRORS)
def sample_job(sample_job_id: str, c: ServiceContext = Depends(ctx)):
    return tts_results.sample_job(c, sample_job_id)


@router.get("/tts/jobs/{jid}/samples", response_model=list[TtsAudio], responses=ERRORS)
def samples(jid: str, c: ServiceContext = Depends(ctx)):
    return tts_results.audio_results(c, jid)


@router.get("/tts/jobs/{jid}/audio/{filename}", response_model=None,
            response_class=StreamingResponse, responses=_binary_responses("audio/wav"))
def audio(jid: str, filename: str, request: Request, c: ServiceContext = Depends(ctx)):
    return audio_response(request, tts_results.sample_audio_asset(c, jid, filename))
