"""Upload session status, read by pages that continue an interrupted dataset upload."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from . import models as m
from .context import ServiceContext

router = APIRouter()


def ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


class UploadSessionError(BaseModel):
    code: str
    message: str
    # False when the files themselves were rejected: only a new upload can import them.
    retryable: bool


class DatasetUploadSessionStatus(BaseModel):
    id: str
    # receiving: bytes missing; ready: every byte received, not imported; finalizing: importing now.
    state: Literal["receiving", "ready", "finalizing", "completed", "failed"]
    # Bytes received per manifest file, in manifest order: where each file continues.
    received: list[int]
    chunk_bytes: int
    progress_id: str
    # Seconds left before an idle session and its staged files are removed.
    expires_in: float
    result: m.DatasetUploadInfo | None = None
    error: UploadSessionError | None = None


@router.get(
    "/projects/{pid}/datasets/upload-sessions/{sid}",
    response_model=DatasetUploadSessionStatus, response_model_exclude_unset=True,
)
def upload_session_status(pid: str, sid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return c.upload_sessions.status(pid, sid)
