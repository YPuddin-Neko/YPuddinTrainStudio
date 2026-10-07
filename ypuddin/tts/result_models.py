"""Persisted speech result projections; absent historical values stay absent."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from ypuddin.server.models import Job

from .issues import TtsIssue
from .sample_config import GptSovitsSampleOptions


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def engine_options(self, handler):
        value = handler(self)
        if value.get("gpt_sovits") is None:
            value.pop("gpt_sovits", None)
        return value


class TtsCheckpointFile(_Result):
    id: str
    name: str
    role: str
    size: int
    download_url: str


class TtsStageCheckpointInfo(_Result):
    epoch: int | None = Field(None, ge=0)
    global_step: int | None = Field(None, ge=0)


class GptSovitsCheckpointInfo(_Result):
    variant: Literal["v5dev", "v5turbo"]
    stage: Literal["gpt", "sovits", "both"]
    gpt: TtsStageCheckpointInfo
    sovits: TtsStageCheckpointInfo


class TtsCheckpoint(_Result):
    id: str
    revision: str
    source_job_id: str
    project_id: str | None
    version_id: str | None
    name: str
    path: str
    relative_path: str
    step: int | None
    upstream_step: int | None
    size: int
    created_at: float | None
    discovered_at: float
    modified_at: float
    can_preview: bool
    unavailable_reason: TtsIssue | None
    files: list[TtsCheckpointFile]
    gpt_sovits: GptSovitsCheckpointInfo | None = None


class TtsCheckpointPage(_Result):
    items: list[TtsCheckpoint]
    next_cursor: str | None
    as_of: float


class TtsSampleRequestSnapshot(_Result):
    text: str | None
    reference_audio: str | None
    reference_text: str | None
    seed: int | None
    cfg_value: float | None
    inference_timesteps: int | None
    gpt_sovits: GptSovitsSampleOptions | None = None


class TtsSampleSource(_Result):
    job_id: str
    name: str | None
    record_exists: bool
    checkpoint_available: bool


class TtsAudio(_Result):
    id: str
    sample_job_id: str
    source_job_id: str
    project_id: str | None
    version_id: str | None
    checkpoint_id: str | None
    filename: str
    text: str | None
    requested_seed: int | None
    seed: int | None
    cfg_value: float | None
    inference_timesteps: int | None
    duration_seconds: float | None
    sample_rate: int | None
    created_at: float | None
    size: int | None
    available: bool
    url: str | None
    gpt_sovits: GptSovitsSampleOptions | None = None


class TtsSampleJob(_Result):
    job: Job
    source: TtsSampleSource
    checkpoint_id: str | None
    checkpoint_revision: str | None
    request: TtsSampleRequestSnapshot
    audio: TtsAudio | None


class TtsSampleJobPage(_Result):
    items: list[TtsSampleJob]
    next_cursor: str | None
    as_of: float


Checkpoint = TtsCheckpoint
CheckpointPage = TtsCheckpointPage
SampleJob = TtsSampleJob
SampleJobPage = TtsSampleJobPage
Audio = TtsAudio
RequestSnapshot = TtsSampleRequestSnapshot
