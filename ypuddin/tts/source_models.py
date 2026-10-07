"""Speech source registrations and complete, paginated scan results."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .issues import TtsIssue
from .version_config import TtsConfigScope

SourceSplit = Literal["train", "validation"]
SourceState = Literal["unchecked", "checking", "valid", "invalid", "stale", "error"]


class SourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TtsSourceSummary(SourceModel):
    clips_count: int = Field(ge=0)
    valid_clips_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def counts(self) -> Self:
        if self.clips_count != self.valid_clips_count + self.invalid_count:
            raise ValueError("样本总数须等于有效行数与无效行数之和。")
        return self


class TtsSource(SourceModel):
    id: str
    scope: TtsConfigScope
    data_revision: int = Field(ge=1)
    split: SourceSplit
    path: str
    revision: int = Field(ge=1)
    state: SourceState
    check_id: str | None
    snapshot_id: str | None
    checked_at: float | None
    summary: TtsSourceSummary | None
    issues: list[TtsIssue] = Field(max_length=100)
    issues_total: int | None = Field(ge=0)
    issues_truncated: bool


class TtsSourcesResponse(SourceModel):
    scope: TtsConfigScope
    data_revision: int = Field(ge=1)
    items: list[TtsSource]


class TtsSourcePutBody(SourceModel):
    expected_data_revision: int = Field(ge=1)
    path: str = Field(min_length=1)


class TtsSourceCheckBody(SourceModel):
    expected_data_revision: int = Field(ge=1)


class TtsSourceRow(SourceModel):
    id: str
    source_id: str
    snapshot_id: str
    line: int = Field(ge=1)
    text: str | None
    audio_name: str | None
    reference_audio_name: str | None
    dataset_id: int | None = Field(ge=0)
    duration_seconds: float | None = Field(ge=0)
    sample_rate: int | None = Field(ge=0)
    channels: int | None = Field(ge=0)
    audio_url: str | None
    reference_audio_url: str | None
    issues: list[TtsIssue]


class TtsRowsResponse(SourceModel):
    source_id: str
    snapshot_id: str
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=200)
    total: int = Field(ge=0)
    items: list[TtsSourceRow]


class TtsSourceChanged(SourceModel):
    scope: TtsConfigScope
    source_id: str
    split: SourceSplit
    source_revision: int = Field(ge=1)
    data_revision: int = Field(ge=1)
    check_id: str | None
    snapshot_id: str | None
    state: SourceState | None
    reason: Literal["registered", "check_started", "check_completed", "check_failed", "stale", "removed"]
