"""Complete, nullable speech validation reports for saved version recipes."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_serializer, model_validator

from .environment_models import TtsEnvironmentIdentity
from .issues import TtsIssue
from .source_models import TtsSourceSummary
from .version_config import TtsConfigScope

Count = Annotated[int, Field(ge=0)]
Timestamp = Annotated[float, Field(ge=0)]
Split = Literal["train", "validation"]
SourceState = Literal["missing", "unchecked", "checking", "valid", "invalid", "stale", "error"]
EnvironmentKey = Literal[
    "python", "upstream", "model", "dependencies", "cuda", "bf16", "tokenizer", "lora_targets"
]
ENVIRONMENT_KEYS = (
    "python",
    "upstream",
    "model",
    "dependencies",
    "cuda",
    "bf16",
    "tokenizer",
    "lora_targets",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True, validate_default=True)


class TtsTokenFilterReport(_Model):
    state: Literal["unchecked", "ready", "disabled", "not_applicable", "blocked", "error"]
    method: Literal["voxcpm_pinned_estimated_sequence_length"] = "voxcpm_pinned_estimated_sequence_length"
    max_batch_tokens: Count
    max_sample_tokens: Count | None = None
    before_count: Count | None = None
    kept_count: Count | None = None
    filtered_count: Count | None = None
    input_fingerprint: str | None = None
    issues: list[TtsIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_counts(self) -> TtsTokenFilterReport:
        counts = (self.before_count, self.kept_count, self.filtered_count)
        if any(value is not None for value in counts):
            if (
                any(value is None for value in counts)
                or self.before_count != self.kept_count + self.filtered_count
            ):
                raise ValueError("过滤统计必须完整且前后数量一致。")
        elif self.state == "ready":
            raise ValueError("已完成的过滤检查必须返回准确数量。")
        return self


class TtsBatchingReport(_Model):
    state: Literal["unchecked", "ready", "not_applicable", "blocked", "error"]
    world_size: Literal[1] = 1
    batch_size: Annotated[int, Field(ge=1)]
    shuffle: Literal[True] = True
    drop_last: bool
    input_samples: Count | None = None
    full_batches_per_pass: Count | None = None
    tail_samples_per_pass: Count | None = None
    partial_batches_per_pass: Count | None = None
    dropped_tail_samples_per_pass: Count | None = None
    yielded_batches_per_pass: Count | None = None
    yielded_samples_per_pass: Count | None = None
    can_form_full_batch: bool | None = None
    can_yield_batch: bool | None = None
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsValidationExecutionReport(_Model):
    state: Literal["unchecked", "ready", "not_applicable", "blocked", "error"]
    max_batches_per_validation: Count | None = None
    evaluated_batches_per_validation: Count | None = None
    evaluated_samples_per_validation: Count | None = None
    not_evaluated_samples_per_validation: Count | None = None
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsSplitDatasetReport(_Model):
    split: Split
    state: Literal["unchecked", "available", "unavailable", "disabled"]
    checked_at: Timestamp | None = None
    source_id: str | None = None
    source_revision: Annotated[int, Field(ge=1)] | None = None
    snapshot_id: str | None = None
    source_state: SourceState
    source_summary: TtsSourceSummary | None = None
    token_filter: TtsTokenFilterReport
    batching: TtsBatchingReport
    validation_execution: TtsValidationExecutionReport
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsDatasetReport(_Model):
    train: TtsSplitDatasetReport
    validation: TtsSplitDatasetReport

    @model_validator(mode="after")
    def correct_splits(self) -> TtsDatasetReport:
        if self.train.split != "train" or self.validation.split != "validation":
            raise ValueError("数据报告的训练集与验证集位置不匹配。")
        return self


class TtsGptSovitsPreparationReport(_Model):
    state: Literal["unchecked", "not_applicable", "blocked", "error"]
    prepared_samples: None = None
    filtered_samples: None = None
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsGptSovitsStageReport(_Model):
    stage: Literal["gpt", "sovits"]
    state: Literal["unchecked", "not_applicable", "blocked"]
    batch_size: Annotated[int, Field(ge=1)]
    input_samples: None = None
    yielded_batches_per_pass: None = None
    dropped_samples_per_pass: None = None
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsGptSovitsSplitDatasetReport(_Model):
    split: Split
    state: Literal["unchecked", "available", "unavailable", "disabled"]
    checked_at: Timestamp | None = None
    source_id: str | None = None
    source_revision: Annotated[int, Field(ge=1)] | None = None
    snapshot_id: str | None = None
    source_state: SourceState
    source_summary: TtsSourceSummary | None = None
    preparation: TtsGptSovitsPreparationReport
    stages: list[TtsGptSovitsStageReport]
    issues: list[TtsIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_stages(self) -> TtsGptSovitsSplitDatasetReport:
        if tuple(stage.stage for stage in self.stages) != ("gpt", "sovits"):
            raise ValueError("训练阶段报告必须依次包含 GPT 与 SoVITS。")
        return self


class TtsGptSovitsDatasetReport(_Model):
    engine: Literal["gpt-sovits-v5"] = "gpt-sovits-v5"
    train: TtsGptSovitsSplitDatasetReport
    validation: TtsGptSovitsSplitDatasetReport

    @model_validator(mode="after")
    def correct_splits(self) -> TtsGptSovitsDatasetReport:
        if self.train.split != "train" or self.validation.split != "validation":
            raise ValueError("数据报告的训练集与验证集位置不匹配。")
        return self


class TtsEnvironmentCheck(_Model):
    key: EnvironmentKey
    state: Literal["unchecked", "available", "unavailable"]
    checked_at: Timestamp | None = None
    issues: list[TtsIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def actual_check_time(self) -> TtsEnvironmentCheck:
        if (self.state == "unchecked") != (self.checked_at is None):
            raise ValueError("未执行的环境检查没有完成时间，已执行检查必须提供完成时间。")
        return self


class TtsDeviceCheck(_Model):
    device: str
    name: str | None = None
    uuid: str | None = None
    bf16: bool | None = None
    state: Literal["available", "unavailable"]
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsDeviceSelection(_Model):
    requested_devices: list[str]
    eligible_devices: list[str]
    checked_devices: list[TtsDeviceCheck]
    cuda_visible_devices: str | None = None
    cuda_device_order: str | None = None


class TtsEnvironmentReport(_Model):
    state: Literal["unchecked", "available", "unavailable"]
    checked_at: Timestamp | None = None
    checks: list[TtsEnvironmentCheck]
    devices: TtsDeviceSelection | None = None

    @model_validator(mode="after")
    def complete_checks(self) -> TtsEnvironmentReport:
        if tuple(check.key for check in self.checks) != ENVIRONMENT_KEYS:
            raise ValueError("环境报告必须按固定顺序返回全部检查项。")
        states = {check.state for check in self.checks}
        expected = (
            "unavailable"
            if "unavailable" in states
            else "unchecked"
            if "unchecked" in states
            else "available"
        )
        if self.state != expected:
            raise ValueError("环境总状态与各检查项不一致。")
        return self


GPT_SOVITS_ENVIRONMENT_KEYS = ("python", "upstream", "model", "dependencies", "cuda", "precision")


class TtsGptSovitsEnvironmentCheck(TtsEnvironmentCheck):
    key: Literal["python", "upstream", "model", "dependencies", "cuda", "precision"]


class TtsGptSovitsEnvironmentReport(_Model):
    engine: Literal["gpt-sovits-v5"] = "gpt-sovits-v5"
    state: Literal["unchecked", "available", "unavailable"]
    checked_at: Timestamp | None = None
    checks: list[TtsGptSovitsEnvironmentCheck]
    devices: TtsDeviceSelection | None = None

    @model_validator(mode="after")
    def complete_checks(self) -> TtsGptSovitsEnvironmentReport:
        if tuple(check.key for check in self.checks) != GPT_SOVITS_ENVIRONMENT_KEYS:
            raise ValueError("环境报告必须按固定顺序返回全部检查项。")
        states = {check.state for check in self.checks}
        expected = "unavailable" if "unavailable" in states else "unchecked" if "unchecked" in states else "available"
        if self.state != expected:
            raise ValueError("环境总状态与各检查项不一致。")
        return self


class TtsValidationReport(_Model):
    _runtime_fingerprints: list[dict] = PrivateAttr(default_factory=list)
    _runtime_model_identity: dict | None = PrivateAttr(default=None)
    _runtime_config: object | None = PrivateAttr(default=None)
    _runtime_environment: dict | None = PrivateAttr(default=None)
    scope: TtsConfigScope
    revision: Annotated[int, Field(ge=1)]
    data_revision: Annotated[int, Field(ge=1)]
    validation_id: str
    input_fingerprint: str | None = None
    checked_at: Timestamp
    valid: bool
    errors: list[TtsIssue]
    warnings: list[TtsIssue]
    dataset: TtsDatasetReport | TtsGptSovitsDatasetReport
    environment: TtsEnvironmentReport | TtsGptSovitsEnvironmentReport
    environment_binding: TtsEnvironmentIdentity | None = None

    @model_serializer(mode="wrap")
    def environment_identity(self, handler):
        value = handler(self)
        if value.get("environment_binding") is None:
            value.pop("environment_binding", None)
        return value

    @model_validator(mode="after")
    def result_consistency(self) -> TtsValidationReport:
        if isinstance(self.dataset, TtsGptSovitsDatasetReport) != isinstance(self.environment, TtsGptSovitsEnvironmentReport):
            raise ValueError("数据报告与环境报告必须属于同一语音引擎。")
        if any(issue.severity != "error" for issue in self.errors):
            raise ValueError("errors 只能包含 error 级问题。")
        if any(issue.severity != "warning" for issue in self.warnings):
            raise ValueError("warnings 只能包含 warning 级问题。")
        if self.valid and (
            self.errors
            or not self.input_fingerprint
            or self.dataset.train.state != "available"
            or self.dataset.validation.state not in ("available", "disabled")
            or self.environment.state != "available"
        ):
            raise ValueError("成功校验必须具有完整身份且数据和环境均可用。")
        return self


class TtsValidationBody(_Model):
    revision: Annotated[int, Field(ge=1)]
    data_revision: Annotated[int, Field(ge=1)]
    gpu_devices: list[Annotated[str, Field(pattern=r"^cuda:[0-9]+$")]] = Field(default_factory=list, max_length=1)


ValidationReport = TtsValidationReport
