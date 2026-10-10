"""Local speech environment discovery and preparation API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .issues import TtsIssue

Engine = Literal["voxcpm1.5", "gpt-sovits-v5"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TtsEnvironmentIdentity(_Model):
    id: str
    revision: str
    engine: Engine
    python_path: str
    trainer_path: str
    upstream_revision: str
    dependency_fingerprint: str
    resources_marker_sha256: str | None = None


class TtsEnvironmentCheckResult(_Model):
    engine: Engine
    state: Literal["unchecked", "ready", "missing", "incompatible", "error"] = "unchecked"
    checked_at: float | None = None
    python_version: str | None = None
    dependency_fingerprint: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    imports_checked: bool = False
    source_checked: bool = False
    gpu_state: Literal["unchecked"] = "unchecked"
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsPythonCandidate(_Model):
    id: str
    kind: Literal["deployment", "managed"]
    python_path: str
    available: bool
    checks: list[TtsEnvironmentCheckResult] = Field(default_factory=list)


class TtsManagedEnvironment(TtsEnvironmentIdentity):
    kind: Literal["deployment", "managed"]
    state: Literal["ready", "missing", "changed", "unavailable"]
    created_at: float
    checked_at: float
    check: TtsEnvironmentCheckResult
    issues: list[TtsIssue] = Field(default_factory=list)


class TtsEnvironmentOperation(_Model):
    id: str
    action: Literal["check", "prepare"]
    engine: Engine
    candidate_id: str | None = None
    make_default: bool = True
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    phase: str
    progress: float | None = Field(default=None, ge=0, le=1)
    created_at: float
    updated_at: float
    environment_id: str | None = None
    retry_of: str | None = None
    cancellable: bool
    issues: list[TtsIssue] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    log_count: int = 0
    logs_truncated: bool = False


class TtsEnvironmentSnapshot(_Model):
    candidates: list[TtsPythonCandidate]
    environments: list[TtsManagedEnvironment]
    defaults: dict[Engine, str | None]
    operations: list[TtsEnvironmentOperation]


class TtsEnvironmentCheckRequest(_Model):
    engine: Engine
    candidate_id: str | None = None


class TtsEnvironmentPrepareRequest(TtsEnvironmentCheckRequest):
    make_default: bool = True


class TtsEnvironmentDefaultRequest(_Model):
    environment_id: str
