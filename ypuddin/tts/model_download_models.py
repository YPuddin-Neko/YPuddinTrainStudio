"""Public contracts for complete, revisioned speech model packages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TtsModelExtract(_Model):
    member: str
    path: str
    size: int = Field(ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class TtsModelFile(_Model):
    path: str
    role: str
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    filename: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extract: list[TtsModelExtract] = Field(default_factory=list)


class TtsModelPackage(_Model):
    id: str
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: str
    engine: Literal["voxcpm1.5", "gpt-sovits-v5"]
    variant: Literal["v5dev", "v5turbo"] | None = None
    providers: list[Literal["huggingface"]] = Field(default_factory=lambda: ["huggingface"])
    size: int = Field(gt=0)
    files: list[TtsModelFile]
    license: str
    url: str


class TtsModelIssue(_Model):
    code: str
    message: str


class TtsModelBindings(_Model):
    model_path: str
    pretrained_gpt: str = ""
    pretrained_sovits: str = ""


class TtsInstalledModel(_Model):
    id: str
    package_id: str
    package_revision: str
    name: str
    engine: Literal["voxcpm1.5", "gpt-sovits-v5"]
    variant: Literal["v5dev", "v5turbo"] | None = None
    path: str
    status: Literal["ready", "missing", "changed", "unavailable"]
    ready: bool
    issues: list[TtsModelIssue] = Field(default_factory=list)
    bindings: TtsModelBindings


class TtsModelDownload(_Model):
    id: str
    package_id: str
    package_revision: str
    name: str
    engine: Literal["voxcpm1.5", "gpt-sovits-v5"]
    variant: Literal["v5dev", "v5turbo"] | None = None
    provider: Literal["huggingface"] = "huggingface"
    target_path: str
    status: Literal["queued", "downloading", "verifying", "completed", "failed", "cancelled"]
    phase: Literal["queued", "download", "reuse", "extract", "verify", "publish", "completed", "failed", "cancelled"]
    current_file: str | None = None
    downloaded_bytes: int = Field(default=0, ge=0)
    total_bytes: int = Field(ge=0)
    bytes_per_second: float = Field(default=0, ge=0)
    eta_seconds: float | None = Field(default=None, ge=0)
    progress_at: float | None = None
    error_code: str | None = None
    error: str | None = None
    created_at: float
    finished_at: float | None = None
    installed_id: str | None = None


class TtsModelDownloadRequest(_Model):
    package_id: str
    provider: Literal["huggingface"] = "huggingface"
