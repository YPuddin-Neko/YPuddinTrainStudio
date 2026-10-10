"""Response shapes for the unscoped VoxCPM draft validation endpoint."""

from pydantic import BaseModel, ConfigDict

from .validation_models import TtsDeviceSelection


class _Output(BaseModel):
    model_config = ConfigDict(extra="allow")


class TtsLegacyValidationIssue(_Output):
    loc: str
    msg: str


class TtsLegacyDataset(_Output):
    path: str
    samples: int
    duration_seconds: float
    sample_rate: int
    warnings: list[str]


class TtsLegacyUpstream(_Output):
    path: str
    revision: str


class TtsLegacyModel(_Output):
    path: str
    architecture: str
    sample_rate: int


class TtsLegacyRuntime(_Output):
    python: str | None = None
    prefix: str | None = None
    torch: str | None = None
    cuda: str | None = None
    hip: str | None = None
    devices: TtsDeviceSelection | None = None
    device: str | None = None
    voxcpm_module: str | None = None


class TtsLegacyEnvironment(_Output):
    engine: str | None = None
    sample_rate: int | None = None
    upstream: TtsLegacyUpstream | None = None
    model: TtsLegacyModel | None = None
    validation_dataset: TtsLegacyDataset | None = None
    runtime: TtsLegacyRuntime | None = None
    runtime_checked: bool | None = None


class TtsLegacyValidationReport(_Output):
    valid: bool
    errors: list[TtsLegacyValidationIssue]
    warnings: list[TtsLegacyValidationIssue]
    dataset: TtsLegacyDataset | None
    environment: TtsLegacyEnvironment | None
