"""Response models: the documented shape of every JSON the service returns.

They exist for the OpenAPI document (the frontend generates its TypeScript types from it), so every
model allows extra keys -- a field added by the implementation is never silently dropped from a
response, it just shows up in the schema on the next export. All ``*_at`` / ``ts`` values are Unix
seconds (float, UTC).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Out(BaseModel):
    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- system
class GpuInfo(_Out):
    index: int
    kind: Literal["cuda", "mps"] = "cuda"
    name: str
    total_mb: int | None = None


class Health(_Out):
    version: str
    api_version: int
    torch: str | None
    cuda: str | None
    mps: bool = False
    gpus: list[GpuInfo]
    families: list[str]


class GpuStats(_Out):
    index: int
    kind: Literal["cuda", "mps"] = "cuda"
    name: str
    util_pct: float | None = None
    mem_used_mb: int | None = None
    mem_total_mb: int | None = None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    telemetry_source: str | None = None
    telemetry_note: str | None = None
    cuda_available: bool | None = None


class RamStats(_Out):
    used_mb: int
    total_mb: int


class DiskStats(_Out):
    path: str
    used_gb: float
    total_gb: float


class SystemStats(_Out):
    cpu_pct: float
    ram: RamStats
    disks: list[DiskStats]
    gpus: list[GpuStats]


class SystemInfo(_Out):
    python: str
    platform: str
    packages: dict[str, str | None]
    ypuddin: str
    cuda: str | None = None
    cuda_available: bool = False


class SettingsPaths(_Out):
    data_root: str
    cache_dir: str
    models_dir: str
    output_dir: str


class SettingsServer(_Out):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class SettingsUi(_Out):
    language: Literal["zh-CN", "en"]
    theme: Literal["light", "dark", "system"]


class Settings(_Out):
    paths: SettingsPaths
    server: SettingsServer
    ui: SettingsUi


class FsEntry(_Out):
    name: str
    is_dir: bool
    size: int | None
    mtime: float


class FsList(_Out):
    path: str
    parent: str | None
    entries: list[FsEntry]


# --------------------------------------------------------------------------- config / plan / presets
class ConfigError(_Out):
    loc: str
    msg: str


class ConfigWarning(_Out):
    code: str
    msg: str


class ValidateResult(_Out):
    ok: bool
    errors: list[ConfigError]
    warnings: list[ConfigWarning]
    config: dict[str, Any] | None = None


class PlanBucket(_Out):
    w: int
    h: int
    items: int
    batches: int


class PlanParams(_Out):
    """Empty (all zeros) when the family cannot build a meta backbone for the config."""

    base: int = 0
    trainable: int = 0
    adapted_layers: int = 0
    by_algo: dict[str, int] = Field(default_factory=dict)


class PlanActivation(_Out):
    w: int
    h: int
    mb: float


class PlanMemory(_Out):
    """Empty (all zeros, no estimate) when the family cannot build a meta backbone for the config."""

    weights_mb: float = 0
    swapped_mb: float = 0
    text_encoder_mb: float = 0
    adapter_mb: float = 0
    optimizer_mb: float = 0
    heuristic: bool = True
    activations_mb_by_bucket: list[PlanActivation] = Field(default_factory=list)
    peak_mb_estimate: float | None = None
    gpu_total_mb: float | None = None
    suggestions: list[str] = Field(default_factory=list)


class Plan(_Out):
    ok: bool
    errors: list[ConfigError]
    warnings: list[ConfigWarning]
    images: int = 0
    items: int = 0
    captioned: int = 0
    buckets: list[PlanBucket] = Field(default_factory=list)
    steps_per_epoch: int = 0
    total_steps: int = 0
    epochs: int | None = None
    params: PlanParams = Field(default_factory=PlanParams)
    memory: PlanMemory = Field(default_factory=PlanMemory)
    text_encoding: str | None = None


class Preset(_Out):
    name: str
    description: str
    config: dict[str, Any]
    builtin: bool
    updated_at: float | None


class FamilyPreset(_Out):
    name: str
    description: str
    include: list[str]
    exclude: list[str]
    layers: int  # Linear modules matched on the official geometry (0 when the family has no meta backbone)


class FamilyWeight(_Out):
    field: str  # ModelConfig field name, e.g. "dit_path"
    label: str
    hint: str


class FamilySampling(_Out):
    steps: int
    cfg: float
    shift: float | None  # None: resolution dependent (Krea 2), the trainer derives it per preview size
    sampler: str


class FamilyLatent(_Out):
    channels: int
    stride: int
    patch: int
    align: int  # image side lengths must be multiples of this


class FamilyInfo(_Out):
    name: str
    label: str
    architecture: str
    adapter_prefix: str
    capabilities: list[str]
    text_modes: list[str]  # valid values of dataset.text_encoding for this family
    presets: list[FamilyPreset]
    default_preset: str
    sampling: FamilySampling
    latent: FamilyLatent
    text_max_len: int
    weights: list[FamilyWeight]
    linear_modules: int  # total Linear modules of the official geometry (0 if unknown)


class ModelAsset(_Out):
    id: str
    family: str
    kind: str
    path: str
    size: int
    dtype: str | None
    is_default: bool
    exists: bool
    created_at: float


# --------------------------------------------------------------------------- projects / datasets
class ProjectStats(_Out):
    jobs: int
    artifacts: int


class Project(_Out):
    id: str
    name: str
    note: str
    archived: bool
    created_at: float
    updated_at: float
    dataset_ids: list[str]
    active_version_id: str | None = None
    version_count: int = 1
    stats: ProjectStats


class VersionProgress(_Out):
    phase: str = "ready"
    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0


class VersionPaths(_Out):
    root: str
    config: str
    datasets: str
    runs: str
    cache: str


class VersionStats(_Out):
    datasets: int
    images: int
    jobs: int
    artifacts: int


class ProjectVersion(_Out):
    id: str
    project_id: str
    name: str
    note: str
    parent_version_id: str | None = None
    archived: bool
    busy: bool
    status: Literal["copying", "ready", "failed"]
    error: str | None = None
    created_at: float
    updated_at: float
    dataset_ids: list[str]
    paths: VersionPaths
    stats: VersionStats
    progress: VersionProgress


class DatasetSource(_Out):
    id: str
    project_id: str | None
    version_id: str | None = None
    path: str
    repeats: int
    caption_ext: str
    is_reg: bool
    prior_weight: float
    class_prompt: str | None
    created_at: float


class ResolutionCount(_Out):
    w: int
    h: int
    count: int


class AspectCount(_Out):
    ar: str
    count: int


class DatasetStats(_Out):
    images: int = 0
    captioned: int = 0
    resolutions: list[ResolutionCount] = Field(default_factory=list)
    ar_hist: list[AspectCount] = Field(default_factory=list)
    masks: int = 0
    error: str | None = None


class DatasetInfo(_Out):
    source: DatasetSource
    stats: DatasetStats
    index_status: Literal["indexing", "ready", "failed", "stale"] | str
    cache: dict[str, Any]


class DatasetImage(_Out):
    hash: str
    rel_path: str
    width: int
    height: int
    caption: str
    has_mask: bool


class ImagePage(_Out):
    items: list[DatasetImage]
    total: int
    page: int
    page_size: int


class Caption(_Out):
    caption: str


class TagBatchResult(_Out):
    changed: int


# --------------------------------------------------------------------------- jobs
JobStatus = Literal[
    "queued", "scheduled", "running", "pausing", "cancelling", "paused", "completed", "failed", "cancelled"
]


class JobProgress(_Out):
    phase: str | None = None
    step: int | None = None
    total_steps: int | None = None
    steps_per_epoch: int | None = None
    epoch: int | None = None
    eta_s: float | None = None
    it_s: float | None = None
    vram_peak_mb: float | None = None
    vram_metric: str | None = None
    device: str | None = None
    estimated_peak_mb: float | None = None
    wait_reason: str | None = None


class JobLatest(_Out):
    loss: float | None = None
    loss_ema: float | None = None
    lr: dict[str, float] | None = None


class Job(_Out):
    id: str
    type: str
    name: str
    project_id: str | None
    version_id: str | None = None
    status: JobStatus | str
    priority: int
    scheduled_at: float | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    run_dir: str | None
    progress: JobProgress
    latest: JobLatest
    error: str | None
    resume_from: str | None
    pid: int | None
    exit_code: int | None


class JobPage(_Out):
    items: list[Job]
    total: int
    page: int
    page_size: int


class ValidationPoint(_Out):
    step: int
    per_t: dict[str, float]
    mean: float


class JobMetrics(_Out):
    steps: list[int]
    loss: list[float | None]
    loss_ema: list[float | None]
    lr: dict[str, list[float]]
    grad_norm: list[float | None]
    vram_mb: list[float | None]
    vram_metric: str | None = None
    it_s: list[float | None]
    validation: list[ValidationPoint]


class JobSample(_Out):
    step: int
    prompt_index: int
    prompt: str
    seed: int
    url: str
    width: int
    height: int
    created_at: float


class JobCheckpoint(_Out):
    step: int
    kind: Literal["weights", "full"] | str
    path: str
    size: int | None
    created_at: float
    artifact_id: str | None = None
    ema: bool = False


class LogLine(_Out):
    ts: float | None
    level: str
    msg: str


class JobLog(_Out):
    lines: list[LogLine]
    next_offset: int


class QueueSettings(_Out):
    model_config = ConfigDict(extra="forbid")
    held: bool = False
    max_concurrent: int = Field(1, ge=1, le=64)
    memory_admission: bool = True


# --------------------------------------------------------------------------- artifacts
class Artifact(_Out):
    id: str
    project_id: str | None
    version_id: str | None = None
    job_id: str | None
    name: str
    path: str
    size: int
    kind: str
    step: int | None
    created_at: float
    algo: str | None = None
    rank: int | str | None = None
    alpha: float | None = None
    factor: int | None = None
    family: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Ok(_Out):
    ok: bool = True
