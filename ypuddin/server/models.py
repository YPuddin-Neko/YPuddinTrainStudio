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


class ModelBrowseRoot(_Out):
    path: str


# --------------------------------------------------------------------------- system
class GpuInfo(_Out):
    index: int
    kind: Literal["cuda", "mps", "dtk", "rocm"] = "cuda"
    name: str
    total_mb: int | None = None
    hip_runtime: str | None = None


class Health(_Out):
    version: str
    api_version: int
    torch: str | None
    cuda: str | None
    hip: str | None = None
    hip_available: bool = False
    mps: bool = False
    gpus: list[GpuInfo]
    families: list[str]


class GpuStats(_Out):
    index: int
    kind: Literal["cuda", "mps", "dtk", "rocm"] = "cuda"
    name: str
    util_pct: float | None = None
    mem_used_mb: int | None = None
    mem_free_mb: int | None = None
    mem_reserved_mb: int | None = None
    mem_total_mb: int | None = None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    power_source: str | None = None
    power_estimated: bool | None = None
    power_sample_seconds: float | None = None
    temperature_source: str | None = None
    temp_max_c: float | None = None
    temp_sensor_count: int | None = None
    telemetry_source: str | None = None
    telemetry_note: str | None = None
    cuda_available: bool | None = None
    hip_runtime: str | None = None


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
    hip: str | None = None
    hip_available: bool = False


class SettingsPaths(_Out):
    state_dir: str = ""
    samples_dir: str = ""
    logs_dir: str = ""
    bootstrap_env_dir: str = ""
    data_root: str
    cache_dir: str
    models_dir: str
    output_dir: str
    output_mode: Literal["project", "custom"] = "project"


class SettingsServer(_Out):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    open_browser: bool = True


class SettingsUi(_Out):
    language: Literal["zh-CN", "en"]
    theme: Literal["light", "dark", "system"]


class SettingsNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proxy_mode: Literal["system", "direct", "custom"] = "system"
    proxy_url: str = ""
    proxy_username: str = ""
    proxy_password_configured: bool = False


class SettingsDownloads(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pypi: Literal["ustc", "tuna", "aliyun", "official"] = "ustc"
    pytorch: Literal["mirror", "aliyun", "sjtu", "official"] = "mirror"
    fallback: bool = True


class Settings(_Out):
    paths: SettingsPaths
    server: SettingsServer
    ui: SettingsUi
    network: SettingsNetwork = Field(default_factory=SettingsNetwork)
    downloads: SettingsDownloads = Field(default_factory=SettingsDownloads)


class FsEntry(_Out):
    name: str
    is_dir: bool
    size: int | None
    mtime: float


class StoragePathPreview(_Out):
    path: str
    browse_root: str


class StorageDefaults(_Out):
    bootstrap_env_dir: StoragePathPreview
    output_dir: StoragePathPreview
    state_dir: StoragePathPreview
    samples_dir: StoragePathPreview
    logs_dir: StoragePathPreview


class BrowseRoot(_Out):
    path: str


class MembershipResult(_Out):
    changed: int
    included: bool


class FsList(_Out):
    exists: bool = True
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


class ConfigInspectionField(_Out):
    loc: str
    path: list[str | int]
    value: Any
    kind: Literal["unknown", "inactive"]
    condition: str | None = None


class ConfigInspection(_Out):
    fields: list[ConfigInspectionField]
    errors: list[ConfigError]


class PlanBucket(_Out):
    base: int = 0  # bucket-mode base resolution; 0 for native sizes
    w: int
    h: int
    items: int
    batches: int | None


class PlanParams(_Out):
    """Empty (all zeros) when the family cannot build a meta backbone for the config."""

    base: int = 0
    trainable: int = 0
    adapted_layers: int = 0
    by_algo: dict[str, int] = Field(default_factory=dict)
    training_mode: Literal["adapter", "full"] = "adapter"
    components: dict[str, int] = Field(default_factory=dict)


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
    gradients_mb: float = 0
    estimate_scope: Literal["per_device"] = "per_device"
    communication_mb_estimate: float = 0
    optimizer_workspace_mb_estimate: float = 0
    initialization_peak_mb_estimate: float | None = None
    estimate_notes: list[str] = Field(default_factory=list)
    sharding: dict[str, Any] | None = None
    heuristic: bool = True
    activations_mb_by_bucket: list[PlanActivation] = Field(default_factory=list)
    peak_mb_estimate: float | None = None
    training_peak_mb_estimate: float | None = None
    known_training_residency_mb: float | None = None
    unestimated_components: list[str] = Field(default_factory=list)
    gpu_total_mb: float | None = None
    suggestions: list[str] = Field(default_factory=list)


class NativePlan(_Out):
    images: int
    downscaled: int
    sizes: int
    logical_batches: int
    max_pixels: int
    alignment: int
    batch_size: int
    forward_groups: int | None
    synchronization_groups: int = 0


class ImageFitGeometry(_Out):
    path: str
    source_width: int
    source_height: int
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int
    right: int
    bottom: int
    padding_pixels: int
    cropped_pixels: int


class ImageFitPlan(_Out):
    mode: Literal["crop", "pad"]
    crop_anchor: str = "center"
    padded_images: int
    cropped_images: int
    padding_pixels: int
    total_pixels: int
    padding_fraction: float
    total_shapes: int
    truncated: bool
    items: list[ImageFitGeometry]


class PlanDistributed(_Out):
    strategy: Literal["single", "ddp", "fsdp"] = "single"
    parameter_storage: Literal["replicated", "sharded"] = "replicated"
    gradient_storage: Literal["replicated", "sharded"] = "replicated"
    optimizer_storage: Literal["replicated", "sharded"] = "replicated"
    world_size: int
    per_device_batch_size: int
    effective_batch_size: int
    batches_per_rank: int
    dropped_samples: int
    tail_policy: str


class PlanSourceBalance(_Out):
    source_index: int
    path: str
    is_reg: bool
    images: int
    repeats: int
    repeated_images: int
    resolution_variants: int
    items: int


class Plan(_Out):
    ok: bool
    errors: list[ConfigError]
    warnings: list[ConfigWarning]
    compute_policy: dict[str, Any] | None = None
    source_balance: list[PlanSourceBalance] | None = None
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
    native: NativePlan | None = None
    image_fit: ImageFitPlan | None = None
    distributed: PlanDistributed | None = None


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
    kind: str = ""
    required: bool = True
    downloadable: bool = True


class FamilySampling(_Out):
    steps: int
    cfg: float
    shift: float | None  # None: resolution dependent (Krea 2), the trainer derives it per preview size
    sampler: str
    guidance: float | None = None


class FamilyLatent(_Out):
    channels: int
    stride: int
    patch: int
    align: int  # image side lengths must be multiples of this


class FamilyInfo(_Out):
    runtime_backend: Literal["cuda", "hip", "mps", "cpu"] | None = None
    attention_backends: list[str] = Field(default_factory=lambda: ["auto", "sdpa", "xformers", "flash_attn"])
    name: str
    label: str
    architecture: str
    objective: str = "rectified_flow"
    sampling_samplers: list[str] = ["euler", "heun", "er_sde"]
    sampling_schedulers: list[str] = ["uniform", "simple", "sgm_uniform", "normal"]
    objective_timestep_sampling: list[str] = [
        "uniform",
        "logit_normal",
        "shift",
        "resolution_shift",
        "mode",
        "cosmap",
    ]
    objective_weighting: list[str] = ["none", "sigma_sqrt", "cosmap", "snr_like", "cosmos"]
    adapter_prefix: str
    capabilities: list[str]
    caption_formats: list[Literal["txt", "json"]] = Field(default_factory=lambda: ["txt", "json"])
    text_modes: list[str]  # valid values for a frozen encoder
    training_capabilities: dict[str, Any] = Field(default_factory=dict)
    presets: list[FamilyPreset]
    default_preset: str
    sampling: FamilySampling
    latent: FamilyLatent
    text_max_len: int
    weights: list[FamilyWeight]
    linear_modules: int  # total Linear modules of the official geometry (0 if unknown)


class ModelAsset(_Out):
    compatible_families: list[str] = Field(default_factory=list)
    id: str
    family: str
    kind: str
    path: str
    size: int
    dtype: str | None
    is_default: bool
    purpose: Literal["training", "inference"] = "training"
    variant: Literal["raw", "turbo"] | None = None
    exists: bool
    unsupported_reason: str | None = None  # Live admission projection; retained DB defaults are unchanged.
    created_at: float


# --------------------------------------------------------------------------- projects / datasets
class ProjectStats(_Out):
    jobs: int
    artifacts: int


class ProjectActivity(_Out):
    id: str
    name: str
    status: str
    step: int | None = None
    total_steps: int | None = None
    created_at: float
    finished_at: float | None = None
    error: str | None = None


class Project(_Out):
    id: str
    name: str
    note: str
    archived: bool
    created_at: float
    updated_at: float
    dataset_ids: list[str]
    active_version_id: str | None = None
    active_version_name: str | None = None
    active_version_number: int | None = None
    version_count: int = 1
    layout_version: int = 1
    stats: ProjectStats
    image_count: int | None = None
    latest_job: ProjectActivity | None = None
    category: str | None = None
    cover_url: str | None = None
    active_family: str | None = None


class ProjectPage(_Out):
    items: list[Project]
    total: int
    page: int
    page_size: int


class ProjectCategory(_Out):
    name: str
    count: int


class ProjectCategories(_Out):
    items: list[ProjectCategory]
    uncategorized: int
    total: int


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
    traindata: str
    reg: str
    samples: str
    output: str


class VersionStats(_Out):
    datasets: int
    images: int
    jobs: int
    artifacts: int


class ProjectVersion(_Out):
    id: str
    project_id: str
    number: int = 1
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
    family: str | None = None


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
    can_rename: bool | None = None
    can_append: bool = False


class ResolutionCount(_Out):
    w: int
    h: int
    count: int


class AspectCount(_Out):
    ar: str
    count: int


class DatasetStats(_Out):
    training_images: int | None = None
    held_out_images: int | None = None
    images: int = 0
    captioned: int = 0
    resolutions: list[ResolutionCount] = Field(default_factory=list)
    ar_hist: list[AspectCount] = Field(default_factory=list)
    masks: int = 0
    error: str | None = None


class DatasetInfo(_Out):
    masked_loss: bool | None = None
    source: DatasetSource
    stats: DatasetStats
    index_status: Literal["indexing", "ready", "failed", "stale"] | str
    cache: dict[str, Any]


class DatasetUploadInfo(DatasetInfo):
    datasets: list[DatasetInfo]


class DatasetImportProgress(_Out):
    id: str
    phase: Literal["receiving", "extracting", "validating", "copying", "registering", "completed", "failed"]
    bytes_done: int
    bytes_total: int | None
    files_done: int
    files_total: int | None
    elapsed_seconds: float
    phase_elapsed_seconds: float
    bytes_per_second: float | None
    eta_seconds: float | None
    error: str | None


class CaptionField(_Out):
    path: list[str]
    role: Literal[
        "quality",
        "count",
        "character",
        "character_name",
        "character_variant",
        "character_full",
        "series",
        "artist",
        "appearance",
        "tags",
        "environment",
        "nl",
        "trigger",
    ]
    value: str | list[str]
    present: bool


class CaptionStructure(_Out):
    format: Literal["full", "nested", "simple", "flat", "legacy_override", "unknown"]
    document: dict[str, Any]
    fields: list[CaptionField]
    revision: str
    editable: bool
    legacy_override: bool
    reason: str | None = None


class DatasetImage(_Out):
    hash: str
    rel_path: str
    width: int
    height: int
    caption: str
    caption_tags: str | None = None
    caption_description: str | None = None
    caption_structure: CaptionStructure | None = None
    caption_format: str | None = None
    caption_error: str | None = None
    caption_status: Literal["captioned", "missing", "invalid"] = "missing"
    has_mask: bool
    training_enabled: bool | None = None


class ImagePage(_Out):
    items: list[DatasetImage]
    total: int
    page: int
    page_size: int


class CaptionTagCount(_Out):
    tag: str
    count: int


class DatasetCaptionStats(_Out):
    images: int
    captioned: int
    missing: int
    invalid: int
    formats: dict[str, int]
    unique_tags: int
    tags: list[CaptionTagCount]


class Caption(_Out):
    caption: str
    caption_structure: CaptionStructure | None = None


class TagBatchResult(_Out):
    changed: int


# --------------------------------------------------------------------------- jobs
JobStatus = Literal[
    "queued", "scheduled", "running", "pausing", "cancelling", "paused", "completed", "failed", "cancelled"
]


class JobProgress(_Out):
    phase: str | None = None
    done: int | None = None
    total: int | None = None
    cell_index: int | None = None
    sample_step: int | None = None
    sample_steps: int | None = None
    step: int | None = None
    total_steps: int | None = None
    steps_per_epoch: int | None = None
    epoch: int | None = None
    eta_s: float | None = None
    it_s: float | None = None
    vram_peak_mb: float | None = None
    vram_metric: str | None = None
    device: str | None = None
    devices: list[str] = Field(default_factory=list)
    gpu_count: int | None = None
    estimated_peak_mb: float | None = None
    wait_reason: str | None = None


class JobLatest(_Out):
    loss_mean: float | None = None
    loss_count: int | None = None
    loss_mean_scope: Literal["run", "since_resume"] | None = None
    loss: float | None = None
    loss_ema: float | None = None
    lr: dict[str, float] | None = None


class Job(_Out):
    id: str
    type: str
    name: str
    project_id: str | None
    version_id: str | None = None
    project_name: str | None = None
    version_name: str | None = None
    version_number: int | None = None
    status: JobStatus | str
    priority: int
    gpu_devices: list[str] = Field(default_factory=list)
    scheduled_at: float | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    run_dir: str | None
    samples_dir: str | None = None
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
    loss: float | None = Field(description="Actual training loss at this exact step; null when unavailable.")


class JobCheckpoint(_Out):
    step: int
    kind: Literal["weights", "full"] | str
    path: str
    size: int | None
    created_at: float
    artifact_id: str | None = None
    ema: bool = False


class LogLine(_Out):
    offset: int = 0
    kind: Literal["record", "traceback", "text"] = "text"
    ts: float | None
    level: str
    source: str | None = None
    msg: str


class JobLog(_Out):
    lines: list[LogLine]
    start_offset: int = 0
    next_offset: int
    has_more: bool = False
    has_earlier: bool = False


class QueueSettings(_Out):
    model_config = ConfigDict(extra="forbid")
    held: bool = False
    max_concurrent: int | None = Field(None, ge=1, le=64)
    memory_admission: bool = True


class QueueDevice(_Out):
    device: str
    name: str
    mem_used_mb: float | None = None
    mem_reserved_mb: float | None = None
    mem_free_mb: float | None = None
    mem_total_mb: float | None = None
    job_id: str | None = None
    job_name: str | None = None
    status: str | None = None


class QueueDevices(_Out):
    devices: list[QueueDevice]
    max_concurrent: int | None = None


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
