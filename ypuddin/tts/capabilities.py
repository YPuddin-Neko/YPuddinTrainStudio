"""Static engine capabilities and ordered fields from the executable configuration schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .core import UPSTREAM_REVISION
from .gpt_sovits.config import GptSettings, GptSovitsVersionConfig, SovitsSettings
from .version_config import TtsVersionConfig


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TtsCapabilitySetting(_Model):
    key: str
    value: Any = None
    reason_code: str
    reason: str


class TtsAudioCapability(_Model):
    format: Literal["pcm_wav"] = "pcm_wav"
    sample_rate: Literal[44100] = 44100
    channels: Literal[1] = 1


class TtsEngineCapability(_Model):
    id: Literal["voxcpm1.5"] = "voxcpm1.5"
    training_modes: list[Literal["lora"]] = Field(default_factory=lambda: ["lora"])
    devices: list[Literal["cuda"]] = Field(default_factory=lambda: ["cuda"])
    gpu_count: Literal[1] = 1
    training_precision: Literal["bf16_autocast"] = "bf16_autocast"
    audio: TtsAudioCapability = Field(default_factory=TtsAudioCapability)
    upstream_revision: str = UPSTREAM_REVISION
    fixed: list[TtsCapabilitySetting]
    unsupported: list[TtsCapabilitySetting]
    schema_url: str = "/api/tts/schema/train?engine=voxcpm1.5"


class GptSovitsAudioCapability(_Model):
    format: Literal["pcm_wav"] = "pcm_wav"
    sample_rates: list[int] = Field(default_factory=lambda: [32000, 44100, 48000])
    output_sample_rate: Literal[48000] = 48000
    channels: Literal[1] = 1
    languages: list[str] = Field(default_factory=lambda: ["zh", "en", "ja", "ko", "yue"])
    manifest_formats: list[str] = Field(default_factory=lambda: ["jsonl", "list"])


class GptSovitsEngineCapability(_Model):
    id: Literal["gpt-sovits-v5"] = "gpt-sovits-v5"
    variants: list[Literal["v5dev", "v5turbo"]] = Field(default_factory=lambda: ["v5dev", "v5turbo"])
    training_modes: list[Literal["gpt_finetune", "sovits_lora"]] = Field(default_factory=lambda: ["gpt_finetune", "sovits_lora"])
    devices: list[Literal["cuda"]] = Field(default_factory=lambda: ["cuda"])
    gpu_count: Literal[1] = 1
    training_precision: Literal["configurable_fp16_fp32"] = "configurable_fp16_fp32"
    audio: GptSovitsAudioCapability = Field(default_factory=GptSovitsAudioCapability)
    upstream_revision: str = "f652b1da5af29a6955f9c3911aa71b7daa6618bc"
    upstream_branch: str = "cuda_graph_accel_v5"
    fixed: list[TtsCapabilitySetting]
    unsupported: list[TtsCapabilitySetting]
    schema_url: str = "/api/tts/schema/train?engine=gpt-sovits-v5"


class TtsCapabilities(_Model):
    contract_version: Literal[1] = 1
    engines: list[TtsEngineCapability | GptSovitsEngineCapability]


class TtsSchemaGroup(_Model):
    id: str
    fields: list[str]


class TtsTrainSchema(_Model):
    contract_version: Literal[1] = 1
    engine: Literal["voxcpm1.5", "gpt-sovits-v5"] = "voxcpm1.5"
    schema_version: Literal[1] = 1
    json_schema: dict[str, Any]
    groups: list[TtsSchemaGroup]
    fixed: list[TtsCapabilitySetting]
    unsupported: list[TtsCapabilitySetting]


GROUPS = (
    ("environment", ("engine", "python_path", "trainer_path", "model_path")),
    (
        "batching",
        ("batch_size", "grad_accum_steps", "num_workers", "preprocessing_num_workers", "max_batch_tokens"),
    ),
    (
        "optimization",
        (
            "num_iters",
            "learning_rate",
            "warmup_steps",
            "weight_decay",
            "max_grad_norm",
            "max_steps",
            "loss_diff_weight",
            "loss_stop_weight",
        ),
    ),
    ("intervals", ("save_interval", "valid_interval", "log_interval")),
    (
        "lora",
        (
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_enable_lm",
            "lora_target_modules_lm",
            "lora_enable_dit",
            "lora_target_modules_dit",
            "lora_enable_proj",
            "lora_target_proj_modules",
        ),
    ),
)


def _fixed() -> list[TtsCapabilitySetting]:
    rows = (
        ("optimizer", "AdamW", "沿用 VoxCPM 训练器的 AdamW 优化器。"),
        ("scheduler", "cosine_with_warmup", "使用带预热的余弦学习率调度。"),
        ("betas", None, "使用所选 PyTorch 环境的 AdamW 默认 betas。"),
        ("epsilon", None, "使用所选 PyTorch 环境的 AdamW 默认 epsilon。"),
        ("sample_rate", 44100, "VoxCPM 1.5 使用 44100 Hz 单声道 PCM WAV。"),
        ("gpu_count", 1, "每个任务使用一张 NVIDIA GPU。"),
        ("training_precision", "bf16_autocast", "训练使用 BF16 autocast；AudioVAE 保持 FP32。"),
        ("output_dir", "managed", "检查点目录按项目、版本和任务自动分配。"),
        ("tensorboard_dir", "managed", "TensorBoard 目录由任务自动分配。"),
        ("distribute", False, "使用单进程、单卡训练。"),
        ("checkpoint_base_model", "local_path", "检查点记录任务使用的本地底模路径。"),
        ("config_path", "managed", "训练器配置从已保存的版本参数生成。"),
        ("out_sample_rate", 44100, "音频输出采样率由 VoxCPM 1.5 适配器固定。"),
    )
    return [
        TtsCapabilitySetting(key=key, value=value, reason_code=f"tts.fixed.{key}", reason=reason)
        for key, value, reason in rows
    ]


def _unsupported() -> list[TtsCapabilitySetting]:
    rows = (
        ("other_models", "此参数配置适用于 VoxCPM 1.5，不适用于 VoxCPM 2 或其他引擎。"),
        ("full_finetune", "当前任务仅支持 LoRA 训练。"),
        ("multi_gpu", "当前任务未接入多卡训练。"),
        ("pause", "当前语音训练任务不支持暂停。"),
        ("resume", "当前语音训练任务不支持恢复训练。"),
        ("checkpoint_retention", "固定训练入口没有可配置的检查点保留数量。"),
        ("training_seed", "固定训练入口没有可配置的训练随机种子。"),
        ("optimizer_selection", "固定训练入口不提供优化器类型选择。"),
        ("scheduler_selection", "固定训练入口不提供学习率调度器类型选择。"),
    )
    return [
        TtsCapabilitySetting(key=key, reason_code=f"tts.unsupported.{key}", reason=reason)
        for key, reason in rows
    ]


def get_capabilities() -> TtsCapabilities:
    return TtsCapabilities(engines=[
        TtsEngineCapability(fixed=_fixed(), unsupported=_unsupported()),
        GptSovitsEngineCapability(fixed=_gsv_fixed(), unsupported=_gsv_unsupported()),
    ])


def _gsv_fixed() -> list[TtsCapabilitySetting]:
    rows = (
        ("gpu_count", 1, "每个任务使用一张 NVIDIA GPU，两个训练阶段依次执行。"),
        ("output_dir", "managed", "预处理数据和训练产物保存到本任务目录，原始录音保持不变。"),
        ("checkpoint_pair", ["gpt.ckpt", "sovits.pth"], "每个结果包含配套 GPT 与 SoVITS 权重；单阶段训练配对未训练阶段的底模。"),
        ("save_every_weights", True, "各阶段保留间隔权重；任务完成后发布配套结果用于试听。"),
        ("gpt_accumulation", "upstream", "沿用上游 GPT 手动优化循环：首次更新累计 5 批，随后每 4 批更新。"),
        ("out_sample_rate", 48000, "v5 使用配套声码器输出 48000 Hz 音频。"),
    )
    return [TtsCapabilitySetting(key=k, value=v, reason_code=f"tts.gpt_sovits.fixed.{k}", reason=r) for k, v, r in rows]


def _gsv_unsupported() -> list[TtsCapabilitySetting]:
    rows = (
        ("validation_source", "上游训练入口不接收独立验证清单。"),
        ("resume", "当前任务从所选基础权重开始训练，不支持恢复优化器和随机数状态。"),
        ("pause", "当前语音训练任务不支持暂停。"),
        ("multi_gpu", "当前任务未接入多卡训练。"),
        ("gradient_clip", "上游 GPT 配置中的 gradient_clip 未被训练循环使用。"),
        ("sovits_full_finetune", "此入口使用上游 SoVITS LoRA 训练脚本，保留其非 CFM 模块训练行为。"),
        ("dataset_tools", "请提供已切分和校对文本的录音；此入口不执行 ASR、降噪或人声分离。"),
    )
    return [TtsCapabilitySetting(key=k, reason_code=f"tts.gpt_sovits.unsupported.{k}", reason=r) for k, r in rows]


def get_train_schema(engine: str = "voxcpm1.5") -> TtsTrainSchema:
    if engine == "gpt-sovits-v5":
        return TtsTrainSchema(
            engine=engine,
            json_schema=GptSovitsVersionConfig.model_json_schema(),
            groups=[
                TtsSchemaGroup(id="environment", fields=[k for k in GptSovitsVersionConfig.model_fields if k not in {"gpt", "sovits"}]),
                TtsSchemaGroup(id="gpt", fields=[f"gpt.{k}" for k in GptSettings.model_fields]),
                TtsSchemaGroup(id="sovits", fields=[f"sovits.{k}" for k in SovitsSettings.model_fields]),
            ],
            fixed=_gsv_fixed(), unsupported=_gsv_unsupported(),
        )
    if engine != "voxcpm1.5":
        raise ValueError("不支持此语音引擎。")
    fields = [field for _, group in GROUPS for field in group]
    if len(fields) != len(set(fields)) or set(fields) != set(TtsVersionConfig.model_fields):
        raise RuntimeError("TTS schema groups do not cover the configuration exactly once")
    return TtsTrainSchema(
        json_schema=TtsVersionConfig.model_json_schema(),
        groups=[TtsSchemaGroup(id=key, fields=list(group)) for key, group in GROUPS],
        fixed=_fixed(),
        unsupported=_unsupported(),
    )
