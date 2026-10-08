"""Portable speech training recipes, separate from version and machine settings."""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model, field_validator, model_validator

from .gpt_sovits.config import GptSettings, SovitsSettings
from .version_config import TtsSavedConfig, TtsVersionConfig

VOX_PARAMETER_FIELDS = (
    "batch_size", "grad_accum_steps", "num_workers", "preprocessing_num_workers",
    "max_batch_tokens", "num_iters", "learning_rate", "warmup_steps", "weight_decay",
    "max_grad_norm", "max_steps", "loss_diff_weight", "loss_stop_weight", "save_interval",
    "valid_interval", "log_interval", "lora_rank", "lora_alpha", "lora_dropout",
    "lora_enable_lm", "lora_target_modules_lm", "lora_enable_dit", "lora_target_modules_dit",
    "lora_enable_proj", "lora_target_proj_modules",
)
GSV_PARAMETER_FIELDS = ("stage", "gpt", "sovits")


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False,
        validate_default=True, revalidate_instances="always",
    )


class _VoxPreset(_Model):
    @model_validator(mode="after")
    def coherent(self):
        # Share the training constraints without making local paths portable fields.
        TtsVersionConfig.model_validate(self.model_dump())
        return self


TtsVoxPresetConfig = create_model(
    "TtsVoxPresetConfig",
    __base__=_VoxPreset,
    **{
        name: (TtsVersionConfig.model_fields[name].annotation, deepcopy(TtsVersionConfig.model_fields[name]))
        for name in ("engine", *VOX_PARAMETER_FIELDS)
    },
)


class TtsGptSovitsPresetConfig(_Model):
    engine: Literal["gpt-sovits-v5"] = "gpt-sovits-v5"
    stage: Literal["both", "gpt", "sovits"] = "both"
    gpt: GptSettings = Field(default_factory=GptSettings)
    sovits: SovitsSettings = Field(default_factory=SovitsSettings)


TtsPresetConfig = Annotated[
    TtsVoxPresetConfig | TtsGptSovitsPresetConfig, Field(discriminator="engine")
]


class _Named(_Model):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=4096)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        reserved = {"con", "prn", "aux", "nul"} | {
            f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
        }
        if not value.replace("-", "").replace("_", "").isalnum() or value.casefold() in reserved:
            raise ValueError("预设名称须为字母、数字、中文、短横线或下划线，且不能使用系统保留名称。")
        return value


class TtsPresetDocument(_Named):
    format: Literal["ypuddin-tts-preset"]
    schema_version: Literal[1]
    config: TtsPresetConfig

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("预设格式版本必须是整数。")
        return value


class TtsPreset(TtsPresetDocument):
    id: str = Field(pattern=r"^tsp_[0-9a-f]{12}$")
    revision: int = Field(ge=1)
    created_at: float
    updated_at: float


class TtsPresetCreateBody(_Named):
    config: TtsSavedConfig


class TtsPresetUpdateBody(TtsPresetCreateBody):
    expected_revision: int = Field(ge=1)


class TtsPresetResolveBody(_Model):
    config: dict[str, JsonValue]


class TtsPresetResolveResponse(_Model):
    preset_id: str
    preset_revision: int = Field(ge=1)
    config: TtsSavedConfig
    changed_fields: list[str]


class TtsPresetDeleted(_Model):
    ok: Literal[True] = True
