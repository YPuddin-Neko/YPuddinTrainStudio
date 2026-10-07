"""VoxCPM 1.5 LoRA settings, separate from image training configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TtsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    engine: Literal["voxcpm1.5"] = "voxcpm1.5"
    python_path: str = ""
    trainer_path: str = ""
    model_path: str = ""
    train_manifest: str = ""
    val_manifest: str = ""
    batch_size: int = Field(1, ge=1, le=1024)
    grad_accum_steps: int = Field(1, ge=1, le=1024)
    num_workers: int = Field(0, ge=0, le=64)
    num_iters: int = Field(1000, ge=1, le=10_000_000)
    save_interval: int = Field(100, ge=1, le=10_000_000)
    learning_rate: float = Field(1e-4, gt=0, le=1)
    warmup_steps: int = Field(100, ge=0, le=10_000_000)
    lora_rank: int = Field(32, ge=1, le=512)
    lora_alpha: int = Field(16, ge=1, le=4096)

    @model_validator(mode="after")
    def _warmup(self) -> TtsConfig:
        if self.warmup_steps > self.num_iters:
            raise ValueError("预热步数不能大于训练迭代次数。")
        return self
