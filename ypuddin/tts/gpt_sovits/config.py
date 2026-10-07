"""Settings consumed by the official GPT-SoVITS v5 trainers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False,
                              validate_default=True, revalidate_instances="always")


class GptSettings(_Model):
    epochs: int = Field(15, ge=1, le=10000)
    batch_size: int = Field(8, ge=1, le=1024)
    precision: Literal["16-mixed", "32-true"] = "16-mixed"
    seed: int = Field(1234, ge=0, le=4294967295)
    save_every_epoch: int = Field(5, ge=1, le=10000)
    save_latest: bool = True
    learning_rate: float = Field(0.01, gt=0, le=1)
    initial_learning_rate: float = Field(1e-5, gt=0, le=1)
    final_learning_rate: float = Field(1e-4, gt=0, le=1)
    warmup_steps: int = Field(2000, ge=1, le=10000000)
    decay_steps: int = Field(40000, ge=2, le=10000000)
    dpo: bool = False
    max_seconds: int = Field(54, ge=1, le=54)
    num_workers: int = Field(4, ge=1, le=64)

    @model_validator(mode="after")
    def coherent(self):
        if self.warmup_steps >= self.decay_steps:
            raise ValueError("GPT 预热步数须小于调度总步数。")
        if self.save_every_epoch > self.epochs or self.epochs % self.save_every_epoch:
            raise ValueError("GPT 保存间隔须整除训练轮数，以保留最后一轮导出权重。")
        return self


class SovitsSettings(_Model):
    epochs: int = Field(2, ge=1, le=10000)
    batch_size: int = Field(1, ge=1, le=1024)
    precision: Literal["fp16", "fp32"] = "fp16"
    seed: int = Field(1234, ge=0, le=4294967295)
    save_every_epoch: int = Field(1, ge=1, le=10000)
    save_latest: bool = True
    learning_rate: float = Field(1e-4, gt=0, le=1)
    adam_beta1: float = Field(0.8, ge=0, lt=1)
    adam_beta2: float = Field(0.99, ge=0, lt=1)
    adam_epsilon: float = Field(1e-9, gt=0, le=1)
    lr_decay: float = Field(0.999875, gt=0, le=1)
    log_interval: int = Field(100, ge=1, le=10000000)
    lora_rank: Literal[16, 32, 64, 128] = 32
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def coherent(self):
        if self.save_every_epoch > self.epochs or self.epochs % self.save_every_epoch:
            raise ValueError("SoVITS 保存间隔须整除训练轮数，以保留最后一轮导出权重。")
        return self


class GptSovitsVersionConfig(_Model):
    engine: Literal["gpt-sovits-v5"] = "gpt-sovits-v5"
    variant: Literal["v5dev", "v5turbo"] = "v5dev"
    stage: Literal["both", "gpt", "sovits"] = "both"
    python_path: str = ""
    trainer_path: str = ""
    model_path: str = ""
    pretrained_gpt: str = ""
    pretrained_sovits: str = ""
    gpt: GptSettings = Field(default_factory=GptSettings)
    sovits: SovitsSettings = Field(default_factory=SovitsSettings)


class GptSovitsExecutionConfig(GptSovitsVersionConfig):
    train_manifest: str = ""
    val_manifest: str = ""


def parse_config(value) -> GptSovitsExecutionConfig:
    return GptSovitsExecutionConfig.model_validate(value.model_dump() if isinstance(value, BaseModel) else value)
