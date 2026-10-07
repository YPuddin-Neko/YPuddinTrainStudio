"""Resolve saved TTS recipes into the complete upstream execution settings."""

from __future__ import annotations

from typing import Any

from .config import TtsConfig
from .gpt_sovits.config import GptSovitsExecutionConfig, GptSovitsVersionConfig
from .version_config import TtsVersionConfig


class TtsExecutionConfig(TtsVersionConfig):
    train_manifest: str = ""
    val_manifest: str = ""


def parse_execution_config(value: TtsConfig | TtsVersionConfig | GptSovitsVersionConfig | dict[str, Any]) -> TtsExecutionConfig | GptSovitsExecutionConfig:
    if isinstance(value, (TtsConfig, TtsVersionConfig, GptSovitsVersionConfig)):
        value = value.model_dump()
    if value.get("engine") == "gpt-sovits-v5":
        return GptSovitsExecutionConfig.model_validate(value)
    return TtsExecutionConfig.model_validate(value)


def lora_settings(config: TtsExecutionConfig) -> dict[str, Any]:
    return {
        "enable_lm": config.lora_enable_lm, "enable_dit": config.lora_enable_dit,
        "enable_proj": config.lora_enable_proj, "r": config.lora_rank,
        "alpha": config.lora_alpha, "dropout": config.lora_dropout,
        "target_modules_lm": list(config.lora_target_modules_lm),
        "target_modules_dit": list(config.lora_target_modules_dit),
        "target_proj_modules": list(config.lora_target_proj_modules),
    }
