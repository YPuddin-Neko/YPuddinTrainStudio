"""Change a speech version's model type through its existing revisioned configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ypuddin.tts.gpt_sovits.config import GptSovitsVersionConfig
from ypuddin.tts.version_config import (
    TtsConfigResponse,
    TtsConfigSaveBody,
    TtsEngine,
    TtsSavedConfig,
    default_config,
)

from .errors import ApiError
from .project_deletion import deleting
from .tts_projects import _read, save_config
from .versions import assert_version_writable


class TtsEngineChangeBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, revalidate_instances="always")

    expected_revision: int = Field(ge=1)
    engine: TtsEngine
    variant: Literal["v5dev", "v5turbo"] | None = None

    @field_validator("variant")
    @classmethod
    def variant_for_engine(cls, value: str | None, info: ValidationInfo) -> str | None:
        if info.data.get("engine") == "voxcpm1.5":
            raise ValueError("VoxCPM 不使用模型变体，请移除 variant 字段。")
        return value


def select_engine_config(config: TtsSavedConfig, engine: TtsEngine,
                         variant: Literal["v5dev", "v5turbo"] | None = None) -> TtsSavedConfig:
    if config.engine != engine:
        config = default_config(engine)
    if isinstance(config, GptSovitsVersionConfig) and variant not in {None, config.variant}:
        config = GptSovitsVersionConfig.model_validate({
            **config.model_dump(), "variant": variant,
            "model_path": "", "pretrained_gpt": "", "pretrained_sovits": "",
        })
    return config


def change_engine(c: Any, pid: str, vid: str, body: TtsEngineChangeBody) -> TtsConfigResponse:
    with c.db.lock:
        c.require_project_type(pid, "tts")
        if deleting(c, pid):
            raise ApiError("项目正在删除。", code="project.deleting", status=409)
        assert_version_writable(c, pid, vid)
        saved = _read(c, pid, vid)
        if saved.revision != body.expected_revision:
            raise ApiError("配置已更新，请重新读取后保存。", code="tts.config_conflict", status=409,
                           details={"current_revision": saved.revision})
        config = select_engine_config(saved.config, body.engine, body.variant)
        return save_config(c, pid, vid, TtsConfigSaveBody(expected_revision=body.expected_revision, config=config))
