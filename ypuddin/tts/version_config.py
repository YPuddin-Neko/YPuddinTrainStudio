"""Version-owned TTS settings and their atomic on-disk envelope."""

from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .gpt_sovits.config import GptSovitsVersionConfig

TtsEngine = Literal["voxcpm1.5", "gpt-sovits-v5"]

AttentionTarget = Literal["q_proj", "v_proj", "k_proj", "o_proj"]
ProjectionTarget = Literal["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, strict=True, validate_default=True, revalidate_instances="always"
    )


class TtsVersionConfig(_Model):
    engine: Literal["voxcpm1.5"] = "voxcpm1.5"
    python_path: str = ""
    trainer_path: str = ""
    model_path: str = ""
    batch_size: int = Field(1, ge=1, le=1024)
    grad_accum_steps: int = Field(1, ge=1, le=1024)
    num_workers: int = Field(0, ge=0, le=64)
    preprocessing_num_workers: int = Field(1, ge=1, le=64)
    max_batch_tokens: int = Field(0, ge=0)
    num_iters: int = Field(1000, ge=1, le=10_000_000)
    learning_rate: float = Field(1e-4, gt=0, le=1)
    warmup_steps: int = Field(100, ge=0, le=10_000_000)
    weight_decay: float = Field(0.01, ge=0)
    max_grad_norm: float = Field(0.0, ge=0)
    max_steps: int = Field(0, ge=0, le=10_000_000)
    loss_diff_weight: float = Field(1.0, ge=0)
    loss_stop_weight: float = Field(1.0, ge=0)
    save_interval: int = Field(100, ge=1, le=10_000_000)
    valid_interval: int | None = Field(None, ge=1, le=10_000_000)
    log_interval: int = Field(1, ge=1, le=10_000_000)
    lora_rank: int = Field(32, ge=1, le=512)
    lora_alpha: int = Field(16, ge=1, le=4096)
    lora_dropout: float = Field(0.0, ge=0, lt=1)
    lora_enable_lm: bool = True
    lora_target_modules_lm: list[AttentionTarget] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"], max_length=4, json_schema_extra={"uniqueItems": True}
    )
    lora_enable_dit: bool = True
    lora_target_modules_dit: list[AttentionTarget] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"], max_length=4, json_schema_extra={"uniqueItems": True}
    )
    lora_enable_proj: bool = False
    lora_target_proj_modules: list[ProjectionTarget] = Field(
        default=["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"],
        max_length=3,
        json_schema_extra={"uniqueItems": True},
    )

    @field_validator("lora_target_modules_lm", "lora_target_modules_dit", "lora_target_proj_modules")
    @classmethod
    def unique_targets(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("LoRA 目标模块不能重复。")
        return value

    @model_validator(mode="after")
    def related_settings(self) -> TtsVersionConfig:
        if self.warmup_steps > min(self.num_iters, self.max_steps or self.num_iters):
            raise ValueError("预热步数不能大于训练迭代次数或学习率调度步数。")
        if 0 < self.max_batch_tokens < self.batch_size:
            raise ValueError("最大批次 token 数启用时不能小于批量大小。")
        if self.loss_diff_weight == 0 and self.loss_stop_weight == 0:
            raise ValueError("扩散损失和停止损失权重至少有一个大于 0。")
        components = (
            ("LM", self.lora_enable_lm, self.lora_target_modules_lm),
            ("DiT", self.lora_enable_dit, self.lora_target_modules_dit),
            ("投影层", self.lora_enable_proj, self.lora_target_proj_modules),
        )
        if not any(enabled for _, enabled, _ in components):
            raise ValueError("至少启用一个 LoRA 组件。")
        for label, enabled, targets in components:
            if enabled and not targets:
                raise ValueError(f"启用 {label} 时，LoRA 目标模块不能为空。")
        return self


TtsSavedConfig = Annotated[TtsVersionConfig | GptSovitsVersionConfig, Field(discriminator="engine")]


def default_config(engine: TtsEngine = "voxcpm1.5") -> TtsVersionConfig | GptSovitsVersionConfig:
    if engine == "gpt-sovits-v5":
        return GptSovitsVersionConfig()
    if engine != "voxcpm1.5":
        raise ValueError("不支持此语音引擎。")
    return TtsVersionConfig()


class TtsConfigEnvelope(_Model):
    schema_version: Literal[1] = 1
    revision: int = Field(ge=1)
    config: TtsSavedConfig

    @field_validator("config", mode="before")
    @classmethod
    def legacy_engine(cls, value: object) -> object:
        if isinstance(value, dict) and "engine" not in value:
            return {"engine": "voxcpm1.5", **value}
        return value

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("配置格式版本必须是整数。")
        return value


class TtsConfigScope(_Model):
    project_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)


class TtsConfigResponse(_Model):
    scope: TtsConfigScope
    revision: int = Field(ge=1)
    data_revision: int = Field(ge=1)
    config: TtsSavedConfig


class TtsConfigSaveBody(_Model):
    expected_revision: int = Field(ge=1)
    config: TtsSavedConfig

    @field_validator("config", mode="before")
    @classmethod
    def legacy_engine(cls, value: object) -> object:
        return TtsConfigEnvelope.legacy_engine(value)


class TtsConfigPathError(ValueError):
    """The configuration path was redirected or is not a regular file."""


_USE_DIR_FD = os.open in os.supports_dir_fd and os.rename in os.supports_dir_fd


def _linked(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _parent_identity(path: Path) -> tuple[tuple[Path, tuple[int, int]], ...]:
    identities = []
    for parent in (*reversed(path.parents), path):
        info = parent.lstat()
        if _linked(info):
            raise TtsConfigPathError("配置目录不能包含符号链接或目录重定向。")
        if not stat.S_ISDIR(info.st_mode):
            raise TtsConfigPathError("配置的父路径必须是目录。")
        identities.append((parent, _identity(info)))
    return tuple(identities)


@dataclass(frozen=True)
class _Location:
    path: Path
    parent_fd: int | None
    parents: tuple[tuple[Path, tuple[int, int]], ...]

    def verify_parents(self) -> None:
        if _parent_identity(self.path.parent) != self.parents:
            raise TtsConfigPathError("配置目录在读写期间发生变化，请重新打开版本。")

    def target_info(self) -> os.stat_result | None:
        try:
            info = (
                os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
                if self.parent_fd is not None
                else self.path.lstat()
            )
        except FileNotFoundError:
            return None
        if _linked(info) or not stat.S_ISREG(info.st_mode):
            raise TtsConfigPathError("配置必须是普通文件，不能使用符号链接或重定向。")
        return info

    def open(self, name: str, flags: int) -> int:
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        if self.parent_fd is not None:
            return os.open(name, flags, 0o600, dir_fd=self.parent_fd)
        return os.open(self.path.parent / name, flags, 0o600)

    def remove(self, name: str) -> None:
        if self.parent_fd is not None:
            os.unlink(name, dir_fd=self.parent_fd)
        else:
            self.verify_parents()
            os.unlink(self.path.parent / name)


@contextmanager
def _location(path: Path) -> Iterator[_Location]:
    if ".." in path.parts:
        raise TtsConfigPathError("配置路径不能包含上级目录跳转。")
    path = Path(os.path.abspath(path))
    parents = _parent_identity(path.parent)
    descriptor = None
    try:
        if _USE_DIR_FD:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            for parent, identity in parents:
                child = os.open(parent if descriptor is None else parent.name, flags, dir_fd=descriptor)
                if descriptor is not None:
                    os.close(descriptor)
                descriptor = child
                if _identity(os.fstat(descriptor)) != identity:
                    raise TtsConfigPathError("配置目录在读写期间发生变化，请重新打开版本。")
        location = _Location(path, descriptor, parents)
        location.verify_parents()
        yield location
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_envelope(path: Path) -> TtsConfigEnvelope:
    """Read an existing envelope without following links or creating defaults."""
    with _location(path) as location:
        before = location.target_info()
        if before is None:
            raise FileNotFoundError(location.path)
        descriptor = location.open(location.path.name, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            current = os.fstat(stream.fileno())
            if not stat.S_ISREG(current.st_mode) or _identity(current) != _identity(before):
                raise TtsConfigPathError("配置文件在读取期间发生变化，请重新读取。")
            contents = stream.read()
        location.verify_parents()
        after = location.target_info()
        if after is None or (_identity(after), after.st_size, after.st_mtime_ns) != (
            _identity(before), before.st_size, before.st_mtime_ns
        ):
            raise TtsConfigPathError("配置文件在读取期间发生变化，请重新读取。")
        return TtsConfigEnvelope.model_validate_json(contents)


def write_envelope(path: Path, envelope: TtsConfigEnvelope) -> None:
    """Replace one envelope atomically; the caller holds the version's CAS lock."""
    validated = TtsConfigEnvelope.model_validate(envelope)
    contents = json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with _location(path) as location:
        location.target_info()
        temporary = f".{location.path.name}.{uuid.uuid4().hex}.tmp"
        descriptor = location.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            location.verify_parents()
            location.target_info()
            if location.parent_fd is not None:
                os.replace(temporary, location.path.name, src_dir_fd=location.parent_fd, dst_dir_fd=location.parent_fd)
            else:
                os.replace(location.path.parent / temporary, location.path)
            location.verify_parents()
        finally:
            try:
                location.remove(temporary)
            except FileNotFoundError:
                pass
