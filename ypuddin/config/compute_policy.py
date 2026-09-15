"""Resolve versioned training compute settings without importing a tensor runtime."""

from __future__ import annotations

from typing import Literal, TypedDict

from .schema import TrainConfig

DTK_FULL_FP32_MATH_POLICY_ID = "dtk-full-fp32-math-v1"


class TrainingComputePolicy(TypedDict):
    id: str
    mixed_precision: Literal["no"]
    allow_tf32: Literal[False]
    attention: Literal["sdpa"]
    sdpa_backend: Literal["math"]


def resolve_training_compute_config(
    cfg: TrainConfig, device_type: str | None, profile: str
) -> tuple[TrainConfig, TrainingComputePolicy | None]:
    """Return an independent effective configuration and its resume identity.

    The runtime must still apply deterministic algorithms and the math SDPA
    backend. This function only resolves the shared planning/training recipe;
    it neither initializes devices nor changes process-wide backend settings.
    """
    effective = cfg.model_copy(deep=True)
    if not (
        profile == "linux-dtk"
        and device_type == "cuda"
        and cfg.training.mode == "full"
        and cfg.training.train_backbone
        and cfg.loop.deterministic
        and cfg.model.family in {"anima", "sdxl", "krea2"}
    ):
        return effective, None

    effective.loop.mixed_precision = "no"
    effective.memory.allow_tf32 = False
    effective.model.attention = "sdpa"
    policy: TrainingComputePolicy = {
        "id": DTK_FULL_FP32_MATH_POLICY_ID,
        "mixed_precision": "no",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
    }
    return effective, policy


def validate_resume_compute_policy(expected: dict | None, saved: dict | None) -> None:
    """Reject a changed compute recipe instead of migrating optimizer state."""
    if expected == saved:
        return
    if expected is not None and saved is None:
        raise ValueError(
            "此训练状态未记录当前计算配方，不能按新的可复现计算策略恢复。"
            "请使用保存该状态的原版本继续，或从导出权重新建训练；不会自动迁移优化器状态。"
        )
    raise ValueError(
        "训练状态的计算配方与当前设置不同，不能进行严格续训。"
        "请恢复原计算策略并使用对应版本继续，或从导出权重新建训练。"
    )
