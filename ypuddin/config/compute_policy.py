"""Resolve versioned training compute settings without importing a tensor runtime."""

from __future__ import annotations

from typing import Literal, TypedDict

from .schema import TrainConfig

DTK_FULL_FP32_MATH_POLICY_ID = "dtk-full-fp32-math-v1"
DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID = "dtk-krea2-fsdp-bf16-linear-fp32-backward-v1"
DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID = "dtk-sdxl-bf16-conv-fp32-linear-backward-v1"
DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID = "dtk-sdxl-fsdp-bf16-conv-fp32-linear-backward-v1"
BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID = "linear-bf16-forward-fp32-backward-v1"
FP32_CONV_IMPLEMENTATION_ID = "conv2d-fp32-output-bf16-v1"


class _RequiredTrainingComputePolicy(TypedDict):
    id: str
    mixed_precision: Literal["no", "bf16"]
    allow_tf32: Literal[False]
    attention: Literal["sdpa"]
    sdpa_backend: Literal["math"]


class TrainingComputePolicy(_RequiredTrainingComputePolicy, total=False):
    linear_forward: Literal["native-bf16"]
    linear_backward: Literal["fp32-contractions-grad-original-dtype"]
    linear_backward_implementation: str
    fsdp_param_dtype: Literal["bfloat16"]
    fsdp_reduce_dtype: Literal["float32"]
    conv_forward: Literal["fp32-output-bf16"]
    conv_implementation: str


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
        and cfg.training.train_backbone
        and cfg.loop.deterministic
        and cfg.model.family in {"anima", "sdxl", "krea2"}
    ):
        return effective, None

    sdxl_lokr = (
        cfg.model.family == "sdxl"
        and cfg.training.mode == "adapter"
        and not cfg.training.train_text_encoder
        and (cfg.loop.gpu_count == 1 or cfg.loop.distributed_strategy == "ddp")
        and cfg.loop.mixed_precision == "bf16"
        and cfg.adapter.algo == "lokr"
        and cfg.adapter.mode in {"auto", "bypass"}
        and not cfg.adapter.dora
        and cfg.adapter.param_dtype == "fp32"
        and all(rule.algo in {None, "lokr", "none"} for rule in cfg.adapter.rules)
        and not cfg.memory.base_precision.startswith("fp8")
    )
    sdxl_sharded_full = (
        cfg.model.family == "sdxl"
        and cfg.training.mode == "full"
        and cfg.loop.gpu_count >= 2
        and cfg.loop.distributed_strategy == "fsdp"
    )
    if cfg.training.mode != "full" and not sdxl_lokr:
        return effective, None

    effective.memory.allow_tf32 = False
    effective.model.attention = "sdpa"
    if (
        cfg.model.family == "krea2"
        and not cfg.training.train_text_encoder
        and cfg.loop.distributed_strategy == "fsdp"
        and cfg.loop.gpu_count >= 2
        and cfg.loop.mixed_precision == "bf16"
    ):
        return effective, {
            "id": DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
            "mixed_precision": "bf16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "linear_forward": "native-bf16",
            "linear_backward": "fp32-contractions-grad-original-dtype",
            "linear_backward_implementation": BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID,
            "fsdp_param_dtype": "bfloat16",
            "fsdp_reduce_dtype": "float32",
        }
    if (
        cfg.model.family == "sdxl"
        and not cfg.training.train_text_encoder
        and (cfg.loop.gpu_count == 1 or sdxl_lokr or sdxl_sharded_full)
        and cfg.loop.mixed_precision == "bf16"
    ):
        sdxl_policy: TrainingComputePolicy = {
            "id": DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID
            if sdxl_sharded_full
            else DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
            "mixed_precision": "bf16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "linear_forward": "native-bf16",
            "linear_backward": "fp32-contractions-grad-original-dtype",
            "linear_backward_implementation": BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID,
            "conv_forward": "fp32-output-bf16",
            "conv_implementation": FP32_CONV_IMPLEMENTATION_ID,
        }
        if sdxl_sharded_full:
            # FSDP gathers BF16 parameters before the FP32 convolution. This
            # rounding boundary differs from the unsharded FP32 master weights.
            sdxl_policy.update(fsdp_param_dtype="bfloat16", fsdp_reduce_dtype="float32")
        return effective, sdxl_policy
    effective.loop.mixed_precision = "no"
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
