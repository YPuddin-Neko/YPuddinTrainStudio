"""Versioned DTK compute recipes are pure configuration, not runtime detection."""

import pytest

from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    DTK_FULL_FP32_MATH_POLICY_ID,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)


def _config(family="anima", *, mode="full", train_backbone=True, deterministic=True):
    return TrainConfig.model_validate(
        {
            "model": {"family": family, "attention": "flash_attn"},
            "training": {
                "mode": mode,
                "train_backbone": train_backbone,
                "train_text_encoder": not train_backbone,
            },
            "loop": {"deterministic": deterministic, "mixed_precision": "bf16"},
            "memory": {"allow_tf32": True},
        }
    )


@pytest.mark.parametrize("family", ["anima", "sdxl", "krea2"])
@pytest.mark.parametrize("attention", ["auto", "sdpa", "flash_attn", "xformers"])
def test_dtk_full_recipe_resolves_explicit_effective_config_without_mutation(family, attention):
    cfg = _config(family)
    cfg.model.attention = attention
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")

    expected = cfg.to_dict()
    expected["loop"]["mixed_precision"] = "no"
    expected["memory"]["allow_tf32"] = False
    expected["model"]["attention"] = "sdpa"
    assert effective.to_dict() == expected
    assert cfg.to_dict() == original
    assert policy == {
        "id": DTK_FULL_FP32_MATH_POLICY_ID,
        "mixed_precision": "no",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
    }
    repeated, repeated_policy = resolve_training_compute_config(effective, "cuda", "linux-dtk")
    assert repeated.to_dict() == expected
    assert repeated_policy == policy
    assert repeated_policy is not policy
    assert repeated is not effective


@pytest.mark.parametrize(
    "device,profile,family,mode,backbone,deterministic",
    [
        ("cpu", "linux-dtk", "anima", "full", True, True),
        ("mps", "linux-dtk", "anima", "full", True, True),
        (None, "linux-dtk", "anima", "full", True, True),
        ("cuda", "linux-cuda", "anima", "full", True, True),
        ("cuda", "windows-cuda", "anima", "full", True, True),
        ("cuda", "legacy", "anima", "full", True, True),
        ("cuda", "linux-dtk", "anima", "adapter", True, True),
        ("cuda", "linux-dtk", "flux2", "full", True, True),
        ("cuda", "linux-dtk", "toy", "full", True, True),
        ("cuda", "linux-dtk", "anima", "full", False, True),
        ("cuda", "linux-dtk", "anima", "full", True, False),
    ],
)
def test_other_training_keeps_original_behavior_and_independent_nested_config(
    device, profile, family, mode, backbone, deterministic
):
    cfg = _config(family, mode=mode, train_backbone=backbone, deterministic=deterministic)
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, device, profile)
    assert effective.to_dict() == original
    assert policy is None
    effective.loop.mixed_precision = "no"
    effective.model.attention = "sdpa"
    effective.optimizer.args["custom_argument"] = {"value": 1}
    assert cfg.to_dict() == original


def test_matching_and_legacy_unversioned_policies_can_resume():
    _, policy = resolve_training_compute_config(_config(), "cuda", "linux-dtk")
    validate_resume_compute_policy(policy, dict(policy))
    validate_resume_compute_policy(None, None)


def test_old_state_cannot_be_silently_upgraded_to_new_compute_recipe():
    _, policy = resolve_training_compute_config(_config(), "cuda", "linux-dtk")
    with pytest.raises(ValueError, match="原版本继续.*不会自动迁移优化器状态"):
        validate_resume_compute_policy(policy, None)


@pytest.mark.parametrize(
    "changed",
    [
        {"id": "dtk-full-fp32-math-v0"},
        {"mixed_precision": "bf16"},
        {"allow_tf32": True},
        {"attention": "flash_attn"},
        {"sdpa_backend": "flash"},
    ],
)
def test_changed_compute_policy_cannot_resume(changed):
    _, policy = resolve_training_compute_config(_config(), "cuda", "linux-dtk")
    with pytest.raises(ValueError, match="计算配方与当前设置不同"):
        validate_resume_compute_policy(policy, policy | changed)


def test_versioned_state_cannot_resume_with_recipe_disabled():
    _, policy = resolve_training_compute_config(_config(), "cuda", "linux-dtk")
    with pytest.raises(ValueError, match="计算配方与当前设置不同"):
        validate_resume_compute_policy(None, policy)
