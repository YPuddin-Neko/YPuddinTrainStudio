"""Versioned DTK compute recipes are pure configuration, not runtime detection."""

import pytest

from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID,
    BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
    DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID,
    DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID,
    DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID,
    DTK_BACKBONE_ADAPTER_POLICY_IDS,
    DTK_FULL_FP32_MATH_POLICY_ID,
    DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
    DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
    DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID,
    FP32_CONV_IMPLEMENTATION_ID,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)


def _config(family="anima", *, mode="full", train_backbone=True, deterministic=True):
    return TrainConfig.model_validate(
        {
            "model": {"family": family, "attention": "flash_attn", "dtype": "bf16"},
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
    cfg.loop.mixed_precision = "fp16"  # Existing FP32 recipe remains available.
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
    cfg.adapter.algo = "lora"
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
    cfg = _config()
    cfg.loop.mixed_precision = "fp16"
    _, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    with pytest.raises(ValueError, match="计算配方与当前设置不同"):
        validate_resume_compute_policy(policy, policy | changed)


def test_versioned_state_cannot_resume_with_recipe_disabled():
    _, policy = resolve_training_compute_config(_config(), "cuda", "linux-dtk")
    with pytest.raises(ValueError, match="计算配方与当前设置不同"):
        validate_resume_compute_policy(None, policy)


def _krea_bf16_config():
    cfg = _config("krea2")
    cfg.loop.distributed_strategy = "fsdp"
    cfg.loop.gpu_count = 2
    return cfg


def test_krea_bf16_sharding_policy_keeps_native_compute_and_is_idempotent():
    cfg = _krea_bf16_config()
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert effective.loop.mixed_precision == "bf16"
    assert not effective.memory.allow_tf32
    assert effective.model.attention == "sdpa"
    assert policy == {
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
    again, same = resolve_training_compute_config(effective, "cuda", "linux-dtk")
    assert again.to_dict() == effective.to_dict() and same == policy
    assert cfg.to_dict() == original


@pytest.mark.parametrize(
    "section,field,value,expected_id",
    [
        ("loop", "gpu_count", 1, DTK_FULL_FP32_MATH_POLICY_ID),
        ("loop", "distributed_strategy", "ddp", DTK_FULL_FP32_MATH_POLICY_ID),
        ("loop", "mixed_precision", "no", DTK_FULL_FP32_MATH_POLICY_ID),
        ("loop", "mixed_precision", "fp16", DTK_FULL_FP32_MATH_POLICY_ID),
        ("training", "train_text_encoder", True, DTK_FULL_FP32_MATH_POLICY_ID),
        ("model", "family", "anima", DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID),
        ("loop", "deterministic", False, None),
        ("training", "mode", "adapter", None),
        ("training", "train_backbone", False, None),
    ],
)
def test_krea_bf16_policy_never_expands_unverified_scope(section, field, value, expected_id):
    cfg = _krea_bf16_config()
    setattr(getattr(cfg, section), field, value)
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert (policy or {}).get("id") == expected_id
    if expected_id is None:
        assert effective.to_dict() == original
    else:
        assert effective.loop.mixed_precision == (
            "no" if expected_id == DTK_FULL_FP32_MATH_POLICY_ID else "bf16"
        )


@pytest.mark.parametrize(
    "device,profile", [("cpu", "linux-dtk"), ("cuda", "windows-cuda"), ("cuda", "linux-cuda")]
)
def test_krea_bf16_policy_does_not_claim_non_dtk_validation(device, profile):
    cfg = _krea_bf16_config()
    effective, policy = resolve_training_compute_config(cfg, device, profile)
    assert policy is None and effective.to_dict() == cfg.to_dict()


def test_krea_bf16_checkpoint_identity_rejects_external_candidate_fp32_and_changed_operators():
    _, policy = resolve_training_compute_config(_krea_bf16_config(), "cuda", "linux-dtk")
    _, old_fp32 = resolve_training_compute_config(_config("krea2"), "cuda", "linux-dtk")
    validate_resume_compute_policy(policy, dict(policy))
    for saved in (
        None,
        old_fp32,
        policy | {"id": "external-bf16-candidate"},
        {key: value for key, value in policy.items() if key != "linear_backward_implementation"},
        policy | {"linear_backward_implementation": "other-algorithm"},
        policy | {"fsdp_reduce_dtype": "bfloat16"},
    ):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(policy, saved)


def test_sdxl_single_gpu_bf16_conv_and_linear_policy_is_explicit_and_idempotent():
    cfg = _config("sdxl")
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy["id"] == DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID
    assert policy["conv_forward"] == "fp32-output-bf16"
    assert policy["conv_implementation"] == FP32_CONV_IMPLEMENTATION_ID
    assert policy["linear_backward_implementation"] == BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID
    assert "fsdp_param_dtype" not in policy
    assert effective.loop.mixed_precision == "bf16" and not effective.memory.allow_tf32
    assert effective.model.attention == "sdpa"
    again, repeated = resolve_training_compute_config(effective, "cuda", "linux-dtk")
    assert again.to_dict() == effective.to_dict() and repeated == policy
    assert cfg.to_dict() == original
    for saved in (
        None,
        policy | {"conv_implementation": "old-candidate"},
        policy | {"conv_forward": "native-bf16"},
    ):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(policy, saved)


@pytest.mark.parametrize(
    "section,field,value,expected_id",
    [
        ("loop", "gpu_count", 2, DTK_FULL_FP32_MATH_POLICY_ID),
        ("training", "train_text_encoder", True, DTK_FULL_FP32_MATH_POLICY_ID),
        ("loop", "mixed_precision", "no", DTK_FULL_FP32_MATH_POLICY_ID),
        ("loop", "mixed_precision", "fp16", DTK_FULL_FP32_MATH_POLICY_ID),
        ("training", "train_backbone", False, None),
        ("loop", "deterministic", False, None),
    ],
)
def test_sdxl_bf16_recipe_stays_inside_verified_scope(section, field, value, expected_id):
    cfg = _config("sdxl")
    setattr(getattr(cfg, section), field, value)
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert (policy or {}).get("id") == expected_id
    assert effective.loop.mixed_precision == ("no" if expected_id else cfg.loop.mixed_precision)


@pytest.mark.parametrize("mode", ["auto", "bypass"])
@pytest.mark.parametrize("gpu_count", [1, 2])
def test_sdxl_lokr_bypass_recipe_with_homogeneous_rules(mode, gpu_count):
    cfg = _config("sdxl", mode="adapter")
    cfg.adapter.mode = mode
    cfg.loop.gpu_count = gpu_count
    cfg.loop.distributed_strategy = "ddp"
    from ypuddin.config.schema import AdapterRule

    cfg.adapter.rules = [
        AdapterRule(match="*.q", algo="lokr"),
        AdapterRule(match="*.k", algo=None),
        AdapterRule(match="*.v", algo="none"),
    ]
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy["id"] == DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID
    assert effective.adapter == cfg.adapter and effective.loop.mixed_precision == "bf16"


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("adapter", "mode", "merged"),
        ("adapter", "dora", True),
        ("adapter", "param_dtype", "bf16"),
        ("memory", "base_precision", "fp8_e4m3"),
        ("training", "train_text_encoder", True),
        ("loop", "deterministic", False),
    ],
)
def test_sdxl_other_adapter_paths_keep_existing_behavior(section, field, value):
    cfg = _config("sdxl", mode="adapter")
    setattr(getattr(cfg, section), field, value)
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy is None and effective.to_dict() == cfg.to_dict()


def test_sdxl_mixed_algorithm_rules_never_claim_lokr_policy():
    from ypuddin.config.schema import AdapterRule

    cfg = _config("sdxl", mode="adapter")
    cfg.adapter.rules = [AdapterRule(match="*", algo="lora")]
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy is None and effective.to_dict() == cfg.to_dict()


def test_sdxl_adapter_sharding_does_not_claim_ddp_bf16_policy():
    cfg = _config("sdxl", mode="adapter")
    cfg.loop.gpu_count = 2
    cfg.loop.distributed_strategy = "fsdp"
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy["id"] == DTK_BACKBONE_ADAPTER_POLICY_IDS[("sdxl", "lokr", "fsdp")]
    assert policy["fsdp_param_dtype"] == "bfloat16"
    assert effective.loop.mixed_precision == "bf16"


def test_sdxl_fsdp_has_distinct_bf16_gather_identity_and_rejects_unsharded_state():
    cfg = _config("sdxl")
    _, single_policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    cfg.loop.gpu_count = 2
    cfg.loop.distributed_strategy = "fsdp"
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy == single_policy | {
        "id": DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID,
        "fsdp_param_dtype": "bfloat16",
        "fsdp_reduce_dtype": "float32",
    }
    assert effective.loop.mixed_precision == "bf16"
    assert cfg.to_dict() == original
    repeated, repeated_policy = resolve_training_compute_config(effective, "cuda", "linux-dtk")
    assert repeated.to_dict() == effective.to_dict() and repeated_policy == policy
    for saved in (
        None,
        single_policy,
        policy | {"fsdp_param_dtype": "float32"},
        policy | {"id": "diagnostic-dtk-sdxl-dual-bf16-conv-linear-v1"},
    ):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(policy, saved)


@pytest.mark.parametrize("mode", ["full", "adapter"])
@pytest.mark.parametrize("checkpointing", ["none", "block"])
def test_anima_bf16_linear_computation_preserves_rounding_and_has_own_resume_identity(mode, checkpointing):
    cfg = _config("anima", mode=mode)
    cfg.memory.activation_checkpointing = checkpointing
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert effective.loop.mixed_precision == "bf16"
    assert not effective.memory.allow_tf32 and effective.model.attention == "sdpa"
    assert policy == {
        "id": DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID,
        "mixed_precision": "bf16",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
        "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "linear_backward": "fp32-contractions-grad-original-dtype",
        "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
    }
    again, same = resolve_training_compute_config(effective, "cuda", "linux-dtk")
    assert again.to_dict() == effective.to_dict() and same == policy
    assert cfg.to_dict() == original
    for saved in (
        None,
        policy | {"linear_forward": "native-bf16"},
        policy | {"linear_backward_implementation": BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID},
        policy | {"id": "external-anima-candidate"},
    ):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(policy, saved)


@pytest.mark.parametrize("mode,strategy", [("full", "fsdp"), ("adapter", "ddp")])
@pytest.mark.parametrize("gpu_count", [1, 2])
def test_anima_reentrant_checkpoint_cannot_mix_native_first_forward_with_fp32_recompute(
    mode, strategy, gpu_count
):
    cfg = _config("anima", mode=mode)
    cfg.loop.gpu_count = gpu_count
    cfg.loop.distributed_strategy = strategy
    cfg.memory.activation_checkpointing = "unsloth"
    with pytest.raises(ValueError, match="Unsloth"):
        resolve_training_compute_config(cfg, "cuda", "linux-dtk")


@pytest.mark.parametrize("mode,strategy", [("full", "ddp"), ("adapter", "fsdp")])
def test_anima_multi_gpu_policy_rejects_other_training_strategies(mode, strategy):
    cfg = _config("anima", mode=mode)
    cfg.loop.gpu_count = 2
    cfg.loop.distributed_strategy = strategy
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    if mode == "full":
        assert policy["id"] == DTK_FULL_FP32_MATH_POLICY_ID
        assert effective.loop.mixed_precision == "no"
    else:
        assert policy["id"] == "dtk-anima-backbone-lokr-fsdp-bf16-compute-preview-v2"
        assert effective.loop.mixed_precision == "bf16"


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("adapter", "mode", "merged"),
        ("adapter", "dora", True),
        ("adapter", "param_dtype", "bf16"),
        ("memory", "base_precision", "fp8_e4m3"),
        ("training", "train_text_encoder", True),
        ("loop", "mixed_precision", "fp16"),
    ],
)
@pytest.mark.parametrize("gpu_count", [1, 2])
def test_anima_other_adapter_paths_do_not_claim_verified_linear_recipe(section, field, value, gpu_count):
    cfg = _config("anima", mode="adapter")
    cfg.loop.gpu_count = gpu_count
    cfg.loop.distributed_strategy = "ddp"
    setattr(getattr(cfg, section), field, value)
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy is None and effective.to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "mode,strategy,policy_id",
    [
        ("adapter", "ddp", DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID),
        ("full", "fsdp", DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID),
    ],
)
def test_anima_multi_gpu_compute_recipe_has_distinct_resume_boundaries(mode, strategy, policy_id):
    cfg = _config("anima", mode=mode)
    _, single = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    cfg.loop.gpu_count = 2
    cfg.loop.distributed_strategy = strategy
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    expected = single | {"id": policy_id}
    if strategy == "fsdp":
        expected.update(fsdp_param_dtype="bfloat16", fsdp_reduce_dtype="float32")
    assert policy == expected and effective.loop.mixed_precision == "bf16"
    assert cfg.to_dict() == original
    again, repeated = resolve_training_compute_config(effective, "cuda", "linux-dtk")
    assert again.to_dict() == effective.to_dict() and repeated == policy
    for saved in (
        None,
        single,
        policy | {"id": "diagnostic-anima-dual"},
        policy | {"linear_backward_implementation": BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID},
        policy | {"fsdp_reduce_dtype": "bfloat16"},
    ):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(policy, saved)


@pytest.mark.parametrize("mode,strategy", [("adapter", "ddp"), ("full", "fsdp")])
@pytest.mark.parametrize(
    "device,profile", [("cuda", "linux-cuda"), ("cuda", "windows-cuda"), ("cpu", "linux-dtk")]
)
def test_anima_multi_gpu_recipe_does_not_change_other_runtimes(mode, strategy, device, profile):
    cfg = _config("anima", mode=mode)
    cfg.loop.gpu_count, cfg.loop.distributed_strategy = 2, strategy
    effective, policy = resolve_training_compute_config(cfg, device, profile)
    assert policy is None and effective.to_dict() == cfg.to_dict()
