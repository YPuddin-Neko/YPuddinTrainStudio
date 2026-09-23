"""Bounded Klein 4B/9B DTK arithmetic contracts; CPU checks are not GPU acceptance."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lokr import LoKr
from ypuddin.adapters.lora import LoRA
from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    DTK_KLEIN4B_ADAPTER_POLICY_IDS,
    DTK_KLEIN9B_ADAPTER_POLICY_IDS,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)
from ypuddin.train.preview_compute import preview_linear_compute
from ypuddin.train.trainer import Trainer


def config(algo="lora", strategy="ddp", variant="klein-base-4b"):
    return TrainConfig.model_validate(
        {
            "model": {
                "family": "flux2",
                "flux2_variant": variant,
                "dtype": "bf16",
                "attention": "sdpa",
            },
            "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": False},
            "adapter": {"algo": algo, "param_dtype": "fp32", "mode": "bypass"},
            "loop": {
                "gpu_count": 2,
                "distributed_strategy": strategy,
                "mixed_precision": "bf16",
                "deterministic": True,
            },
            "memory": {"base_precision": "bf16", "activation_checkpointing": "none", "allow_tf32": False},
            "dataset": {"text_encoding": "cached"},
        }
    )


@pytest.mark.parametrize(
    "variant,policy_ids",
    [("klein-base-4b", DTK_KLEIN4B_ADAPTER_POLICY_IDS), ("klein-base-9b", DTK_KLEIN9B_ADAPTER_POLICY_IDS)],
)
@pytest.mark.parametrize("algo,strategy", DTK_KLEIN4B_ADAPTER_POLICY_IDS)
def test_versioned_klein_recipe_and_exact_resume_guard(algo, strategy, variant, policy_ids):
    cfg = config(algo, strategy, variant)
    before = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert cfg.to_dict() == before
    assert policy["id"] == policy_ids[algo, strategy]
    assert effective.loop.mixed_precision == "bf16" and effective.model.dtype == "bf16"
    assert policy["preview_operator_components"] == ["backbone"]
    if algo == "lora" or strategy == "fsdp" or variant == "klein-base-9b":
        assert policy["operator_components"] == policy["trainable_components"] == ["backbone"]
    else:
        assert "operator_components" not in policy and "linear_backward_implementation" not in policy
    assert ("adapter_implementation" in policy) == (algo == "lora")
    assert ("fsdp_param_dtype" in policy) == (strategy == "fsdp")
    assert resolve_training_compute_config(effective, "cuda", "linux-dtk")[1] == policy
    validate_resume_compute_policy(policy, dict(policy))
    for old in [
        None,
        policy | {"id": "old"},
        policy | {"linear_backward_implementation": "old"},
        policy | {"preview_linear_implementation": "old"},
    ]:
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(policy, old)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model", "flux2_variant", "auto"),
        ("model", "dtype", "fp32"),
        ("loop", "gpu_count", 1),
        ("loop", "gpu_count", 3),
        ("loop", "mixed_precision", "fp16"),
        ("loop", "deterministic", False),
        ("training", "train_text_encoder", True),
        ("training", "train_backbone", False),
        ("training", "mode", "full"),
        ("adapter", "dora", True),
        ("adapter", "mode", "merged"),
        ("adapter", "param_dtype", "bf16"),
        ("memory", "base_precision", "fp8_e4m3"),
        ("memory", "compile", True),
        ("memory", "blocks_to_swap", 1),
    ],
)
@pytest.mark.parametrize("variant", ["klein-base-4b", "klein-base-9b"])
def test_unmeasured_scope_remains_native(section, field, value, variant):
    cfg = config(variant=variant)
    setattr(getattr(cfg, section), field, value)
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert policy is None and effective.to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "variant,policy_ids",
    [("klein-base-4b", DTK_KLEIN4B_ADAPTER_POLICY_IDS), ("klein-base-9b", DTK_KLEIN9B_ADAPTER_POLICY_IDS)],
)
@pytest.mark.parametrize("algo,strategy", DTK_KLEIN4B_ADAPTER_POLICY_IDS)
def test_actual_trainer_installs_backward_and_preview_contracts(
    monkeypatch, algo, strategy, variant, policy_ids
):
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", "linux-dtk")
    cfg = config(algo, strategy, variant)
    trainer = Trainer(cfg, device="cuda")
    adapter = (LoRA if algo == "lora" else LoKr)(8, 8, rank=2)
    model = nn.Sequential(AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), adapter, mode="bypass"))
    trainer.loaded = SimpleNamespace(backbone=model)
    if algo == "lora" or strategy == "fsdp" or variant == "klein-base-9b":
        trainer._install_backbone_adapter_compute_operators()
    trainer._validate_training_compute_policy()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.randn(2, 3, 8))
    output.float().square().sum().backward()
    assert output.dtype == torch.bfloat16
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16),
        preview_linear_compute(trainer.compute_policy, model),
    ):
        preview = model(torch.randn(2, 3, 8))
    assert preview.dtype == torch.bfloat16 and torch.isfinite(preview).all()
    restore = getattr(trainer, "_text_adapter_compute_restore", None)
    if restore is not None:
        restore()


def test_klein_lokr_preview_preserves_native_training_and_disables_hooks_afterward():
    from Test.tests.unit.test_fp16_preview_compute import MatmulDtypes

    torch.manual_seed(381)
    policy = resolve_training_compute_config(config("lokr"), "cuda", "linux-dtk")[1]
    model = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoKr(8, 8, rank=2), mode="bypass")
    native = copy.deepcopy(model)
    x = torch.randn(2, 3, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        before = native(x)
        with preview_linear_compute(policy, model):
            after = model(x)
    before.float().square().sum().backward()
    after.float().square().sum().backward()
    assert torch.equal(before, after)
    assert all(
        torch.equal(a.grad, b.grad)
        for a, b in zip(native.adapter.parameters(), model.adapter.parameters(), strict=True)
    )
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        with preview_linear_compute(policy, model), MatmulDtypes() as traced:
            model.base(x)
        with MatmulDtypes() as restored:
            model.base(x)
    assert set(traced.dtypes) == {torch.float32}
    assert set(restored.dtypes) == {torch.bfloat16}
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("variant", ["klein-base-4b", "klein-base-9b"])
@pytest.mark.parametrize(
    "profile,device", [("linux-cuda", "cuda"), ("windows-cuda", "cuda"), ("linux-dtk", "cpu")]
)
def test_klein_dtk_policy_does_not_change_other_devices(variant, profile, device):
    cfg = config(variant=variant)
    effective, policy = resolve_training_compute_config(cfg, device, profile)
    assert policy is None and effective.to_dict() == cfg.to_dict()
