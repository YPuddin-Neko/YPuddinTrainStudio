"""Versioned TE component scope, installation and resume rejection contracts."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lora import LoRA
from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
    DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
    DTK_SDXL_LONG_TEXT_POLICY_ID,
    DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS,
    DTK_TEXT_LORA_POLICY_IDS,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)
from ypuddin.train import Trainer


def config(family="anima", joint=False, count=1):
    return TrainConfig.model_validate(
        {
            "model": {"family": family, "attention": "xformers"},
            "training": {"mode": "adapter", "train_backbone": joint, "train_text_encoder": True},
            "adapter": {"algo": "lora", "rank": 2, "param_dtype": "fp32", "mode": "auto"},
            "loop": {"mixed_precision": "bf16", "deterministic": True, "gpu_count": count},
            "dataset": {"text_encoding": "online"},
            "memory": {"allow_tf32": True, "offload_text_encoder": False},
        }
    )


@pytest.mark.parametrize("family", ["anima", "sdxl", "krea2"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_component_policy_is_explicit_idempotent_and_does_not_mutate_request(family, joint, count):
    cfg = config(family, joint, count)
    before = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert cfg.to_dict() == before
    assert policy["id"] == DTK_TEXT_LORA_POLICY_IDS[family]
    assert policy["linear_backward_implementation"] == BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
    assert policy["text_linear_backward_implementation"] == BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
    text = ["text_encoder"] + (["text_encoder_2"] if family == "sdxl" else [])
    assert policy["operator_components"] == sorted(["backbone", *text])
    assert policy["trainable_components"] == sorted(text + (["backbone"] if joint else []))
    assert policy["distributed_strategy"] == ("single" if count == 1 else "ddp")
    assert effective.loop.mixed_precision == "bf16" and not effective.memory.allow_tf32
    assert effective.model.attention == "sdpa"
    again, same = resolve_training_compute_config(effective, "cuda", "linux-dtk")
    assert again.to_dict() == effective.to_dict() and same == policy
    if family == "sdxl":
        assert policy["sdxl_max_token_length"] == 75
        assert policy["conv_forward"] == "fp32-output-bf16"
    else:
        assert "conv_forward" not in policy and "sdxl_max_token_length" not in policy


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("adapter", "algo", "lokr"),
        ("adapter", "algo", "loha"),
        ("adapter", "algo", "full"),
        ("adapter", "mode", "merged"),
        ("adapter", "param_dtype", "bf16"),
        ("adapter", "dora", True),
        ("memory", "base_precision", "fp8_e4m3"),
        ("memory", "activation_checkpointing", "unsloth"),
        ("memory", "compile", True),
        ("memory", "blocks_to_swap", 1),
        ("memory", "offload_text_encoder", True),
        ("dataset", "text_encoding", "cached"),
        ("loop", "distributed_strategy", "fsdp"),
        ("loop", "deterministic", False),
        ("loop", "mixed_precision", "fp16"),
        ("training", "train_text_encoder", False),
    ],
)
def test_text_policy_does_not_claim_unsupported_scope(section, key, value):
    cfg = config(joint=True)
    setattr(getattr(cfg, section), key, value)
    _, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    if key == "train_text_encoder":
        assert policy["id"] not in DTK_TEXT_LORA_POLICY_IDS.values()
        assert policy["operator_components"] == ["backbone"]
    else:
        assert policy is None


@pytest.mark.parametrize(
    "device,profile", [("cpu", "linux-dtk"), ("cuda", "windows-cuda"), ("cuda", "linux-cuda")]
)
def test_other_platforms_remain_native(device, profile):
    cfg = config()
    effective, policy = resolve_training_compute_config(cfg, device, profile)
    assert policy is None and effective.to_dict() == cfg.to_dict()


def adapted():
    return AdaptedLinear(FrozenLinear.from_linear(nn.Linear(4, 4)), LoRA(4, 4, rank=2))


def tiny_trainer(tmp_path, monkeypatch, family="anima", joint=False):
    import ypuddin.train.trainer as module

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    cfg = config(family, joint)
    cfg.checkpoint.output_dir = str(tmp_path)
    trainer = Trainer(cfg, device="cuda:0")  # Metadata only; real modules remain on CPU.
    backbone = nn.Sequential(adapted() if joint else nn.Linear(4, 4))
    if family == "sdxl":
        backbone.add_module("conv", nn.Conv2d(1, 1, 1))
    text = {
        name: nn.Sequential(adapted())
        for name in trainer.compute_policy["operator_components"]
        if name != "backbone"
    }
    encoder = SimpleNamespace(
        trainable_modules=lambda: text, encode=lambda *_: pytest.fail("encoded before guard")
    )
    trainer.loaded = SimpleNamespace(backbone=backbone, text=encoder)
    monkeypatch.setattr(backbone, "to", lambda _: backbone)
    return trainer, text


@pytest.mark.parametrize("family", ["anima", "sdxl", "krea2"])
@pytest.mark.parametrize("joint", [False, True])
def test_trainer_installs_actual_text_and_frozen_backbone_before_optimizer(
    tmp_path, monkeypatch, family, joint
):
    trainer, text = tiny_trainer(tmp_path, monkeypatch, family, joint)
    ids = {
        name: id(p)
        for component, m in trainer._text_compute_modules().items()
        for name, p in ((component + "." + n, p) for n, p in m.named_parameters())
    }
    with pytest.raises(ValueError, match="未完整安装"):
        trainer._validate_training_compute_policy()
    trainer._place_training_model()
    try:
        trainer._validate_training_compute_policy()
        assert trainer._text_adapter_operator_counts["backbone"]["lora"] == int(joint)
        assert all(trainer._text_adapter_operator_counts[name]["lora"] == 1 for name in text)
        after = {
            name: id(p)
            for component, m in trainer._text_compute_modules().items()
            for name, p in ((component + "." + n, p) for n, p in m.named_parameters())
        }
        assert after == ids
        with pytest.raises(ValueError, match="重复安装"):
            trainer._install_text_adapter_compute_operators()
        text["text_encoder"][0].adapter.delta_apply = lambda x: x
        # Crucially _text_cond validates before encoding, outside backbone autocast.
        with pytest.raises(ValueError, match="收缩策略"):
            trainer._text_cond(["caption"])
    finally:
        trainer._text_adapter_compute_restore()


@pytest.mark.parametrize(
    "changed",
    [None, "adapter_implementation", "trainable_components", "operator_components", "distributed_strategy"],
)
def test_old_or_changed_text_policy_is_rejected_before_model_or_optimizer_mutation(
    tmp_path, monkeypatch, changed
):
    import ypuddin.train.trainer as module

    trainer, _ = tiny_trainer(tmp_path, monkeypatch)
    state = tmp_path / "state-4"
    state.mkdir()
    trainer.cfg.checkpoint.resume = str(state)
    saved = copy.deepcopy(trainer.compute_policy) if changed else None
    if changed:
        saved[changed] = "old"
    (state / "state.json").write_text(
        json.dumps({"format": 2, "progress": {"extra": {"deterministic": True, "compute_policy": saved}}})
    )
    previous = b"original configuration\n"
    (tmp_path / "config.toml").write_bytes(previous)
    weights = {n: p.clone() for n, p in trainer.loaded.backbone.named_parameters()}
    monkeypatch.setattr(module, "get_family", lambda _: SimpleNamespace())
    monkeypatch.setattr(trainer, "_check_capabilities", lambda: None)
    monkeypatch.setattr(trainer, "_seed_all", lambda: pytest.fail("device/model setup occurred"))
    with pytest.raises(ValueError, match="计算"):
        trainer.prepare_data()
    assert (tmp_path / "config.toml").read_bytes() == previous
    assert all(torch.equal(p, weights[n]) for n, p in trainer.loaded.backbone.named_parameters())


@pytest.mark.parametrize("length", [150, 225])
@pytest.mark.parametrize("count", [1, 2])
def test_sdxl_long_text_has_separate_numeric_identity(length, count):
    cfg = config("sdxl", joint=True, count=count)
    cfg.training.train_text_encoder = False
    cfg.adapter.algo = "lokr"
    _, old = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert old["id"] == DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID
    cfg.model.sdxl_max_token_length = length
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert effective.loop.mixed_precision == "bf16"
    assert policy["id"] == (
        DTK_SDXL_LONG_TEXT_POLICY_ID if count == 1 else DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS[("lokr", "ddp")]
    )
    assert policy["sdxl_max_token_length"] == length
    assert policy["linear_backward_implementation"] == BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
    with pytest.raises(ValueError, match="计算"):
        validate_resume_compute_policy(policy, old)
    cfg.training.train_text_encoder = True
    cfg.adapter.algo = "lora"
    _, te = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert te["sdxl_max_token_length"] == length and te["id"] == DTK_TEXT_LORA_POLICY_IDS["sdxl"]


@pytest.mark.parametrize(
    "key,value", [("allow_tf32", True), ("compile", True), ("offload_text_encoder", True)]
)
def test_text_configuration_drift_fails_before_encoder_runs(tmp_path, monkeypatch, key, value):
    trainer, _ = tiny_trainer(tmp_path, monkeypatch)
    trainer._place_training_model()
    try:
        setattr(trainer.cfg.memory, key, value)
        with pytest.raises(ValueError, match="当前训练设置"):
            trainer._text_cond(["caption"])
    finally:
        trainer._text_adapter_compute_restore()


@pytest.mark.parametrize("length", [150, 225])
def test_long_caption_trainer_installs_new_forward_and_preserves_native_lokr(tmp_path, monkeypatch, length):
    import ypuddin.train.trainer as module
    from ypuddin.adapters.lokr import LoKr
    from ypuddin.train.linear_backward import validate_linear_backward_installation

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    cfg = config("sdxl", joint=True)
    cfg.training.train_text_encoder = False
    cfg.adapter.algo = "lokr"
    cfg.model.sdxl_max_token_length = length
    cfg.checkpoint.output_dir = str(tmp_path)
    trainer = Trainer(cfg, device="cuda:0")
    adapter = LoKr(4, 4, rank=2)
    original_apply = adapter.delta_apply
    backbone = nn.Sequential(
        AdaptedLinear(FrozenLinear.from_linear(nn.Linear(4, 4)), adapter), nn.Conv2d(1, 1, 1)
    )
    trainer.loaded = SimpleNamespace(backbone=backbone)
    monkeypatch.setattr(backbone, "to", lambda _: backbone)
    trainer._place_training_model()
    try:
        trainer._validate_training_compute_policy()
        assert adapter.delta_apply == original_apply
        validate_linear_backward_installation(
            backbone,
            trainer._linear_backward_counts,
            expected_implementation=BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        )
        trainer.cfg.model.sdxl_max_token_length = 75
        with pytest.raises(ValueError, match="当前训练设置"):
            trainer._validate_training_compute_policy()
    finally:
        trainer._linear_backward_restore()
        trainer._conv_forward_restore()


@pytest.mark.parametrize("family", ["anima", "sdxl", "krea2"])
def test_default_auto_text_mode_has_same_actual_online_policy(family):
    online = config(family)
    automatic = online.model_copy(deep=True)
    automatic.dataset.text_encoding = "auto"
    effective, policy = resolve_training_compute_config(automatic, "cuda", "linux-dtk")
    _, expected = resolve_training_compute_config(online, "cuda", "linux-dtk")
    assert policy == expected
    assert effective.dataset.text_encoding == "auto"
    trainer = object.__new__(Trainer)
    trainer.cfg = automatic
    assert trainer._resolve_text_mode() == "online"
