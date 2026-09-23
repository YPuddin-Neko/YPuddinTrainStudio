"""Scope and original Trainer integration for the measured SDXL joint-text preview."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lora import LoRA
from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    DTK_SDXL_TEXT_LORA_PREVIEW_POLICY_IDS,
    DTK_TEXT_LORA_ALL_POLICY_IDS,
    DTK_TEXT_LORA_POLICY_IDS,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)
from ypuddin.train.preview_compute import preview_linear_compute
from ypuddin.train.trainer import Trainer


def config(gpus=1):
    return TrainConfig.model_validate(
        {
            "model": {"family": "sdxl", "sdxl_max_token_length": 150},
            "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": True},
            "loop": {
                "mixed_precision": "bf16",
                "deterministic": True,
                "gpu_count": gpus,
                "distributed_strategy": "ddp",
            },
            "adapter": {"algo": "lora", "mode": "bypass", "param_dtype": "fp32"},
            "dataset": {"text_encoding": "online"},
            "memory": {"offload_text_encoder": False},
        }
    )


@pytest.mark.parametrize("gpus", [1, 2])
def test_joint_text_preview_policy_identity(gpus):
    c = config(gpus)
    _, p = resolve_training_compute_config(c, "cuda", "linux-dtk")
    assert p["id"] == DTK_SDXL_TEXT_LORA_PREVIEW_POLICY_IDS["single" if gpus == 1 else "ddp"]
    assert p["id"] in DTK_TEXT_LORA_ALL_POLICY_IDS
    assert p["preview_operator_components"] == ["backbone"]
    assert p["operator_components"] == ["backbone", "text_encoder", "text_encoder_2"]
    validate_resume_compute_policy(p, dict(p))
    old = {k: v for k, v in p.items() if not k.startswith("preview_")}
    old["id"] = DTK_TEXT_LORA_POLICY_IDS["sdxl"]
    with pytest.raises(ValueError):
        validate_resume_compute_policy(p, old)
    for section, key, value in [
        ("model", "sdxl_max_token_length", 75),
        ("model", "sdxl_max_token_length", 225),
        ("training", "train_backbone", False),
        ("loop", "gpu_count", 3),
    ]:
        changed = c.model_copy(deep=True)
        setattr(getattr(changed, section), key, value)
        _, actual = resolve_training_compute_config(changed, "cuda", "linux-dtk")
        assert actual["id"] == DTK_TEXT_LORA_POLICY_IDS["sdxl"]
        assert "preview_linear_implementation" not in actual


def test_actual_text_operator_installation_preview_scope_and_native_training(monkeypatch):
    c, p = resolve_training_compute_config(config(), "cuda", "linux-dtk")
    monkeypatch.setattr("ypuddin.train.trainer.current_profile", lambda: "linux-dtk")

    def layer():
        return AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoRA(8, 8, rank=2), mode="bypass")

    backbone = nn.Sequential(layer(), nn.Conv2d(1, 1, 1))
    encoders = {"text_encoder": layer(), "text_encoder_2": layer()}
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = c
    trainer.compute_policy = p
    trainer.device = SimpleNamespace(type="cuda")
    trainer.loaded = SimpleNamespace(
        backbone=backbone, text=SimpleNamespace(trainable_modules=lambda: encoders)
    )
    monkeypatch.setattr(backbone, "to", lambda *a, **k: backbone)
    Trainer._place_training_model(trainer)
    Trainer._validate_training_compute_policy(trainer)
    assert set(trainer._text_adapter_operator_counts) == {"backbone", "text_encoder", "text_encoder_2"}
    x = torch.randn(2, 8)
    rng = torch.random.get_rng_state()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        before = backbone[0](x)
        with preview_linear_compute(p, backbone):
            after = backbone[0](x)
    assert torch.equal(before, after) and torch.equal(rng, torch.random.get_rng_state())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        text_before = encoders["text_encoder"](x)
        with preview_linear_compute(p, backbone):
            text_after = encoders["text_encoder"](x)
    assert torch.equal(text_before, text_after)
    encoders["text_encoder"].base.forward = lambda x: x
    with pytest.raises(ValueError):
        Trainer._validate_training_compute_policy(trainer)
