import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lora import LoRA
from ypuddin.train.fp16_adapter_compute import _FP16OperandsFP32Compute as HalfLinear
from ypuddin.train.fp16_adapter_compute import install_fp16_adapter_compute as install
from ypuddin.train.fp16_adapter_compute import validate_fp16_adapter_compute


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_rounded_operands_and_gradient_boundaries(dtype, strided, bias):
    torch.manual_seed(82)
    x = torch.randn(2, 3, 7, dtype=dtype)
    if strided:
        x = x.transpose(0, 1)
    x = x.requires_grad_()
    w = torch.randn(9, 7, dtype=dtype, requires_grad=True)
    b = torch.randn(9, dtype=dtype, requires_grad=True) if bias else None
    output = HalfLinear.apply(x, w, b)
    expected = F.linear(
        x.detach().half().float(), w.detach().half().float(), None if b is None else b.detach().half().float()
    ).half()
    assert output.dtype == torch.float16 and torch.equal(output, expected)
    g = torch.randn_like(output)
    output.backward(g)
    assert torch.equal(x.grad, (g.float() @ w.detach().half().float()).to(dtype))
    assert torch.equal(
        w.grad, (g.float().reshape(-1, 9).T @ x.detach().half().float().reshape(-1, 7)).to(dtype)
    )
    if b is not None:
        assert torch.equal(b.grad, g.float().reshape(-1, 9).sum(0).to(dtype))


def test_native_no_grad_restored_and_no_parameter_replacement():
    torch.manual_seed(9)
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoRA(8, 8, rank=2), mode="bypass")
    layer.eval()
    original = copy.deepcopy(layer)
    parameters = tuple(id(p) for p in layer.parameters())
    undo, counts = install(layer)
    x = torch.randn(2, 3, 8)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        assert torch.equal(layer(x), original(x))
    with torch.autocast("cpu", dtype=torch.float16):
        layer(x).float().square().sum().backward()
    assert validate_fp16_adapter_compute(layer, counts) == {"base_modules": 1, "lora_modules": 1}
    assert tuple(id(p) for p in layer.parameters()) == parameters
    assert all(p.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in layer.adapter.parameters())
    undo()
    assert "forward" not in layer.base.__dict__ and "delta_apply" not in layer.adapter.__dict__


def config():
    from ypuddin.config import TrainConfig

    return TrainConfig.model_validate(
        {
            "model": {"family": "anima", "dtype": "bf16"},
            "loop": {
                "deterministic": True,
                "mixed_precision": "fp16",
                "gpu_count": 1,
                "distributed_strategy": "ddp",
            },
            "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": False},
            "adapter": {"algo": "lora", "mode": "bypass", "param_dtype": "fp32"},
        }
    )


def test_policy_scope_and_resume_identity():
    from ypuddin.config.compute_policy import (
        DTK_ANIMA_LORA_SINGLE_FP16_POLICY_ID,
        resolve_training_compute_config,
        validate_resume_compute_policy,
    )

    c = config()
    _, policy = resolve_training_compute_config(c, "cuda", "linux-dtk")
    assert policy["id"] == DTK_ANIMA_LORA_SINGLE_FP16_POLICY_ID
    assert policy["mixed_precision"] == "fp16" and "preview_linear_implementation" not in policy
    for section, key, value in [
        ("model", "family", "sdxl"),
        ("model", "dtype", "fp16"),
        ("memory", "base_precision", "fp32"),
        ("memory", "base_precision", "fp8_e4m3"),
        ("loop", "gpu_count", 2),
        ("loop", "distributed_strategy", "fsdp"),
        ("loop", "deterministic", False),
        ("loop", "mixed_precision", "bf16"),
        ("training", "mode", "full"),
        ("training", "train_text_encoder", True),
        ("training", "train_backbone", False),
        ("adapter", "algo", "lokr"),
        ("adapter", "param_dtype", "bf16"),
        ("adapter", "dora", True),
        ("memory", "compile", True),
        ("memory", "blocks_to_swap", 1),
        ("memory", "activation_checkpointing", "unsloth"),
    ]:
        changed = c.model_copy(deep=True)
        setattr(getattr(changed, section), key, value)
        _, actual = resolve_training_compute_config(changed, "cuda", "linux-dtk")
        assert actual is None or actual["id"] != policy["id"]
    for device, profile in [("cpu", "linux-dtk"), ("cuda", "linux-cuda")]:
        assert resolve_training_compute_config(c, device, profile)[1] is None
    validate_resume_compute_policy(policy, dict(policy))
    with pytest.raises(ValueError):
        validate_resume_compute_policy(policy, None)
    with pytest.raises(ValueError):
        validate_resume_compute_policy(policy, {**policy, "id": "old"})


def test_install_guards_restore_and_parameter_identity():
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoRA(8, 8, rank=2), mode="bypass")
    ids = tuple(map(id, layer.parameters()))
    undo, counts = install(layer)
    with pytest.raises(ValueError):
        install(layer)
    with pytest.raises(ValueError):
        validate_fp16_adapter_compute(layer, {"base_modules": 0, "lora_modules": 1})
    assert tuple(map(id, layer.parameters())) == ids
    undo()
    undo()
    with pytest.raises(ValueError):
        validate_fp16_adapter_compute(layer, counts)
    with pytest.raises(ValueError):
        install(nn.Linear(8, 8))
    layer.adapter.to(torch.bfloat16)
    with pytest.raises(ValueError):
        install(layer)
    assert "forward" not in layer.base.__dict__


def test_native_non_fp16_and_original_eval():
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoRA(8, 8, rank=2), mode="bypass")
    original = copy.deepcopy(layer)
    undo, _ = install(layer)
    x = torch.randn(2, 8)
    assert torch.equal(layer(x), original(x))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(layer(x), original(x))
    undo()


def test_fp16_grad_scaler_overflow_and_restore():
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoRA(8, 8, rank=2), mode="bypass")
    undo, _ = install(layer)
    optimizer = torch.optim.AdamW(layer.adapter.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    x = torch.randn(2, 8)
    with torch.autocast("cpu", dtype=torch.float16):
        loss = layer(x).float().square().sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    saved = copy.deepcopy(scaler.state_dict())
    restored = torch.amp.GradScaler("cpu")
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    optimizer.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in layer.adapter.parameters()]
    with torch.autocast("cpu", dtype=torch.float16):
        loss = layer(x).float().square().sum() * float("inf")
    restored.scale(loss).backward()
    restored.step(optimizer)
    restored.update()
    assert restored.get_scale() == saved["scale"] * saved["backoff_factor"]
    assert all(torch.equal(a, b) for a, b in zip(before, layer.adapter.parameters(), strict=True))
    undo()


def test_trainer_installs_and_revalidates_actual_operators(monkeypatch):
    from types import SimpleNamespace

    from ypuddin.config.compute_policy import resolve_training_compute_config
    from ypuddin.train.trainer import Trainer

    c, policy = resolve_training_compute_config(config(), "cuda", "linux-dtk")
    monkeypatch.setattr("ypuddin.train.trainer.current_profile", lambda: "linux-dtk")
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoRA(8, 8, rank=2), mode="bypass")
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = c
    trainer.compute_policy = policy
    trainer.device = SimpleNamespace(type="cuda")
    trainer.loaded = SimpleNamespace(backbone=layer)
    monkeypatch.setattr(layer, "to", lambda *a, **k: layer)
    Trainer._place_training_model(trainer)
    Trainer._validate_training_compute_policy(trainer)
    layer.base.forward = lambda x: x
    with pytest.raises(ValueError):
        Trainer._validate_training_compute_policy(trainer)
