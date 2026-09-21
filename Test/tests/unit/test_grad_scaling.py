"""Actual torch GradScaler integration: unscale, overflow and cold restore."""

import copy
from types import SimpleNamespace

import pytest
import torch

from ypuddin.config import TrainConfig
from ypuddin.train import Trainer
from ypuddin.train.state import Progress


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    return request.param


def make_trainer(device):
    t = object.__new__(Trainer)
    t.cfg = TrainConfig.model_validate(
        {"optimizer": {"type": "adamw", "lr": 0.01, "grad_clip_norm": 1}, "loop": {"log_every": 1}}
    )
    t.device = torch.device(device)
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device))
    t.optimizer = torch.optim.AdamW([p], lr=0.01)
    t.scheduler = torch.optim.lr_scheduler.ExponentialLR(t.optimizer, 0.9)
    t.adapters = SimpleNamespace(parameters=lambda: [p])
    t.progress = Progress(total_steps=8)
    t._loss_ema = None
    t._update_ema = lambda: None
    t._step_hooks = lambda step: None
    t.events = []
    t.emit = lambda kind, **kw: t.events.append((kind, kw))
    t.grad_scaler = torch.amp.GradScaler(device, init_scale=128, growth_interval=2)
    return t, p


def step(t, p, *, overflow=False):
    loss = p.square().sum()
    t._backward(loss)
    if overflow:
        p.grad[0] = float("inf")
    t._optimizer_step(float(loss.detach()), 0.1)


def test_overflow_skips_optimizer_scheduler_and_progress_then_recovers(device):
    t, p = make_trainer(device)
    step(t, p)
    before = p.detach().clone()
    scheduler = copy.deepcopy(t.scheduler.state_dict())
    scale = t.grad_scaler.get_scale()
    step(t, p, overflow=True)
    assert torch.equal(p, before)
    assert t.scheduler.state_dict() == scheduler
    assert t.progress.step == 1
    assert t.grad_scaler.get_scale() == scale / 2
    assert t.grad_scaler.state_dict()["_growth_tracker"] == 0
    assert p.grad is None
    assert any(e[1].get("code") == "amp.overflow" for e in t.events)
    step(t, p)
    assert t.progress.step == 2
    assert not torch.equal(p, before)
    assert torch.isfinite(p).all()


def test_scaled_clipping_matches_unscaled_update(device):
    scaled, a = make_trainer(device)
    plain, b = make_trainer(device)
    plain.grad_scaler = None
    step(scaled, a)
    step(plain, b)
    assert torch.equal(a, b)


def test_scaler_cold_restore_preserves_growth_and_update(device):
    t, p = make_trainer(device)
    step(t, p)
    saved = copy.deepcopy(t.grad_scaler.state_dict())
    cold, q = make_trainer(device)
    q.data.copy_(p)
    cold.optimizer.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
    cold.scheduler.load_state_dict(copy.deepcopy(t.scheduler.state_dict()))
    cold.progress = copy.deepcopy(t.progress)
    cold._restore_grad_scaler({"grad_scaler": saved})
    step(t, p)
    step(cold, q)
    assert torch.equal(p, q)
    assert t.grad_scaler.state_dict() == cold.grad_scaler.state_dict()
    assert t.grad_scaler.get_scale() == 256


def test_missing_scaler_state_cannot_claim_exact_resume():
    t, _ = make_trainer("cpu")
    with pytest.raises(ValueError, match="不能精确恢复"):
        t._restore_grad_scaler({})
    t.grad_scaler = None
    with pytest.raises(ValueError, match="不能精确恢复"):
        t._restore_grad_scaler({"grad_scaler": {"scale": 128}})
