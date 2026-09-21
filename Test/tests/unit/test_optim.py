import math

import pytest
import torch
from torch import nn

from ypuddin.config import OptimizerConfig, SchedulerConfig
from ypuddin.optim import KahanWrapper, build_optimizer, build_scheduler


@pytest.mark.parametrize(
    "kind", ["constant", "linear", "cosine", "cosine_restarts", "polynomial", "warmup_stable_decay", "rex"]
)
def test_scheduler_shapes(kind):
    p = nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=1.0)
    cfg = SchedulerConfig(type=kind, warmup_steps=10, min_lr_ratio=0.1, num_cycles=2, decay_steps=20)
    sch = build_scheduler(cfg, opt, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(0.1)  # warmup starts at 1/10
    assert lrs[9] == pytest.approx(1.0)
    assert all(0 <= v <= 1.0 + 1e-9 for v in lrs)
    if kind in ("linear", "cosine", "polynomial", "rex", "warmup_stable_decay"):
        assert lrs[-1] == pytest.approx(0.1, abs=0.02)
    if kind == "constant":
        assert lrs[-1] == 1.0


def test_build_optimizer_groups_and_kahan():
    lin = nn.Linear(8, 8).to(torch.bfloat16)
    groups = [
        {"params": [lin.weight], "lr": 1e-2, "weight_decay": 0.0, "name": "w"},
        {"params": [lin.bias], "lr": 1e-3, "weight_decay": 0.0, "name": "b"},
    ]
    opt = build_optimizer(OptimizerConfig(type="adamw", lr=5e-3, kahan=True), groups)
    assert isinstance(opt, KahanWrapper)
    assert [g["lr"] for g in opt.param_groups] == [1e-2, 1e-3]
    # many tiny updates: bf16 alone would round them away, Kahan accumulates them
    target = torch.full_like(lin.weight, 0.0)
    w0 = lin.weight.detach().float().clone()
    for _ in range(200):
        opt.zero_grad()
        loss = ((lin.weight.float() - w0 - 0.05) ** 2).mean()  # pull weight up by 0.05 total
        loss.backward()
        opt.step()
    moved = (lin.weight.detach().float() - w0).mean().item()
    assert moved > 0.03
    sd = opt.state_dict()
    assert "kahan" in sd
    opt.load_state_dict(sd)
    _ = target, math
