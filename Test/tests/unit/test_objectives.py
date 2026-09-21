import math

import pytest
import torch

from ypuddin.config import ObjectiveConfig
from ypuddin.objectives import (
    Objective,
    TimestepSampler,
    mobius_shift,
    noisy_input_and_target,
    timestep_weight,
)


@pytest.mark.parametrize("mode", ["uniform", "logit_normal", "shift", "resolution_shift", "mode", "cosmap"])
def test_sampler_in_range_and_deterministic(mode):
    cfg = ObjectiveConfig(timestep_sampling=mode)
    s = TimestepSampler(cfg)
    g = torch.Generator().manual_seed(0)
    t = s.sample(4096, generator=g, num_tokens=1024)
    assert t.min() > 0 and t.max() < 1
    g2 = torch.Generator().manual_seed(0)
    torch.testing.assert_close(t, s.sample(4096, generator=g2, num_tokens=1024))


def test_logit_normal_matches_sigmoid_of_normal():
    cfg = ObjectiveConfig(timestep_sampling="logit_normal", stratified=False)
    t = TimestepSampler(cfg).sample(200000, generator=torch.Generator().manual_seed(1))
    logit = torch.log(t / (1 - t))
    assert abs(logit.mean().item()) < 0.02 and abs(logit.std().item() - 1) < 0.02


def test_stratified_covers_quantiles():
    cfg = ObjectiveConfig(timestep_sampling="uniform", stratified=True)
    t = TimestepSampler(cfg).sample(8, generator=torch.Generator().manual_seed(0)).sort().values
    # exactly one sample per 1/8-wide stratum
    assert all((t[i] >= i / 8) and (t[i] < (i + 1) / 8) for i in range(8))


def test_shift_moves_mass_up_and_icdf_consistent():
    cfg = ObjectiveConfig(timestep_sampling="shift", shift=3.0, stratified=False)
    s = TimestepSampler(cfg)
    t = s.sample(100000, generator=torch.Generator().manual_seed(0))
    assert t.mean() > 0.6
    q = s.icdf([0.5])
    torch.testing.assert_close(q, mobius_shift(torch.tensor([0.5]), 3.0))
    assert abs(t.median().item() - q.item()) < 0.01


def test_t_min_max_clamp():
    cfg = ObjectiveConfig(timestep_sampling="uniform", t_min=0.2, t_max=0.8)
    t = TimestepSampler(cfg).sample(1000, generator=torch.Generator().manual_seed(0))
    assert t.min() >= 0.2 and t.max() <= 0.8


def test_noisy_input_and_target_invariants():
    x0 = torch.randn(3, 4, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.0, 0.5, 1.0])
    x_t, target = noisy_input_and_target(x0, noise, t)
    torch.testing.assert_close(x_t[0], x0[0])
    torch.testing.assert_close(x_t[2], noise[2])
    torch.testing.assert_close(x_t[1], 0.5 * x0[1] + 0.5 * noise[1])
    torch.testing.assert_close(target, noise - x0)


@pytest.mark.parametrize("scheme", ["none", "sigma_sqrt", "cosmap", "snr_like", "cosmos"])
def test_weights_finite_positive(scheme):
    t = torch.linspace(0.01, 0.99, 50)
    w = timestep_weight(t, scheme)
    assert torch.isfinite(w).all() and (w > 0).all()


def test_cosmos_weight_formula():
    t = torch.tensor([0.5])
    assert timestep_weight(t, "cosmos").item() == pytest.approx(0.5)
    assert timestep_weight(torch.tensor([0.5]), "cosmap").item() == pytest.approx(4 / math.pi)


@pytest.mark.parametrize("loss", ["mse", "huber", "pseudo_huber"])
def test_objective_loss_masked(loss):
    obj = Objective(ObjectiveConfig(loss=loss, huber_c=0.5))
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.3, 0.7])
    full, per = obj.loss(pred, target, t)
    assert per.shape == (2,) and full.dim() == 0
    mask = torch.zeros(2, 8, 8)
    mask[:, :4] = 1
    masked, per_m = obj.loss(pred, target, t, mask=mask)
    ref = (pred - target) ** 2 if loss == "mse" else None
    if ref is not None:
        torch.testing.assert_close(per_m, ref[:, :, :4].flatten(1).mean(1))
    assert torch.isfinite(masked)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
        ),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
        ),
    ],
)
def test_cpu_noise_generator_matches_on_all_devices(device):
    obj = Objective(ObjectiveConfig(ip_noise_gamma=0.2))
    x = torch.ones(2, 4, 8, 8)
    t = torch.tensor([0.2, 0.7])
    expected = obj.prepare(x, t, generator=torch.Generator().manual_seed(9))
    gen = torch.Generator().manual_seed(9)
    actual = obj.prepare(x.to(device), t.to(device), generator=gen)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.cpu(), b, rtol=1e-6, atol=1e-6)
