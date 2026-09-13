import math

import pytest
import torch

from ypuddin.config import ObjectiveConfig
from ypuddin.objectives.ddpm import DDPMObjective, unit_to_timesteps


def make_objective(*, prediction_type="epsilon", steps=1000, zero_snr=False, **cfg):
    return DDPMObjective(
        ObjectiveConfig(timestep_sampling="uniform", **cfg),
        prediction_type=prediction_type,
        num_train_timesteps=steps,
        zero_terminal_snr=zero_snr,
    )


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("shape", [(3, 2), (3, 4, 2, 2)])
def test_noising_and_target_match_ddpm_equations(prediction_type, shape):
    # Independent scalar schedule calculation; same SDXL beta endpoints as
    # sd-scripts/sdxl_train.py and diffusers DDPMScheduler.scaled_linear.
    obj = make_objective(prediction_type=prediction_type, steps=8)
    betas = [(math.sqrt(0.00085) + i / 7 * (math.sqrt(0.012) - math.sqrt(0.00085))) ** 2 for i in range(8)]
    cumulative = [math.prod(1 - beta for beta in betas[: i + 1]) for i in range(8)]
    x0 = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) / 10
    t = torch.tensor([0.0, 0.5, 1.0])
    noisy, target, noise = obj.prepare(x0, t, generator=torch.Generator().manual_seed(7))
    expected_noise = torch.randn(shape, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(noise, expected_noise, rtol=0, atol=0)
    for row, index in enumerate([0, 4, 7]):
        signal, sigma = math.sqrt(cumulative[index]), math.sqrt(1 - cumulative[index])
        torch.testing.assert_close(noisy[row], signal * x0[row] + sigma * noise[row], rtol=2e-6, atol=2e-6)
        expected = noise[row] if prediction_type == "epsilon" else signal * noise[row] - sigma * x0[row]
        torch.testing.assert_close(target[row], expected, rtol=2e-6, atol=2e-6)
    assert not torch.equal(noisy[0], x0[0])  # t=0 is beta[0], not a clean flow endpoint.
    assert not torch.equal(noisy[-1], noise[-1])


def test_default_schedule_has_the_sdxl_endpoints_and_1000_steps():
    obj = make_objective()
    assert obj.betas.shape == (1000,)
    assert obj.betas.dtype == torch.float32
    assert obj.betas[0].item() == pytest.approx(0.00085)
    assert obj.betas[-1].item() == pytest.approx(0.012)
    assert obj.alphas_cumprod[-1].item() == pytest.approx(0.0046601, rel=1e-5)
    assert (obj.alphas_cumprod[1:] < obj.alphas_cumprod[:-1]).all()


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("steps", [2, 8, 1000])
def test_zero_snr_preserves_initial_noise_and_makes_terminal_pure_noise(prediction_type, steps):
    obj = make_objective(prediction_type=prediction_type, zero_snr=True, steps=steps)
    original = make_objective(steps=steps)
    torch.testing.assert_close(obj.alphas_cumprod[0], original.alphas_cumprod[0], rtol=0, atol=2e-7)
    assert obj.alphas_cumprod[-1].item() == 0
    assert obj.betas[-1].item() == 1
    assert torch.isfinite(obj.betas).all()
    assert ((obj.betas > 0) & (obj.betas <= 1)).all()
    assert (obj.alphas_cumprod[1:] < obj.alphas_cumprod[:-1]).all()
    x0 = torch.tensor([[2.0, -3.0]])
    noisy, target, noise = obj.prepare(x0, torch.tensor([1.0]), generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(noisy, noise, rtol=0, atol=0)
    torch.testing.assert_close(target, noise if prediction_type == "epsilon" else -x0, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unit_timestep_bins_and_boundaries(dtype):
    boundary = torch.tensor(0.25, dtype=dtype)
    below = torch.nextafter(boundary, torch.tensor(0.0, dtype=dtype))
    t = torch.stack((boundary * 0, below, boundary, boundary * 2, boundary * 4))
    torch.testing.assert_close(unit_to_timesteps(t, 4), torch.tensor([0, 0, 1, 2, 3]))
    assert unit_to_timesteps(t, 4).dtype == torch.int64


@pytest.mark.parametrize("value", [-0.001, 1.001, float("nan"), float("inf"), -float("inf")])
def test_rejects_invalid_unit_timesteps(value):
    with pytest.raises(ValueError, match="unit values"):
        make_objective().prepare(torch.ones(1, 2), torch.tensor([value]))


@pytest.mark.parametrize("steps", [0, 1, -1, 3.5, True])
def test_rejects_invalid_schedule_length(steps):
    with pytest.raises(ValueError, match="integer >= 2"):
        make_objective(steps=steps)


def test_sampler_keeps_unit_interface_and_covers_each_discrete_bin():
    obj = make_objective(steps=8, stratified=True)
    t = obj.sample_t(8, generator=torch.Generator().manual_seed(2), num_tokens=4096)
    assert t.dtype == torch.float32 and t.device.type == "cpu"
    assert ((t > 0) & (t < 1)).all()
    torch.testing.assert_close(obj.timesteps(t).sort().values, torch.arange(8))
    again = obj.sample_t(8, generator=torch.Generator().manual_seed(2), num_tokens=256)
    torch.testing.assert_close(t, again, rtol=0, atol=0)  # no resolution-dependent flow shift
    torch.testing.assert_close(obj.timesteps(obj.sampler.icdf([0, 0.5, 1])), torch.tensor([0, 4, 7]))


def test_sampler_honors_clipped_range_and_allows_unshifted_logit_normal():
    obj = make_objective(t_min=0.2, t_max=0.8)
    torch.testing.assert_close(obj.sampler.icdf([0, 1]), torch.tensor([0.2, 0.8]))
    logit = DDPMObjective(ObjectiveConfig(timestep_sampling="logit_normal", logit_mean=0))
    torch.testing.assert_close(logit.sampler.icdf([0.5]), torch.tensor([0.5]))


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_input_perturbation_keeps_original_noise_target(prediction_type):
    obj = make_objective(prediction_type=prediction_type, ip_noise_gamma=0.3)
    baseline = make_objective(prediction_type=prediction_type)
    x0, t = torch.ones(2, 4, 2, 2), torch.tensor([0.2, 0.8])
    noisy, target, noise = obj.prepare(x0, t, generator=torch.Generator().manual_seed(8))
    original_noisy, original_target, original_noise = baseline.prepare(
        x0, t, generator=torch.Generator().manual_seed(8)
    )
    torch.testing.assert_close(target, original_target, rtol=0, atol=0)
    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    assert not torch.equal(noisy, original_noisy)
    gen = torch.Generator().manual_seed(8)
    torch.randn(x0.shape, generator=gen)
    perturbation = torch.randn(x0.shape, generator=gen)
    sigma = (1 - obj.alphas_cumprod[obj.timesteps(t)]).sqrt().view(-1, 1, 1, 1)
    torch.testing.assert_close(noisy - original_noisy, sigma * 0.3 * perturbation, rtol=1e-5, atol=2e-7)


def test_mask_and_prior_weights_preserve_existing_batch_mean_and_gradients():
    obj = make_objective()
    pred = torch.tensor([[[[2.0, 4.0], [8.0, 16.0]]], [[[3.0] * 2] * 2], [[[5.0] * 2] * 2]])
    pred = pred.expand(-1, 2, -1, -1).clone().requires_grad_(True)
    mask = torch.tensor([[[1.0, 0.5], [0, 0]], [[1.0, 1.0], [1, 1]], [[0.0, 0.0], [0, 0]]])
    weights = torch.tensor([1.0, 0.25, 0.5])
    loss, per = obj.loss(
        pred, torch.zeros_like(pred), torch.tensor([0.0, 0.5, 1.0]), mask=mask, sample_weight=weights
    )
    torch.testing.assert_close(per, torch.tensor([8.0, 9.0, 0.0]))
    assert not per.requires_grad
    assert loss.item() == pytest.approx((8 + 9 * 0.25) / 3)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[0, :, 1] == 0).all() and (pred.grad[2] == 0).all()
    assert torch.count_nonzero(pred.grad[1]) == 8


@pytest.mark.parametrize(
    "kind, expected", [("mse", 4.0), ("huber", 0.875), ("pseudo_huber", math.sqrt(4.25) - 0.5)]
)
def test_supported_losses_are_independent_of_prediction_parameterization(kind, expected):
    obj = make_objective(loss=kind, huber_c=0.5, prediction_type="v_prediction")
    loss, per = obj.loss(torch.full((1, 2), 2.0), torch.zeros(1, 2), torch.tensor([0.5]))
    assert loss.item() == pytest.approx(expected)
    assert per.item() == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_noising_keeps_first_step_noise(dtype):
    obj = make_objective(prediction_type="v_prediction")
    result = obj.prepare(
        torch.zeros(1, 4, dtype=dtype), torch.tensor([0.0]), generator=torch.Generator().manual_seed(5)
    )
    assert all(t.dtype == dtype and torch.isfinite(t).all() for t in result)
    assert torch.count_nonzero(result[0]) > 0


@pytest.mark.parametrize("mode", ["shift", "resolution_shift", "mode", "cosmap"])
def test_rejects_flow_timestep_transforms(mode):
    with pytest.raises(ValueError, match="objective.timestep_sampling"):
        DDPMObjective(ObjectiveConfig(timestep_sampling=mode))


@pytest.mark.parametrize("weighting", ["sigma_sqrt", "cosmap", "snr_like", "cosmos"])
def test_rejects_flow_loss_weighting(weighting):
    with pytest.raises(ValueError, match="objective.weighting"):
        make_objective(weighting=weighting)


def test_rejects_unknown_prediction_and_inconsistent_batch():
    with pytest.raises(ValueError, match="prediction_type"):
        make_objective(prediction_type="flow")
    with pytest.raises(ValueError, match="one unit timestep per sample"):
        make_objective().prepare(torch.ones(2, 4), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="shapes must match"):
        make_objective().loss(torch.ones(2, 4), torch.ones(1, 4), torch.tensor([0.2, 0.8]))


def test_description_records_schedule_without_mutating_shared_config():
    cfg = ObjectiveConfig(timestep_sampling="uniform")
    obj = DDPMObjective(cfg, prediction_type="v_prediction", zero_terminal_snr=True)
    assert obj.describe()["prediction_type"] == "v_prediction"
    assert obj.describe()["zero_terminal_snr"] is True
    assert obj.describe()["beta_schedule"] == "scaled_linear"
    assert obj.cfg is not cfg
    assert cfg.weighting == "none" and cfg.timestep_sampling == "uniform"
