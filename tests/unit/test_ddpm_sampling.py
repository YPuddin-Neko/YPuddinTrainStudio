import math

import pytest
import torch

from ypuddin.config import ObjectiveConfig
from ypuddin.objectives.ddpm import DDPMObjective, unit_to_timesteps
from ypuddin.sampling.ddpm import _schedule, sample_ddpm, sampling_timestep


def generator(seed=17):
    return torch.Generator().manual_seed(seed)


def official_scheduler(sampler, prediction_type, zero_snr, train_steps):
    diffusers = pytest.importorskip("diffusers")
    options = dict(
        num_train_timesteps=train_steps,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type=prediction_type,
        timestep_spacing="linspace",
    )
    if sampler == "euler":
        return diffusers.EulerDiscreteScheduler(**options, rescale_betas_zero_snr=zero_snr)
    scheduler = diffusers.HeunDiscreteScheduler(**options, clip_sample=False)
    if zero_snr:
        # Heun lacks Euler's zero-SNR option: use the official DDPM rescaling,
        # then the same explicit finite inference endpoint as official Euler.
        reference = diffusers.DDPMScheduler(**options, rescale_betas_zero_snr=True, clip_sample=False)
        scheduler.betas = reference.betas.clone()
        scheduler.alphas = reference.alphas.clone()
        scheduler.alphas_cumprod = reference.alphas_cumprod.clone()
        scheduler.alphas_cumprod[-1] = 2**-24
    return scheduler


@pytest.mark.parametrize("steps", [1, 2, 7, 32])
@pytest.mark.parametrize("zero_snr", [False, True])
def test_schedule_matches_official_euler_and_keeps_training_terminal(steps, zero_snr):
    obj = DDPMObjective(
        ObjectiveConfig(timestep_sampling="uniform"), num_train_timesteps=32, zero_terminal_snr=zero_snr
    )
    previous = obj.alphas_cumprod.clone()
    indices, sigmas = _schedule(obj, steps)
    reference = official_scheduler("euler", "v_prediction", zero_snr, 32)
    reference.set_timesteps(steps)
    torch.testing.assert_close(indices, reference.timesteps, rtol=0, atol=0)
    torch.testing.assert_close(sigmas, reference.sigmas, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(obj.alphas_cumprod, previous, rtol=0, atol=0)
    assert sigmas[-1] == 0 and torch.isfinite(sigmas).all()
    if zero_snr:
        assert obj.alphas_cumprod[-1] == 0


@pytest.mark.parametrize("sampler", ["euler", "heun"])
@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("zero_snr", [False, True])
@pytest.mark.parametrize("guidance", [1.0, 4.0])
@pytest.mark.parametrize("train_steps", [32, 1000])
def test_full_trajectory_matches_diffusers(sampler, prediction_type, zero_snr, guidance, train_steps):
    steps, shape = 7, (2, 3)
    reference = official_scheduler(sampler, prediction_type, zero_snr, train_steps)
    reference.set_timesteps(steps)
    observed, expected, progress = [], [], []

    def model(x, index):
        return x.sin() * 0.04 + index[:, None] / train_steps * 0.03

    def unconditional(x, _time):
        return x * 0.015 - 0.02

    def predict(x, t):
        index = sampling_timestep(t, train_steps)
        observed.append((x.clone(), index.clone()))
        return model(x, index)

    actual = sample_ddpm(
        predict,
        shape,
        steps=steps,
        sampler=sampler,
        prediction_type=prediction_type,
        zero_terminal_snr=zero_snr,
        num_train_timesteps=train_steps,
        cfg=guidance,
        predict_uncond=unconditional,
        generator=generator(),
        on_step=lambda i, total: progress.append((i, total)),
    )
    y = torch.randn(shape, generator=generator()) * reference.init_noise_sigma
    for index in reference.timesteps:
        vp_input = reference.scale_model_input(y, index)
        expected.append((vp_input.clone(), index.expand(shape[0]).clone()))
        prediction = model(vp_input, index.expand(shape[0]))
        if guidance != 1:
            uncond = unconditional(vp_input, index)
            prediction = uncond + guidance * (prediction - uncond)
        y = reference.step(prediction, index, y).prev_sample
    torch.testing.assert_close(actual, y, rtol=3e-6, atol=1e-5)
    assert len(observed) == (steps if sampler == "euler" else 2 * steps - 1)
    for (x_actual, t_actual), (x_expected, t_expected) in zip(observed, expected, strict=True):
        torch.testing.assert_close(x_actual, x_expected, rtol=3e-6, atol=1e-5)
        torch.testing.assert_close(t_actual, t_expected, rtol=1e-7, atol=1e-6)
    assert progress == [(i, steps) for i in range(1, steps + 1)]


@pytest.mark.parametrize("sampler", ["euler", "heun"])
@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("steps", [1, 5])
@pytest.mark.parametrize("zero_snr", [False, True])
def test_perfect_noise_or_velocity_prediction_recovers_clean_latent(
    sampler, prediction_type, steps, zero_snr
):
    train_steps = 20
    obj = DDPMObjective(
        ObjectiveConfig(timestep_sampling="uniform"),
        num_train_timesteps=train_steps,
        zero_terminal_snr=zero_snr,
    )
    alpha = obj.alphas_cumprod.clone()
    if zero_snr:
        alpha[-1] = 2**-24
    sigma_table = ((1 - alpha) / alpha).sqrt().double()
    truth = torch.tensor([[0.25, -0.5, 1.0]])

    def oracle(vp_input, t):
        index = sampling_timestep(t, train_steps).double()
        lo = index.floor().long()
        hi = (lo + 1).clamp(max=train_steps - 1)
        sigma = torch.lerp(sigma_table[lo], sigma_table[hi], index - lo)[:, None]
        root = (1 + sigma.square()).sqrt()
        noise = (vp_input.double() * root - truth.double()) / sigma
        return (noise if prediction_type == "epsilon" else (noise - sigma * truth) / root).float()

    actual = sample_ddpm(
        oracle,
        tuple(truth.shape),
        prediction_type=prediction_type,
        steps=steps,
        sampler=sampler,
        zero_terminal_snr=zero_snr,
        num_train_timesteps=train_steps,
        generator=generator(),
    )
    torch.testing.assert_close(actual, truth, rtol=0, atol=2e-6)


def test_continuous_inference_index_is_not_training_floor():
    unit = torch.tensor([0.0, 0.4995, 0.999, 1.0])
    torch.testing.assert_close(sampling_timestep(unit), torch.tensor([0.0, 499.5, 999.0, 999.0]))
    torch.testing.assert_close(unit_to_timesteps(unit), torch.tensor([0, 499, 999, 999]))
    with pytest.raises(ValueError, match="unit values"):
        sampling_timestep(torch.tensor([100.0]))


@pytest.mark.parametrize("sampler", ["euler", "heun"])
def test_initial_noise_scaling_and_seed_only_consume_one_cpu_draw(sampler):
    observed = []
    rng = generator()

    def zero_prediction(x, t):
        observed.append((x.clone(), t.clone()))
        return torch.zeros_like(x)

    result = sample_ddpm(
        zero_prediction, (1, 3), steps=4, prediction_type="epsilon", sampler=sampler, generator=rng
    )
    original = generator()
    noise = torch.randn((1, 3), generator=original)
    obj = DDPMObjective(ObjectiveConfig(timestep_sampling="uniform"))
    sigma_max = ((1 - obj.alphas_cumprod[-1]) / obj.alphas_cumprod[-1]).sqrt()
    torch.testing.assert_close(result, noise * sigma_max, rtol=0, atol=0)
    torch.testing.assert_close(observed[0][0], noise * sigma_max / (1 + sigma_max.square()).sqrt())
    torch.testing.assert_close(observed[0][1], torch.tensor([0.999]))
    assert torch.equal(rng.get_state(), original.get_state())
    repeated = sample_ddpm(
        zero_prediction, (1, 3), steps=4, prediction_type="epsilon", sampler=sampler, generator=generator()
    )
    assert torch.equal(result, repeated)
    different = sample_ddpm(
        zero_prediction, (1, 3), steps=4, prediction_type="epsilon", sampler=sampler, generator=generator(18)
    )
    assert not torch.equal(result, different)


@pytest.mark.parametrize("sampler", ["euler", "heun"])
def test_cfg_zero_selects_unconditional_and_one_skips_it(sampler):
    common = dict(steps=4, sampler=sampler, prediction_type="epsilon")

    def conditional(x, t):
        return x * 0.1 + 0.2

    def unconditional(x, t):
        return x * 0.05 - 0.1

    actual = sample_ddpm(
        conditional, (1, 2), cfg=0, predict_uncond=unconditional, generator=generator(), **common
    )
    expected = sample_ddpm(unconditional, (1, 2), generator=generator(), **common)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def forbidden(*_):
        raise AssertionError("unconditional predictor was called for cfg=1")

    sample_ddpm(conditional, (1, 2), predict_uncond=forbidden, generator=generator(), **common)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_predict_and_output_dtype_preserved_with_float32_integration(dtype):
    def predict(x, t):
        assert x.dtype == dtype
        assert t.dtype == torch.float32 and t.shape == (1,)
        assert not torch.is_grad_enabled()
        return x * 0.01

    result = sample_ddpm(
        predict, (1, 2), steps=3, prediction_type="v_prediction", dtype=dtype, generator=generator()
    )
    assert result.dtype == dtype and torch.isfinite(result).all() and not result.requires_grad


@pytest.mark.parametrize("sampler", ["euler", "heun"])
def test_callback_cancellation_is_not_swallowed(sampler):
    calls = []

    def cancel(done, total):
        calls.append((done, total))
        raise RuntimeError("cancelled DDPM preview")

    with pytest.raises(RuntimeError, match="cancelled DDPM preview"):
        sample_ddpm(
            lambda x, t: x * 0, (1, 2), steps=3, prediction_type="epsilon", sampler=sampler, on_step=cancel
        )
    assert calls == [(1, 3)]


@pytest.mark.parametrize(
    "options",
    [
        {"sampler": "er_sde"},
        {"sampler": "unknown"},
        {"scheduler": "simple"},
        {"scheduler": "sgm_uniform"},
        {"scheduler": "normal"},
        {"scheduler": "unknown"},
        {"shift": 3},
        {"shift": math.nan},
        {"cfg": math.inf},
        {"cfg": -1},
        {"cfg": 4},
        {"steps": 0},
        {"steps": True},
        {"steps": 1001},
        {"num_train_timesteps": 1},
        {"prediction_type": "flow"},
        {"zero_terminal_snr": "false"},
        {"dtype": torch.int64},
        {"er_sde_order": 4},
        {"er_sde_s_noise": math.nan},
    ],
)
def test_unsupported_or_invalid_options_fail_before_predict(options):
    def forbidden(*_):
        raise AssertionError("validation must happen before loading/running the denoiser")

    with pytest.raises(ValueError):
        sample_ddpm(forbidden, (1, 2), **{"steps": 3, "prediction_type": "epsilon", **options})


def test_prediction_shape_errors_are_not_broadcast():
    with pytest.raises(ValueError, match="prediction shape"):
        sample_ddpm(lambda x, t: x[:, :1], (1, 2), steps=2, prediction_type="epsilon")
