"""Numerical and operational checks independent from the implementation's quadrature."""

import math

import numpy as np
import pytest
import torch

from ypuddin.sampling.er_sde import _log_phi, er_sde_sample


def trapezoid(values, points):
    # Keep reference tests compatible with the declared NumPy >=1.26 floor.
    return np.sum((values[:-1] + values[1:]) * np.diff(points) * 0.5)


def reference(denoiser, times, *, seed=123, cfg=1.0, noise=1.0, order=3, shape=(2, 3)):
    """Direct author VP equations with dense trapezoidal integration, no production helpers."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=rng, dtype=torch.float64)
    history, states = [], []
    previous_derivative = None
    for index, (time, following) in enumerate(zip(times, times[1:], strict=False)):
        states.append(x.clone())
        value = denoiser(time)
        if following == 0:
            return states, torch.full(shape, value, dtype=torch.float64)
        a, b = time / (1 - time), following / (1 - following)

        def phi(z):
            return z * (np.exp(z**0.3) + 10.0)

        ratio = phi(b) / phi(a)
        alpha, alpha_next = 1 - time, 1 - following
        updated = (alpha_next / alpha * ratio) * x + alpha_next * (1 - ratio) * value
        if index and order >= 2:
            points = np.linspace(b, a, 200001, dtype=np.float64)
            i0 = trapezoid(1.0 / phi(points), points)
            i1 = trapezoid((points - a) / phi(points), points)
            derivative = (value - history[-1][1]) / (a - history[-1][0])
            updated += alpha_next * (b - a + phi(b) * i0) * derivative
            if index >= 2 and order == 3:
                second = 2 * (derivative - previous_derivative) / (a - history[-2][0])
                updated += alpha_next * ((b - a) ** 2 / 2 + phi(b) * i1) * second
            previous_derivative = derivative
        variance = b * b - a * a * ratio * ratio
        if noise and variance > 0:
            updated += (
                alpha_next
                * math.sqrt(variance)
                * noise
                * torch.randn(shape, generator=rng, dtype=torch.float64)
            )
        history.append((a, value))
        x = updated
    raise AssertionError("zero terminal time required")


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("stochastic", [0.0, 1.0])
def test_matches_independent_vp_reference_on_nonuniform_grid(order, stochastic):
    times = [0.82, 0.64, 0.37, 0.21, 0.07, 0.0]

    def denoiser(t):
        return 0.2 + 0.03 * (t / (1 - t)) + 0.015 * (t / (1 - t)) ** 2

    states = []

    def predict(x, t):
        states.append(x.clone())
        return (x - denoiser(float(t[0]))) / t[:, None]

    actual = er_sde_sample(
        predict,
        (2, 3),
        sigmas=torch.tensor(times, dtype=torch.float64),
        generator=torch.Generator().manual_seed(123),
        dtype=torch.float64,
        max_order=order,
        s_noise=stochastic,
    )
    expected_states, expected = reference(denoiser, times, order=order, noise=stochastic)
    assert len(states) == len(expected_states)
    for actual_state, expected_state in zip(states, expected_states, strict=True):
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_constant_denoiser_has_exact_deterministic_mean_trajectory():
    times = [0.9, 0.7, 0.4, 0.15, 0.0]
    states = []
    clean = 0.375

    def predict(x, t):
        states.append(x.clone())
        return (x - clean) / t[:, None]

    result = er_sde_sample(
        predict,
        (2, 3),
        sigmas=torch.tensor(times, dtype=torch.float64),
        generator=torch.Generator().manual_seed(72),
        dtype=torch.float64,
        s_noise=0,
    )

    def phi(z):
        return z * (math.exp(z**0.3) + 10)

    initial = states[0]
    for time, state in zip(times[:-1], states, strict=True):
        ratio = phi(time / (1 - time)) / phi(times[0] / (1 - times[0]))
        exact = (1 - time) * (ratio * initial / (1 - times[0]) + (1 - ratio) * clean)
        torch.testing.assert_close(state, exact, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(result, torch.full_like(result, clean), atol=1e-14, rtol=0)


def test_linear_denoiser_corrections_match_integrated_solution_after_startup():
    times = [0.8, 0.61, 0.49, 0.23, 0.01, 0.0]

    def value(t):
        return -0.1 + 0.17 * t / (1 - t)

    states = []

    def predict(x, t):
        states.append(x.clone())
        return (x - value(float(t[0]))) / t[:, None]

    er_sde_sample(
        predict,
        (2, 3),
        sigmas=torch.tensor(times, dtype=torch.float64),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(123),
        s_noise=0,
    )
    expected, _ = reference(value, times, noise=0)
    for state, target in zip(states, expected, strict=True):
        torch.testing.assert_close(state, target, rtol=3e-9, atol=2e-10)


def test_seed_determinism_and_global_training_rng_are_unchanged():
    times = torch.tensor([0.9, 0.71, 0.43, 0.15, 0.0], dtype=torch.float64)
    before = torch.random.get_rng_state().clone()

    def predict(x, t):
        return x * 0.23

    first = er_sde_sample(predict, (2, 3), sigmas=times, generator=torch.Generator().manual_seed(42))
    second = er_sde_sample(predict, (2, 3), sigmas=times, generator=torch.Generator().manual_seed(42))
    other = er_sde_sample(predict, (2, 3), sigmas=times, generator=torch.Generator().manual_seed(43))
    er_sde_sample(predict, (2, 3), sigmas=times)
    assert torch.equal(first, second) and not torch.equal(first, other)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_single_step_and_cfg_zero_use_only_unconditional_velocity(dtype):
    calls = []

    def unused(x, t):
        pytest.fail("CFG zero must not evaluate the conditional model")

    def uncond(x, t):
        calls.append((x.dtype, tuple(t.shape)))
        return (x.float() - 2) / t[:, None]

    progress = []
    result = er_sde_sample(
        unused,
        (2, 3),
        sigmas=torch.tensor([0.75, 0.0]),
        cfg=0,
        predict_uncond=uncond,
        dtype=dtype,
        on_step=lambda done, total: progress.append((done, total)),
    )
    assert calls == [(dtype, (2,))] and progress == [(1, 1)]
    assert result.dtype == dtype
    torch.testing.assert_close(result.float(), torch.full((2, 3), 2.0), atol=0.02, rtol=0)


def test_guidance_combines_velocities_before_denoising():
    result = er_sde_sample(
        lambda x, t: (x - 3) / t[:, None],
        (2, 3),
        sigmas=torch.tensor([0.75, 0.0], dtype=torch.float64),
        dtype=torch.float64,
        cfg=2,
        predict_uncond=lambda x, t: (x - 1) / t[:, None],
    )
    torch.testing.assert_close(result, torch.full_like(result, 5), rtol=0, atol=3e-15)


def test_callback_cancellation_and_predictor_errors_propagate():
    class Cancelled(Exception):
        pass

    calls = []

    def predict(x, t):
        calls.append(float(t[0]))
        return torch.zeros_like(x)

    def stop(done, total):
        assert (done, total) == (1, 3)
        raise Cancelled("requested")

    with pytest.raises(Cancelled, match="requested"):
        er_sde_sample(predict, (1, 2), sigmas=torch.tensor([0.8, 0.5, 0.2, 0]), on_step=stop)
    assert len(calls) == 1

    def failed(x, t):
        raise LookupError("model failed")

    with pytest.raises(LookupError, match="model failed"):
        er_sde_sample(failed, (1, 2), sigmas=torch.tensor([0.8, 0]))


@pytest.mark.parametrize(
    "times",
    [
        [0.9999999999999999, 0.999999999999, 0.999, 0.2, 0.0],
        [0.9, 1e-50, 1e-150, 1e-250, 0.0],
        [0.6, 0.6000000000000000 - 1e-14, 0.6 - 2e-14, 0.2, 0.0],
    ],
)
def test_extreme_lambdas_and_nearby_steps_remain_finite(times):
    result = er_sde_sample(
        lambda x, t: torch.zeros_like(x),
        (2, 3),
        sigmas=torch.tensor(times, dtype=torch.float64),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.isfinite(result).all()
    assert math.isfinite(_log_phi(times[0] / (1 - times[0])))


@pytest.mark.parametrize(
    "times",
    [
        [1, 0],
        [0.5],
        [0.8, 0.8, 0],
        [0.4, 0.6, 0],
        [0.8, -0.1, 0],
        [0.8, 0.1],
        [float("nan"), 0],
        [float("inf"), 0],
    ],
)
def test_invalid_schedules_fail_before_prediction(times):
    with pytest.raises(ValueError, match="ER-SDE"):
        er_sde_sample(
            lambda x, t: pytest.fail("invalid schedule reached model"),
            (1, 2),
            sigmas=torch.tensor(times, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (0, 2)},
        {"shape": ()},
        {"shape": (True, 2)},
        {"max_order": 0},
        {"max_order": 4},
        {"max_order": 2.0},
        {"dtype": torch.int64},
        {"s_noise": -1},
        {"cfg": float("nan")},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    kwargs = dict(kwargs)
    shape = kwargs.pop("shape", (1, 2))
    with pytest.raises(ValueError, match="ER-SDE"):
        er_sde_sample(lambda x, t: torch.zeros_like(x), shape, sigmas=torch.tensor([0.8, 0.0]), **kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_prediction_is_not_silently_repaired(value):
    with pytest.raises(ValueError, match="NaN or infinity"):
        er_sde_sample(lambda x, t: torch.full_like(x, value), (1, 2), sigmas=torch.tensor([0.8, 0.0]))


def test_prediction_shape_must_match():
    with pytest.raises(ValueError, match="matching shape"):
        er_sde_sample(lambda x, t: x[:, :1], (1, 2), sigmas=torch.tensor([0.8, 0.0]))


@pytest.mark.parametrize("scheduler", ["uniform", "simple", "sgm_uniform", "normal"])
def test_shared_dispatch_uses_solver_and_private_rng(scheduler):
    from ypuddin.sampling.dispatch import sample

    progress = []
    before = torch.random.get_rng_state().clone()
    result = sample(
        lambda x, t: (x - 0.3) / t[:, None],
        (2, 3),
        steps=6,
        shift=3,
        sampler="er_sde",
        scheduler=scheduler,
        generator=torch.Generator().manual_seed(12),
        on_step=lambda done, total: progress.append((done, total)),
    )
    torch.testing.assert_close(result, torch.full_like(result, 0.3), rtol=0, atol=2e-6)
    assert progress == [(step, 6) for step in range(1, 7)]
    assert torch.equal(before, torch.random.get_rng_state())
