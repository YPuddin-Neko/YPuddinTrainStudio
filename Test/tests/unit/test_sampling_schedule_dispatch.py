import math

import pytest
import torch

from ypuddin.sampling import euler_sample, noise_schedule, sample


def generator():
    return torch.Generator().manual_seed(17)


def test_default_dispatch_preserves_original_euler_bitwise():
    def predict(x, t):
        return x.square() * 0.01 + t[:, None]

    options = dict(steps=7, shift=3, cfg=4, predict_uncond=lambda x, t: x * 0.01)
    actual = sample(predict, (2, 3), generator=generator(), **options)
    original = euler_sample(predict, (2, 3), generator=generator(), **options)
    assert torch.equal(actual, original)


def test_schedule_has_distinct_flow_grid_semantics():
    simple = noise_schedule(3, 1, "simple", dtype=torch.float64)
    assert simple.tolist() == pytest.approx([1, 0.667, 0.334, 0])
    normal = noise_schedule(3, 1, "normal", dtype=torch.float64)
    sgm = noise_schedule(3, 1, "sgm_uniform", dtype=torch.float64)
    assert normal.tolist() == pytest.approx([1, 0.5005, 0.001, 0])
    assert sgm.tolist() == pytest.approx([1, 0.667, 0.334, 0])
    shifted_low = 3 / 1002
    expected_low = shifted_low / (shifted_low + (1 - shifted_low) / 3)
    assert noise_schedule(3, 3, "normal", dtype=torch.float64)[-2] == pytest.approx(expected_low)
    assert not torch.equal(noise_schedule(25, 3, "simple"), noise_schedule(25, 3, "sgm_uniform"))


@pytest.mark.parametrize("scheduler", ["uniform", "simple", "sgm_uniform", "normal"])
@pytest.mark.parametrize("steps", [1, 2, 25, 1000])
def test_schedules_are_finite_descending_and_have_terminal_zero(scheduler, steps):
    grid = noise_schedule(steps, 3, scheduler)
    assert len(grid) == steps + 1 and grid[0] == 1 and grid[-1] == 0
    assert torch.isfinite(grid).all() and (grid[:-1] > grid[1:]).all()


def test_heun_is_second_order_for_time_varying_velocity_except_terminal_step():
    seen = []
    initial = torch.randn((1, 1), generator=generator())
    out = sample(
        lambda x, t: t[:, None].expand_as(x),
        (1, 1),
        steps=4,
        sampler="heun",
        generator=generator(),
        on_step=lambda i, n: seen.append((i, n)),
    )
    # Trapezoids for t=1..0.25, then the terminal Euler step.
    assert (initial - out).item() == pytest.approx(0.53125, abs=1e-6)
    assert seen == [(i, 4) for i in range(1, 5)]


@pytest.mark.parametrize("sampler", ["euler", "heun", "er_sde"])
def test_callback_cancellation_is_not_swallowed(sampler):
    def cancel(*_):
        raise RuntimeError("cancelled preview")

    with pytest.raises(RuntimeError, match="cancelled preview"):
        sample(lambda x, t: x, (1, 2), steps=3, sampler=sampler, on_step=cancel)


@pytest.mark.parametrize(
    "options",
    [
        {"sampler": "unknown"},
        {"scheduler": "unknown"},
        {"shift": math.inf},
        {"steps": 0},
        {"cfg": math.nan},
        {"er_sde_s_noise": math.inf},
    ],
)
def test_invalid_options_do_not_silently_fall_back(options):
    with pytest.raises(ValueError):
        sample(lambda x, t: x, (1, 2), **{"steps": 3, **options})
