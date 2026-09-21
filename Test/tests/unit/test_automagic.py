"""CPU checks for actual adaptive updates, accumulation, and exact optimizer resume."""

import io

import pytest
import torch
from torch import nn

from ypuddin.config import OptimizerConfig
from ypuddin.optim import build_optimizer
from ypuddin.optim.automagic import Automagic


def _roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def test_factory_builds_real_automagic_without_optional_packages():
    parameter = nn.Parameter(torch.ones(2))
    optimizer = build_optimizer(OptimizerConfig(type="automagic", lr=1e-6), [{"params": [parameter]}])
    assert isinstance(optimizer, Automagic)
    assert optimizer.manages_learning_rate
    parameter.grad = -torch.ones_like(parameter)
    optimizer.step()
    assert bool((parameter > 1).all())
    assert optimizer.get_learning_rates() == pytest.approx([2e-6])


def test_updates_match_unquantized_v1_reference_with_factored_moments_and_decay():
    parameter = nn.Parameter(torch.tensor([[0.3, -0.6, 0.9], [0.1, 0.7, -0.2]]))
    expected = parameter.detach().clone()
    optimizer = Automagic([parameter], weight_decay=0.03)
    row, col = torch.zeros(2), torch.zeros(3)
    mask, sign = torch.full_like(expected, 1e-6), torch.zeros_like(expected, dtype=torch.bool)
    for index in range(5):
        gradient = torch.tensor([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]]) * (-1 if index == 2 else 1)
        row = 0.999 * row + 0.001 * (gradient.square() + 1e-30).mean(-1)
        col = 0.999 * col + 0.001 * (gradient.square() + 1e-30).mean(-2)
        update = gradient / torch.sqrt((row / row.mean()).unsqueeze(-1) * col.unsqueeze(-2))
        update = update / max(1.0, float(update.square().mean().sqrt()))
        new_sign = update > 0
        mask = (mask + torch.where(sign == new_sign, 1e-6, -1e-6)).clamp(1e-7, 1e-3)
        expected = expected - 0.03 * expected * mask - update * mask
        sign = new_sign
        parameter.grad = gradient.clone()
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(optimizer.state[parameter]["lr_mask"], mask, rtol=0, atol=0)


def test_accumulated_gradients_update_only_once_and_equal_combined_batch():
    left, right = nn.Parameter(torch.tensor([0.4, -0.3])), nn.Parameter(torch.tensor([0.4, -0.3]))
    a, b = Automagic([left]), Automagic([right])
    before = left.detach().clone()
    targets = [torch.tensor([0.0, 0.2]), torch.tensor([0.6, 0.8])]
    for target in targets:
        ((left - target).square().sum() / 2).backward()
        assert torch.equal(left, before)
        assert len(a.state) == 0  # no update or grad-clearing backward hook
    sum((right - target).square().sum() for target in targets).div(2).backward()
    a.step()
    b.step()
    assert torch.equal(left, right)
    assert a.state[left]["step"] == 1
    assert left.grad is not None
    a.zero_grad(set_to_none=True)
    assert left.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_resume_preserves_precision_and_group_mapping_with_missing_gradients(dtype):
    initial = [torch.full((3,), 0.3), torch.full((2, 3), -0.6), torch.tensor(0.9)]
    parameters = [nn.Parameter(value.to(dtype)) for value in initial]

    def make(params):
        return Automagic(
            [
                {"params": [params[0]], "name": "skipped", "lr": 2e-6},
                {"params": params[1:], "name": "used", "weight_decay": 0.01},
            ],
            lr=1e-6,
        )

    def advance(opt, params, start, end):
        for index in range(start, end):
            opt.zero_grad(set_to_none=True)
            for number, param in enumerate(params):
                # State insertion order intentionally differs from group order.
                if number == 0 and index < 5:
                    continue
                param.grad = torch.full_like(param, (-1 if index % 3 else 1) * (number + 1) / 4)
            opt.step()

    uninterrupted = make(parameters)
    advance(uninterrupted, parameters, 0, 4)
    saved_state = _roundtrip(uninterrupted.state_dict())
    resumed_params = [nn.Parameter(param.detach().clone()) for param in parameters]
    advance(uninterrupted, parameters, 4, 30)
    resumed = make(resumed_params)
    resumed.load_state_dict(saved_state)
    advance(resumed, resumed_params, 4, 30)
    for actual, expected in zip(resumed_params, parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert resumed.state[actual]["last_polarity"].dtype == torch.bool
        assert resumed.state[actual]["lr_mask"].dtype == torch.float32
        for key, value in uninterrupted.state[expected].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(resumed.state[actual][key], value, rtol=0, atol=0)
            else:
                assert resumed.state[actual][key] == value
    assert resumed.get_learning_rates() == uninterrupted.get_learning_rates()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_small_low_precision_updates_accumulate_in_fp32_master(dtype):
    parameter = nn.Parameter(torch.ones(2, dtype=dtype))
    optimizer = Automagic([parameter])
    for _ in range(100):
        parameter.grad = -torch.ones_like(parameter)
        optimizer.step()
    assert bool((parameter > 1).all())
    assert optimizer.state[parameter]["master_param"].dtype == torch.float32
    assert parameter.dtype == dtype


def test_group_initial_rates_and_bounds_are_honored_and_reported():
    left, right = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))
    optimizer = Automagic(
        [
            {"params": [left], "lr": 1e-6, "max_lr": 3e-6},
            {"params": [right], "lr": 2e-6, "max_lr": 4e-6},
        ]
    )
    assert optimizer.get_learning_rates() == [1e-6, 2e-6]
    assert len(optimizer.state) == 0
    for _ in range(10):
        left.grad, right.grad = -torch.ones_like(left), -torch.ones_like(right)
        optimizer.step()
    assert optimizer.get_learning_rates() == pytest.approx([3e-6, 4e-6])


def test_closure_has_autograd_and_returns_loss():
    parameter = nn.Parameter(torch.tensor([0.4]))
    optimizer = Automagic([parameter])

    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    assert optimizer.step(closure).item() == pytest.approx(0.16)
    assert parameter.grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 1},
        {"min_lr": 1e-3},
        {"max_lr": 0},
        {"lr_bump": -1},
        {"eps": 0},
        {"clip_threshold": 0},
        {"beta2": 1},
        {"beta2": -1},
        {"weight_decay": -1},
        {"lr": float("nan")},
    ],
)
def test_invalid_configuration_fails_before_training(kwargs):
    with pytest.raises(ValueError, match="Automagic"):
        Automagic([nn.Parameter(torch.ones(2))], **kwargs)


def test_sparse_gradients_are_rejected():
    parameter = nn.Parameter(torch.ones(2))
    optimizer = Automagic([parameter])
    parameter.grad = torch.sparse_coo_tensor([[0]], [1.0], [2], check_invariants=True)
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()


def test_empty_state_roundtrip_and_incompatible_checkpoint():
    parameter = nn.Parameter(torch.ones(2))
    optimizer = Automagic([parameter])
    optimizer.load_state_dict(_roundtrip(optimizer.state_dict()))
    other = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    other.step()
    with pytest.raises((KeyError, ValueError)):
        optimizer.load_state_dict(other.state_dict())
