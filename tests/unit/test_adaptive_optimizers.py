"""Exercise real optional optimizers on CPU; no model assets or accelerator jobs."""

import copy

import pytest
import torch
from torch import nn

from ypuddin.config import OptimizerConfig, SchedulerConfig
from ypuddin.optim import (
    build_optimizer,
    build_scheduler,
    is_schedule_free,
    load_optimizer_state,
    manages_learning_rate,
    optimizer_hyperparameter_snapshot,
    optimizer_learning_rates,
    optimizer_rate_snapshot,
    validate_optimizer_runtime,
)


@pytest.mark.parametrize(
    "kind,module",
    [
        ("prodigy", "prodigyopt"),
        ("prodigy_plus_sf", "prodigyplus"),
    ],
)
def test_real_prodigy_step_and_exact_fp32_state_resume(kind, module):
    pytest.importorskip(module)
    config = OptimizerConfig(type=kind, lr=1)
    parameters = [
        nn.Parameter(torch.linspace(0.1, 0.7, 8).reshape(2, 4)),
        nn.Parameter(torch.linspace(-0.1, -0.7, 4)),
    ]

    def make(params):
        opt = build_optimizer(
            config, [{"params": [p], "lr": 1, "name": str(i)} for i, p in enumerate(params)]
        )
        if is_schedule_free(config):
            opt.train()
        return opt

    def advance(opt, params, start, end):
        for step in range(start, end):
            opt.zero_grad()
            for p in params:
                p.grad = torch.sin(torch.arange(p.numel()).reshape(p.shape) + step + 1)
            opt.step()

    original = make(parameters)
    before = [p.detach().clone() for p in parameters]
    advance(original, parameters, 0, 6)
    restored_params = [nn.Parameter(p.detach().clone()) for p in parameters]
    saved = copy.deepcopy(original.state_dict())
    restored = make(restored_params)
    load_optimizer_state(config, restored, saved)
    validate_optimizer_runtime(config, restored)
    advance(original, parameters, 6, 12)
    advance(restored, restored_params, 6, 12)
    for expected, actual, initial in zip(parameters, restored_params, before, strict=True):
        assert torch.isfinite(expected).all()
        assert not torch.equal(initial, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    from tests.checkpoint_assertions import assert_checkpoint_value_exact

    assert_checkpoint_value_exact(restored.state_dict(), original.state_dict(), "optimizer")

    rates = optimizer_learning_rates(config, original)
    assert rates and all(0 < value < 1 for value in rates.values())
    if is_schedule_free(config):
        raw = [p.detach().clone() for p in parameters]
        original.eval()
        assert all(torch.isfinite(p).all() for p in parameters)
        original.train()
        for parameter, expected in zip(parameters, raw, strict=True):
            torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


def test_came_three_decay_values_and_real_step():
    pytest.importorskip("pytorch_optimizer")
    parameter = nn.Parameter(torch.linspace(0.1, 0.7, 64).reshape(8, 8))
    initial = parameter.detach().clone()
    optimizer = build_optimizer(OptimizerConfig(type="came"), [{"params": [parameter]}])
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.99, 0.9999)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, initial)


@pytest.mark.parametrize("kind,rate", [("prodigy", 1), ("prodigy_plus_sf", 1), ("automagic", 1e-6)])
def test_parameter_group_override_cannot_bypass_managed_rate(kind, rate):
    if kind == "prodigy":
        pytest.importorskip("prodigyopt")
    elif kind == "prodigy_plus_sf":
        pytest.importorskip("prodigyplus")
    config = OptimizerConfig(type=kind, lr=rate)
    with pytest.raises(ValueError, match="per-group/per-layer"):
        build_optimizer(config, [{"params": [nn.Parameter(torch.ones(2))], "lr": rate / 2}])


def test_resume_cannot_restore_old_unprotected_prodigy_rate():
    pytest.importorskip("prodigyopt")
    config = OptimizerConfig(type="prodigy", lr=1)
    parameter = nn.Parameter(torch.ones(2))
    optimizer = build_optimizer(config, [{"params": [parameter]}])
    optimizer.param_groups[0]["lr"] = 1e-4  # value loaded by legacy checkpoint
    with pytest.raises(ValueError, match="managed learning rate"):
        validate_optimizer_runtime(config, optimizer)


def test_automagic_is_not_schedule_free_but_has_no_external_scheduler():
    config = OptimizerConfig(type="automagic", lr=1e-6)
    assert not is_schedule_free(config)
    assert manages_learning_rate(config)


def test_prodigy_keeps_supported_cosine_scheduler():
    pytest.importorskip("prodigyopt")
    config = OptimizerConfig(type="prodigy", lr=1)
    parameter = nn.Parameter(torch.ones(2))
    optimizer = build_optimizer(config, [{"params": [parameter]}])
    scheduler = build_scheduler(SchedulerConfig(type="cosine"), optimizer, 10)
    assert not manages_learning_rate(config)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert 0 < optimizer.param_groups[0]["lr"] < 1
    validate_optimizer_runtime(config, optimizer)


def test_ppsf_shared_adaptation_reporting_uses_effective_rate():
    pytest.importorskip("prodigyplus")
    config = OptimizerConfig(type="prodigy_plus_sf", lr=1)
    optimizer = build_optimizer(config, [{"params": [nn.Parameter(torch.ones(2))], "name": "adapter"}])
    group = optimizer.param_groups[0]
    group.update(d=1e-4, shared_d=2e-4, split_groups=True, split_groups_mean=True, effective_lr=0.4)
    assert optimizer_learning_rates(config, optimizer) == pytest.approx({"adapter": 8e-5})


def test_zero_rate_group_is_frozen_in_automagic():
    from ypuddin.optim.automagic import Automagic

    parameter = nn.Parameter(torch.ones(2))
    optimizer = Automagic([{"params": [parameter], "lr": 0}])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.equal(parameter, torch.ones(2))
    assert optimizer.get_learning_rates() == [0]
    assert not optimizer.state


@pytest.mark.parametrize(
    "kind,options,expected",
    [
        (
            "prodigy",
            {"d_coef": 2.0, "d0": 2e-6, "growth_rate": 1.2, "slice_p": 2},
            {"d_coef": 2.0, "d0": 2e-6, "growth_rate": 1.2, "slice_p": 2},
        ),
        (
            "prodigy_plus_sf",
            {"d_coef": 0.5, "use_schedulefree": False, "factored": False},
            {"d_coef": 0.5, "use_schedulefree": False, "factored": False},
        ),
        (
            "automagic",
            {"beta2": 0.95, "min_lr": 2e-7, "max_lr": 2e-4, "lr_bump": 2e-6},
            {"beta2": 0.95, "min_lr": 2e-7, "max_lr": 2e-4, "lr_bump": 2e-6},
        ),
    ],
)
def test_typed_adaptive_controls_reach_real_optimizer(kind, options, expected):
    if kind == "prodigy":
        pytest.importorskip("prodigyopt")
    elif kind == "prodigy_plus_sf":
        pytest.importorskip("prodigyplus")
    config = OptimizerConfig(type=kind, **options)
    optimizer = build_optimizer(config, [{"params": [nn.Parameter(torch.ones(4))]}])
    assert {key: optimizer.param_groups[0][key] for key in expected} == expected


def test_logged_step_uses_pre_update_d_and_current_sf_mixing():
    pytest.importorskip("prodigyplus")
    config = OptimizerConfig(type="prodigy_plus_sf")
    optimizer = build_optimizer(config, [{"params": [nn.Parameter(torch.ones(2))], "name": "adapter"}])
    group = optimizer.param_groups[0]
    group.update(d=1e-4, shared_d=2e-4, split_groups=True, split_groups_mean=True)
    snapshot = optimizer_rate_snapshot(optimizer)
    group.update(d=5e-4, shared_d=6e-4, effective_lr=0.4)
    rates = optimizer_learning_rates(config, optimizer, before_step=snapshot)
    assert rates == pytest.approx({"adapter": 8e-5})


def test_logged_prodigy_step_keeps_used_lr_and_bias_correction():
    pytest.importorskip("prodigyopt")
    config = OptimizerConfig(type="prodigy", use_bias_correction=True)
    optimizer = build_optimizer(config, [{"params": [nn.Parameter(torch.ones(2))], "name": "adapter"}])
    group = optimizer.param_groups[0]
    group.update(d=1e-4, lr=0.8, k=0)
    snapshot = optimizer_rate_snapshot(optimizer)
    group.update(d=5e-4, lr=0.0, k=1)
    beta1, beta2 = config.betas
    expected = 1e-4 * 0.8 * (1 - beta2) ** 0.5 / (1 - beta1)
    rates = optimizer_learning_rates(config, optimizer, before_step=snapshot)
    assert rates == pytest.approx({"adapter": expected})


def test_resume_rejects_silent_hyperparameter_overrides():
    pytest.importorskip("prodigyopt")
    config = OptimizerConfig(type="prodigy", d_coef=2)
    optimizer = build_optimizer(config, [{"params": [nn.Parameter(torch.ones(2))]}])
    expected = optimizer_hyperparameter_snapshot(config, optimizer)
    optimizer.param_groups[0]["d_coef"] = 1
    with pytest.raises(ValueError, match="d_coef.*differs"):
        validate_optimizer_runtime(config, optimizer, expected_groups=expected)


def test_ppsf_resume_rejects_incompatible_internal_version():
    pytest.importorskip("prodigyplus")
    config = OptimizerConfig(type="prodigy_plus_sf")
    optimizer = build_optimizer(config, [{"params": [nn.Parameter(torch.ones(2))]}])
    optimizer.param_groups[0].pop("optimiser_version")
    with pytest.raises(ValueError, match="PPSF optimizer version"):
        validate_optimizer_runtime(config, optimizer)


def test_in_place_args_mutation_is_revalidated_by_factory():
    config = OptimizerConfig(type="prodigy_plus_sf")
    config.args["fused_back_pass"] = True
    with pytest.raises(ValueError, match="fused_back_pass"):
        build_optimizer(config, [{"params": [nn.Parameter(torch.ones(2))]}])
