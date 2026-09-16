"""Real CPU LoRA precision, random-state and recomputation contracts.

These tests exercise actual tensor operations, not DTK hardware determinism.
"""

import copy
import io

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lora import LoRA
from ypuddin.config.compute_policy import (
    BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID,
    BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
    BF16_LORA_FP32_IMPLEMENTATION_ID,
    FP32_CONV_IMPLEMENTATION_ID,
)
from ypuddin.train.text_adapter_compute import (
    install_text_adapter_compute,
    validate_text_adapter_compute,
)


class Contractions(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if str(func) in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}:
            self.rows.append(
                (
                    str(func),
                    tuple(
                        (v.dtype, tuple(v.shape), tuple(v.stride()))
                        for v in args
                        if isinstance(v, torch.Tensor)
                    ),
                )
            )
        return result


def _policy(*, native_backbone=False, conv=False):
    result = {
        "id": "unit-text-lora-precision-contract",
        "operator_components": ["backbone", "text_encoder"],
        "trainable_components": ["text_encoder"],
        "linear_backward_implementation": (
            BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID
            if native_backbone
            else BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
        ),
        "text_linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        "adapter_implementation": BF16_LORA_FP32_IMPLEMENTATION_ID,
    }
    if conv:
        result["conv_implementation"] = FP32_CONV_IMPLEMENTATION_ID
    return result


class TinyTextTraining(nn.Module):
    def __init__(self, *, dropout=0.0, rank_dropout=0.0):
        super().__init__()
        adapter = LoRA(5, 7, rank=4, alpha=4, dropout=dropout, rank_dropout=rank_dropout)
        # Nonzero up weights expose both contractions and their VJPs.
        nn.init.normal_(adapter.up, std=0.2)
        self.text_encoder = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(7, 5)), adapter, mode="bypass")
        self.backbone = nn.Linear(5, 3).requires_grad_(False)

    def components(self):
        return {"backbone": self.backbone, "text_encoder": self.text_encoder}

    def forward(self, x):
        return self.backbone(self.text_encoder(x).tanh())


def _all_fp32_contractions(record):
    assert record.rows
    assert all(dtype == torch.float32 for _, operands in record.rows for dtype, _, _ in operands)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_lora_real_precision_and_cast_gradient_boundaries(input_dtype, autocast):
    torch.manual_seed(1412)
    model = TinyTextTraining()
    native = copy.deepcopy(model.text_encoder.adapter)
    adapter = model.text_encoder.adapter
    x = torch.randn(3, 2, 7, dtype=input_dtype).transpose(0, 1).requires_grad_()
    native_x = x.detach().clone().requires_grad_()
    restore, _ = install_text_adapter_compute(model.components(), _policy())
    try:
        forward = Contractions()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), forward:
            actual = adapter.delta_apply(x)
        dy = torch.randn_like(actual)
        backward = Contractions()
        with backward:
            actual.backward(dy)
        assert x.grad.dtype == input_dtype
        assert adapter.down.grad.dtype == adapter.up.grad.dtype == torch.float32
        _all_fp32_contractions(forward)
        _all_fp32_contractions(backward)
        if input_dtype == torch.float32 and not autocast:
            # This supported path must remain the original FP32 implementation.
            expected = native.delta_apply(native_x)
            expected.backward(dy)
            assert actual.dtype == torch.float32 and torch.equal(actual, expected)
            assert torch.equal(x.grad, native_x.grad)
            assert torch.equal(adapter.down.grad, native.down.grad)
            assert torch.equal(adapter.up.grad, native.up.grad)
            return
        # Closed-form oracle includes original .to(x.dtype) CastBackward:
        # BF16 x returns BF16-rounded parameter gradients to FP32 masters;
        # FP32 x under autocast returns unrounded FP32 parameter gradients.
        effective_x = x.detach().bfloat16().float()
        down = adapter.down.detach().to(input_dtype).bfloat16().float()
        up = adapter.up.detach().to(input_dtype).bfloat16().float()
        hidden = (effective_x @ down.T).bfloat16()
        expected = (hidden.float() @ up.T).bfloat16()
        d_hidden = (dy.float() @ up).bfloat16()
        expected_up = (dy.float().reshape(-1, 5).T @ hidden.float().reshape(-1, 4)).to(input_dtype).float()
        expected_down = (
            (d_hidden.float().reshape(-1, 4).T @ effective_x.reshape(-1, 7)).to(input_dtype).float()
        )
        assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)
        assert torch.equal(x.grad, (d_hidden.float() @ down).to(input_dtype))
        assert torch.equal(adapter.down.grad, expected_down)
        assert torch.equal(adapter.up.grad, expected_up)
    finally:
        restore()


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_no_grad_keeps_native_dispatch_output_and_random_consumption(input_dtype, autocast):
    torch.manual_seed(213)
    model = TinyTextTraining(dropout=0.3, rank_dropout=0.25)
    adapter = model.text_encoder.adapter
    x = torch.randn(2, 3, 7, dtype=input_dtype)
    before = torch.get_rng_state().clone()
    native = Contractions()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), native:
        expected = adapter.delta_apply(x)
    expected_rng = torch.get_rng_state().clone()
    restore, _ = install_text_adapter_compute(model.components(), _policy())
    try:
        torch.set_rng_state(before)
        observed = Contractions()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), observed:
            actual = adapter.delta_apply(x)
        assert not actual.requires_grad
        assert torch.equal(actual, expected)
        assert observed.rows == native.rows
        assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        restore()


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_dropout_keeps_native_rng_and_repeats_output_and_gradients(input_dtype, autocast):
    torch.manual_seed(774)
    model = TinyTextTraining(dropout=0.3, rank_dropout=0.25)
    adapter = model.text_encoder.adapter
    native = copy.deepcopy(adapter)
    x = torch.randn(2, 3, 7, dtype=input_dtype)
    before = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        native.delta_apply(x)
    native_rng = torch.get_rng_state().clone()
    restore, _ = install_text_adapter_compute(model.components(), _policy())
    try:
        results = []
        for _ in range(2):
            adapter.zero_grad(set_to_none=True)
            torch.set_rng_state(before)
            xx = x.clone().requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                output = adapter.delta_apply(xx)
            output.float().square().sum().backward()
            assert torch.equal(torch.get_rng_state(), native_rng)
            results.append(
                (output.detach().clone(), xx.grad.clone(), *(p.grad.clone() for p in adapter.parameters()))
            )
        assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))
    finally:
        restore()


@pytest.mark.parametrize("autocast", [False, True])
def test_nonreentrant_block_checkpoint_preserves_output_gradients_and_rng(autocast):
    torch.manual_seed(8731)
    direct = TinyTextTraining(dropout=0.2, rank_dropout=0.25)
    recomputed = copy.deepcopy(direct)
    rd, _ = install_text_adapter_compute(direct.components(), _policy())
    rc, _ = install_text_adapter_compute(recomputed.components(), _policy())
    x = torch.randn(2, 3, 7)
    rng = torch.get_rng_state().clone()
    results = []
    try:
        for model, use_checkpoint in ((direct, False), (recomputed, True)):
            torch.set_rng_state(rng)
            xx = x.clone().requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                out = checkpoint(model, xx, use_reentrant=False) if use_checkpoint else model(xx)
            out.float().square().mean().backward()
            assert all(p.grad is None for p in model.backbone.parameters())
            results.append(
                (
                    out.detach(),
                    xx.grad,
                    *(p.grad for p in model.text_encoder.adapter.parameters()),
                    torch.get_rng_state(),
                )
            )
        assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))
    finally:
        rc()
        rd()


def test_installation_preserves_parameter_state_and_call_hooks_and_detects_tampering():
    model = TinyTextTraining()
    modules = model.components()
    ids = {name: id(p) for name, p in model.named_parameters()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    calls = []
    hook = model.text_encoder.register_forward_hook(lambda *_: calls.append("text"))
    restore, counts = install_text_adapter_compute(modules, _policy())
    try:
        assert counts["text_encoder"] == {"linear": {"nn_linear": 0, "frozen_linear": 1}, "lora": 1}
        validate_text_adapter_compute(modules, counts, _policy())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(torch.randn(2, 7)).float().sum().backward()
        assert calls == ["text"]
        assert ids == {name: id(p) for name, p in model.named_parameters()}
        assert state.keys() == model.state_dict().keys()
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
        with pytest.raises(ValueError, match="already installed"):
            install_text_adapter_compute(modules, _policy())
        validate_text_adapter_compute(modules, counts, _policy())

        def replacement(x):
            return x

        model.text_encoder.adapter.delta_apply = replacement
        with pytest.raises(ValueError, match="缺失或被替换"):
            validate_text_adapter_compute(modules, counts, _policy())
        restore()
        assert model.text_encoder.adapter.delta_apply is replacement
    finally:
        restore()
        hook.remove()
    assert "forward" not in model.backbone.__dict__
    assert "forward" not in model.text_encoder.base.__dict__


@pytest.mark.parametrize("native_backbone", [False, True])
def test_components_use_selected_linear_policy_and_restore(native_backbone):
    model = TinyTextTraining()
    native = copy.deepcopy(model.backbone)
    x = torch.randn(2, 5, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected_native = native(x)
    policy = _policy(native_backbone=native_backbone)
    restore, counts = install_text_adapter_compute(model.components(), policy)
    try:
        trace = Contractions()
        with torch.autocast("cpu", dtype=torch.bfloat16), trace:
            actual = model.backbone(x)
        if native_backbone:
            assert torch.equal(actual, expected_native)
            assert all(dt == torch.bfloat16 for _, operands in trace.rows for dt, _, _ in operands)
        else:
            _all_fp32_contractions(trace)
        backward = Contractions()
        with backward:
            actual.sum().backward()
        _all_fp32_contractions(backward)
        validate_text_adapter_compute(model.components(), counts, policy)
    finally:
        restore()


def test_sdxl_conv_and_linear_installation_is_atomic_and_uses_fp32_conv():
    model = TinyTextTraining()
    modules = model.components()
    policy = _policy(native_backbone=True, conv=True)
    # Missing required Conv must roll back every already-installed method.
    with pytest.raises(ValueError, match="ordinary nn.Conv2d"):
        install_text_adapter_compute(modules, policy)
    assert "forward" not in model.backbone.__dict__
    assert "delta_apply" not in model.text_encoder.adapter.__dict__
    conv = nn.Conv2d(2, 2, 3, padding=1).requires_grad_(False)
    modules["backbone"] = nn.ModuleDict({"linear": model.backbone, "conv": conv})
    restore, counts = install_text_adapter_compute(modules, policy)
    try:
        x = torch.randn(2, 2, 4, 5, requires_grad=True)
        expected = torch.nn.functional.conv2d(x, conv.weight, conv.bias, padding=1).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = conv(x)
        assert torch.equal(actual, expected)
        assert actual.dtype == torch.bfloat16
        actual.float().square().mean().backward()
        assert x.grad.dtype == torch.float32 and conv.weight.grad is None
        assert counts["backbone"]["conv"] == {"conv2d": 1}
        validate_text_adapter_compute(modules, counts, policy)
    finally:
        restore()


def test_saved_values_reinstall_identity_and_resume_exact_with_dropout():
    torch.manual_seed(4141)
    hot = TinyTextTraining(dropout=0.2, rank_dropout=0.25)
    cold = copy.deepcopy(hot)
    policy = _policy()
    rh, counts = install_text_adapter_compute(hot.components(), policy)
    optimizer = torch.optim.AdamW(hot.text_encoder.adapter.parameters(), lr=0.001)
    x = torch.randn(2, 3, 7)

    def update(model, opt):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(x).float().square().mean()
        loss.backward()
        opt.step()
        return loss.detach().clone()

    rc = None
    try:
        update(hot, optimizer)
        buffer = io.BytesIO()
        torch.save(
            {
                "model": hot.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "policy": policy,
                "counts": counts,
            },
            buffer,
        )
        expected_loss = update(hot, optimizer)
        expected_rng = torch.get_rng_state().clone()
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        # State values alone do not serialize live bound-method identities.
        with pytest.raises(ValueError, match="missing"):
            validate_text_adapter_compute(cold.components(), saved["counts"], saved["policy"])
        cold.load_state_dict(saved["model"])
        rc, cold_counts = install_text_adapter_compute(cold.components(), saved["policy"])
        cold_optimizer = torch.optim.AdamW(cold.text_encoder.adapter.parameters(), lr=0.001)
        cold_optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        actual_loss = update(cold, cold_optimizer)
        assert torch.equal(actual_loss, expected_loss)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert all(
            torch.equal(a, b)
            for a, b in zip(hot.state_dict().values(), cold.state_dict().values(), strict=True)
        )
        for key, hot_state in optimizer.state_dict()["state"].items():
            cold_state = cold_optimizer.state_dict()["state"][key]
            assert hot_state.keys() == cold_state.keys()
            assert all(torch.equal(value, cold_state[name]) for name, value in hot_state.items())
        validate_text_adapter_compute(cold.components(), cold_counts, saved["policy"])
        wrong = {**saved["policy"], "adapter_implementation": "wrong-version"}
        with pytest.raises(ValueError, match="缺失或被替换"):
            validate_text_adapter_compute(cold.components(), cold_counts, wrong)
    finally:
        if rc is not None:
            rc()
        rh()
