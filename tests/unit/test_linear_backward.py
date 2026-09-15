"""Actual native dispatch, BF16 operand rounding and FP32 backward contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import resolve_training_compute_config
from ypuddin.train.linear_backward import (
    install_linear_bf16_forward_fp32_backward,
    validate_linear_backward_installation,
)
from ypuddin.train.trainer import Trainer


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
                    [
                        (v.dtype, tuple(v.shape), tuple(v.stride()))
                        for v in args
                        if isinstance(v, torch.Tensor)
                    ],
                )
            )
        return result


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("master_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_native_layout_hooks_and_fp32_backward(frozen, master_dtype, input_dtype):
    torch.manual_seed(331)
    source = nn.Linear(7, 5).to(master_dtype)
    layer = FrozenLinear.from_linear(source) if frozen else source
    x = torch.randn(3, 2, 7, dtype=input_dtype).transpose(0, 1).requires_grad_()
    original_ids = [id(p) for p in layer.parameters()]
    original_state = {k: v.clone() for k, v in layer.state_dict().items()}
    hooks = []
    pre = layer.register_forward_pre_hook(lambda *_: hooks.append("pre"))
    post = layer.register_forward_hook(lambda *_: hooks.append("post"))
    native = Contractions()
    with torch.autocast("cpu", dtype=torch.bfloat16), native:
        expected = layer(x)
    hooks.clear()
    restore, counts = install_linear_bf16_forward_fp32_backward(layer)
    try:
        validate_linear_backward_installation(layer, counts)
        forward = Contractions()
        with torch.autocast("cpu", dtype=torch.bfloat16), forward:
            actual = layer(x)
        assert torch.equal(actual, expected) and actual.dtype == torch.bfloat16
        assert native.rows == forward.rows
        assert hooks == ["pre", "post"]
        assert original_ids == [id(p) for p in layer.parameters()]
        dy = torch.randn_like(actual)
        backward = Contractions()
        with backward:
            actual.backward(dy)
        assert backward.rows and all(
            dtype == torch.float32 for _, operands in backward.rows for dtype, _, _ in operands
        )
        # Independent reference uses native autograd over the BF16-rounded
        # operands in FP32, including the original returned gradient dtypes.
        rx = x.detach().to(torch.bfloat16).float().requires_grad_()
        rw = layer.weight.detach().to(torch.bfloat16).float().requires_grad_()
        rb = layer.bias.detach().to(torch.bfloat16).float().requires_grad_()
        gradients = torch.autograd.grad(torch.nn.functional.linear(rx, rw, rb), (rx, rw, rb), dy.float())
        assert torch.equal(x.grad, gradients[0].to(input_dtype))
        if not frozen:
            assert torch.equal(layer.weight.grad, gradients[1].to(master_dtype))
            assert torch.equal(layer.bias.grad, gradients[2].to(master_dtype))
        assert original_state.keys() == layer.state_dict().keys()
        assert all(torch.equal(v, layer.state_dict()[k]) for k, v in original_state.items())
    finally:
        restore()
        pre.remove()
        post.remove()
    assert "forward" not in layer.__dict__


@pytest.mark.parametrize("bf16", [False, True])
def test_no_grad_and_non_bf16_keep_original_dispatch(bf16):
    layer = nn.Linear(7, 5)
    reference = copy.deepcopy(layer)
    x = torch.randn(3, 2, 7).transpose(0, 1)
    native = Contractions()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16), native:
        expected = layer(x)
    restore, _ = install_linear_bf16_forward_fp32_backward(layer)
    try:
        patched = Contractions()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16), patched:
            actual = layer(x)
        assert torch.equal(expected, actual) and native.rows == patched.rows
        assert not actual.requires_grad
        if not bf16:
            x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
            assert torch.equal(layer(x1), reference(x2))
            layer(x1).sum().backward()
            reference(x2).sum().backward()
            assert torch.equal(x1.grad, x2.grad)
            assert torch.equal(layer.weight.grad, reference.weight.grad)
    finally:
        restore()


def test_vector_without_bias_and_nonreentrant_checkpoint():
    torch.manual_seed(445)
    direct = nn.Sequential(nn.Linear(7, 9, bias=False), nn.SiLU(), nn.Linear(9, 5))
    recomputed = copy.deepcopy(direct)
    x1 = torch.randn(7, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    r1, _ = install_linear_bf16_forward_fp32_backward(direct)
    r2, _ = install_linear_bf16_forward_fp32_backward(recomputed)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            a = direct(x1)
            b = checkpoint(recomputed, x2, use_reentrant=False)
        assert torch.equal(a, b)
        ga = torch.autograd.grad(a.float().square().mean(), [x1, *direct.parameters()])
        gb = torch.autograd.grad(b.float().square().mean(), [x2, *recomputed.parameters()])
        assert all(torch.equal(left, right) for left, right in zip(ga, gb, strict=True))
    finally:
        r2()
        r1()


def test_installation_rejects_duplicate_and_detects_replaced_forward():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    restore, counts = install_linear_bf16_forward_fp32_backward(model)
    with pytest.raises(ValueError, match="already installed"):
        install_linear_bf16_forward_fp32_backward(model)
    validate_linear_backward_installation(model, counts)
    model[1].forward = lambda x: x
    with pytest.raises(ValueError, match="forward was replaced"):
        validate_linear_backward_installation(model, counts)
    restore()
    assert model[1].forward(None) is None  # Preserve another owner's replacement.
    with pytest.raises(ValueError, match="no Linear"):
        install_linear_bf16_forward_fp32_backward(nn.Identity())


def test_trainer_policy_guards_missing_installation_dtype_scope_and_tampering(monkeypatch):
    import ypuddin.train.trainer as module

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "krea2"},
            "training": {"mode": "full"},
            "loop": {
                "gpu_count": 2,
                "distributed_strategy": "fsdp",
                "deterministic": True,
                "mixed_precision": "bf16",
            },
        }
    )
    trainer = object.__new__(Trainer)
    trainer.cfg, trainer.compute_policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    trainer.device = torch.device("cuda")  # Metadata only; no CUDA operation.
    trainer.loaded = SimpleNamespace(backbone=nn.Linear(3, 4))
    with pytest.raises(ValueError, match="未完整安装"):
        trainer._validate_training_compute_policy()
    with pytest.raises(ValueError, match="FSDP"):
        trainer._place_training_model()
    restore, trainer._linear_backward_counts = install_linear_bf16_forward_fp32_backward(
        trainer.loaded.backbone
    )
    try:
        trainer._validate_training_compute_policy()
        trainer.cfg.loop.mixed_precision = "no"
        with pytest.raises(ValueError, match="设置不一致"):
            trainer._validate_training_compute_policy()
        trainer.cfg.loop.mixed_precision = "bf16"
        trainer.cfg.training.train_text_encoder = True
        with pytest.raises(ValueError, match="设置不一致"):
            trainer._validate_training_compute_policy()
        trainer.cfg.training.train_text_encoder = False
        trainer.loaded.backbone.forward = lambda x: x
        with pytest.raises(ValueError, match="forward was replaced"):
            trainer._validate_training_compute_policy()
    finally:
        restore()
    trainer.compute_policy = None
    trainer._validate_training_compute_policy()  # Default/legacy behavior has no new operator requirement.
