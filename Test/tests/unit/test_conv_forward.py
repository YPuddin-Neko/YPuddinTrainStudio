"""The versioned SDXL convolution keeps FP32 math and the original output boundary."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import resolve_training_compute_config
from ypuddin.train.conv_forward import install_conv_fp32_forward, validate_conv_forward_installation
from ypuddin.train.trainer import Trainer


@pytest.mark.parametrize(
    "padding_mode,groups,bias",
    [("zeros", 1, True), ("reflect", 2, False), ("replicate", 2, True), ("circular", 1, False)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp32_conv_output_gradient_padding_groups_and_layout(padding_mode, groups, bias, dtype):
    torch.manual_seed(319)
    model = nn.Conv2d(
        4, 6, 3, stride=2, padding=2, dilation=2, padding_mode=padding_mode, groups=groups, bias=bias
    )
    reference = copy.deepcopy(model)
    x = torch.randn(2, 4, 11, 9, dtype=dtype).to(memory_format=torch.channels_last).requires_grad_()
    rx = x.detach().float().requires_grad_()
    original_ids = [id(p) for p in model.parameters()]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    events = []
    pre = model.register_forward_pre_hook(lambda *_: events.append("pre"))
    post = model.register_forward_hook(lambda *_: events.append("post"))
    restore, counts = install_conv_fp32_forward(model)
    try:
        validate_conv_forward_installation(model, counts)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model(x)
        expected = reference(rx).to(torch.bfloat16)
        assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)
        assert events == ["pre", "post"]
        dy = torch.randn_like(actual)
        actual.backward(dy)
        expected.backward(dy)
        assert torch.equal(x.grad, rx.grad.to(dtype))
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            assert torch.equal(a.grad, b.grad)
        assert original_ids == [id(p) for p in model.parameters()]
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
        # Sampling also uses FP32 convolutions; unlike Linear, bypassing in
        # no_grad would change the verified saved preview/next-resume contract.
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            sampled = model(x)
        assert torch.equal(sampled, expected.detach())
        with torch.no_grad():
            outside_autocast = model(x)
        assert outside_autocast.dtype == dtype
        assert torch.equal(outside_autocast, reference(rx).detach().to(dtype))
    finally:
        restore()
        pre.remove()
        post.remove()
    assert "forward" not in model.__dict__


def test_conv_nonreentrant_checkpoint_and_installation_guards():
    layer = nn.Conv2d(2, 3, 3, padding=1)
    other = copy.deepcopy(layer)
    restore, counts = install_conv_fp32_forward(layer)
    restore_other, _ = install_conv_fp32_forward(other)
    try:
        with pytest.raises(ValueError, match="already installed"):
            install_conv_fp32_forward(layer)
        x = torch.randn(1, 2, 6, 5, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            a = layer(x)
            b = checkpoint(other, y, use_reentrant=False)
        assert torch.equal(a, b)
        ga = torch.autograd.grad(a.float().square().mean(), [x, *layer.parameters()])
        gb = torch.autograd.grad(b.float().square().mean(), [y, *other.parameters()])
        assert all(torch.equal(left, right) for left, right in zip(ga, gb, strict=True))
        layer.forward = lambda x: x
        with pytest.raises(ValueError, match="forward was replaced"):
            validate_conv_forward_installation(layer, counts)
    finally:
        restore()
        restore_other()
    assert layer.forward(None) is None


def test_sdxl_product_installs_before_device_move_and_rejects_missing_operator(monkeypatch):
    import ypuddin.train.trainer as module

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "sdxl"},
            "training": {"mode": "full"},
            "loop": {"mixed_precision": "bf16", "deterministic": True},
        }
    )
    trainer = object.__new__(Trainer)
    trainer.cfg, trainer.compute_policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    trainer.device = torch.device("cuda")
    trainer.loaded = SimpleNamespace(backbone=nn.Sequential(nn.Conv2d(2, 2, 1), nn.Linear(2, 2)))
    moved = []

    def metadata_move(device):
        trainer._validate_training_compute_policy()
        moved.append(str(device))

    monkeypatch.setattr(trainer.loaded.backbone, "to", metadata_move)
    trainer._place_training_model()
    assert moved == ["cuda"]
    assert trainer._conv_forward_counts == {"conv2d": 1}
    assert trainer._linear_backward_counts == {"nn_linear": 1, "frozen_linear": 0}
    with pytest.raises(ValueError, match="重复安装"):
        trainer._place_training_model()
    trainer._conv_forward_restore()
    with pytest.raises(ValueError, match="missing"):
        trainer._validate_training_compute_policy()
    trainer._linear_backward_restore()


@pytest.mark.parametrize("changed", ["mode", "dora", "algorithm", "fp8", "param_dtype"])
def test_sdxl_adapter_checks_real_resolved_modules_and_never_rewrites_lokr(monkeypatch, changed):
    import ypuddin.train.trainer as module
    from ypuddin.adapters.frozen import FrozenLinear
    from ypuddin.adapters.linear import AdaptedLinear
    from ypuddin.adapters.lokr import LoKr
    from ypuddin.adapters.lora import LoRA

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "sdxl"},
            "training": {"mode": "adapter"},
            "loop": {"mixed_precision": "bf16", "deterministic": True},
        }
    )
    trainer = object.__new__(Trainer)
    trainer.cfg, trainer.compute_policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    trainer.device = torch.device("cuda")
    base = FrozenLinear.from_linear(nn.Linear(4, 4).to(torch.bfloat16))
    adapted = AdaptedLinear(base, LoKr(4, 4, rank=2), mode="auto")
    delta_apply = adapted.adapter.delta_apply.__func__
    trainer.loaded = SimpleNamespace(backbone=nn.Sequential(nn.Conv2d(4, 4, 1), nn.Linear(4, 4), adapted))
    monkeypatch.setattr(trainer.loaded.backbone, "to", lambda _: None)
    trainer._place_training_model()
    try:
        trainer._validate_training_compute_policy()
        assert trainer._linear_backward_counts == {"nn_linear": 1, "frozen_linear": 1}
        assert adapted.adapter.delta_apply.__func__ is delta_apply
        if changed == "mode":
            adapted.mode = "merged"
        elif changed == "dora":
            adapted.dora = nn.Identity()
        elif changed == "algorithm":
            adapted.adapter = LoRA(4, 4, rank=2)
        elif changed == "fp8":
            adapted.base.weight_scale = torch.ones(())
        else:
            adapted.adapter.to(torch.bfloat16)
        with pytest.raises(ValueError, match="LoKr|FP8"):
            trainer._validate_training_compute_policy()
    finally:
        trainer._conv_forward_restore()
        trainer._linear_backward_restore()
