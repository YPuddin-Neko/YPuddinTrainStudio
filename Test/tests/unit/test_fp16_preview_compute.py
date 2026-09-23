"""Actual FP16 rounding, native training and exception cleanup for DTK previews."""
import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lokr import LoKr
from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    DTK_SDXL_LOKR_SINGLE_FP16_PREVIEW_POLICY_ID,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)
from ypuddin.train.preview_compute import preview_linear_compute


def config():
    return TrainConfig.model_validate({
        "model": {"family": "sdxl", "dtype": "bf16", "attention": "sdpa", "sdxl_max_token_length": 150},
        "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": False},
        "memory": {"base_precision": "bf16", "activation_checkpointing": "none", "allow_tf32": False},
        "adapter": {"algo": "lokr", "mode": "bypass", "param_dtype": "fp32"},
        "loop": {"gpu_count": 1, "distributed_strategy": "ddp", "mixed_precision": "fp16", "deterministic": True},
    })


def policy():
    return resolve_training_compute_config(config(), "cuda", "linux-dtk")[1]


class MatmulDtypes(TorchDispatchMode):
    def __init__(self):
        self.dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}:
            self.dtypes.extend(x.dtype for x in args if isinstance(x, torch.Tensor))
        return func(*args, **(kwargs or {}))


def test_policy_keeps_fp16_training_and_rejects_old_resume():
    cfg = config()
    before = cfg.model_dump()
    effective, selected = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert cfg.model_dump() == before == effective.model_dump()
    assert selected["id"] == DTK_SDXL_LOKR_SINGLE_FP16_PREVIEW_POLICY_ID
    assert selected["mixed_precision"] == "fp16"
    assert "linear_backward_implementation" not in selected
    assert "adapter_implementation" not in selected
    validate_resume_compute_policy(selected, copy.deepcopy(selected))
    with pytest.raises(ValueError, match="计算"):
        validate_resume_compute_policy(selected, None)


@pytest.mark.parametrize("section,field,value", [
    ("model", "family", "anima"), ("model", "dtype", "fp16"),
    ("model", "sdxl_max_token_length", 75), ("model", "sdxl_max_token_length", 225),
    ("memory", "base_precision", "fp8_e4m3"), ("memory", "compile", True),
    ("memory", "blocks_to_swap", 1), ("memory", "activation_checkpointing", "unsloth"),
    ("loop", "gpu_count", 2), ("loop", "distributed_strategy", "fsdp"),
    ("loop", "deterministic", False), ("training", "train_text_encoder", True),
    ("adapter", "algo", "lora"), ("adapter", "param_dtype", "bf16"),
])
def test_unmeasured_fp16_routes_do_not_select_preview_policy(section, field, value):
    cfg = config()
    setattr(getattr(cfg, section), field, value)
    selected = resolve_training_compute_config(cfg, "cuda", "linux-dtk")[1]
    assert (selected or {}).get("id") != DTK_SDXL_LOKR_SINGLE_FP16_PREVIEW_POLICY_ID


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_fp16_base_preview_uses_rounded_operands_and_cleans_hooks(frozen, bias, contiguous):
    torch.manual_seed(94)
    source = nn.Linear(7, 9, bias=bias)
    layer = FrozenLinear.from_linear(source) if frozen else source
    x = torch.randn(3, 2, 7).transpose(0, 1)
    if contiguous:
        x = x.contiguous()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        native = layer(x)
        state = torch.get_rng_state().clone()
        with preview_linear_compute(policy(), layer), MatmulDtypes() as trace:
            actual = layer(x)
        assert trace.dtypes and set(trace.dtypes) == {torch.float32}
        assert actual.dtype == torch.float16
        assert torch.equal(state, torch.get_rng_state())
        assert torch.equal(layer(x), native)
        with torch.autocast("cpu", enabled=False):
            if contiguous:
                expected = F.linear(x.half().float(), source.weight.half().float(),
                                    source.bias.half().float() if bias else None).half()
            else:
                # Native strided Linear dispatch rounds matmul before adding
                # bias. Fusing the bias into FP32 would be a different recipe.
                expected = (x.half().float() @ source.weight.half().float().T).half()
                if bias:
                    expected += source.bias.half()
        assert torch.equal(actual, expected)
    assert not layer._forward_hooks and not layer._forward_pre_hooks


def test_fp16_context_preserves_native_lokr_training_gradients_and_adapter_preview():
    torch.manual_seed(123)
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), LoKr(8, 8, rank=2), mode="bypass")
    original = copy.deepcopy(layer)
    x = torch.randn(2, 3, 8)
    with torch.autocast("cpu", dtype=torch.float16):
        before = original(x)
        with preview_linear_compute(policy(), layer):
            after = layer(x)
    before.float().square().sum().backward()
    after.float().square().sum().backward()
    assert torch.equal(before, after)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(original.adapter.parameters(), layer.adapter.parameters(), strict=True))
    layer.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16), preview_linear_compute(policy(), layer), MatmulDtypes() as trace:
        actual = layer.adapter.delta_apply(x.half())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        expected = original.adapter.delta_apply(x.half())
    assert torch.equal(actual, expected)
    assert trace.dtypes and set(trace.dtypes) == {torch.float16}
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in layer.modules())


def test_exception_leaves_no_hooks_or_dispatch_modes():
    layer = nn.Linear(7, 9)
    with torch.no_grad(), pytest.raises(RuntimeError), preview_linear_compute(policy(), layer):
        layer(torch.randn(2, 3))
    assert not layer._forward_hooks and not layer._forward_pre_hooks
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16), MatmulDtypes() as trace:
        layer(torch.randn(2, 7))
    assert trace.dtypes and set(trace.dtypes) == {torch.float16}


def test_nested_preview_and_disabled_context_restore_native_dispatch():
    layer = nn.Linear(7, 9)
    x = torch.randn(2, 7)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        native = layer(x)
        with preview_linear_compute(policy(), layer):
            hooks = tuple(layer._forward_hooks)
            with preview_linear_compute(policy(), layer), MatmulDtypes() as stable:
                layer(x)
                assert tuple(layer._forward_hooks) == hooks
            with preview_linear_compute(None), MatmulDtypes() as disabled:
                assert torch.equal(layer(x), native)
            with MatmulDtypes() as restored:
                layer(x)
        assert set(stable.dtypes) == set(restored.dtypes) == {torch.float32}
        assert set(disabled.dtypes) == {torch.float16}
    assert not layer._forward_hooks and not layer._forward_pre_hooks
