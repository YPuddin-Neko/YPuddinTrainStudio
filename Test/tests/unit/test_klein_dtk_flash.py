"""Processor math/layout parity; real vendor kernels are covered on the HIP host."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("diffusers")
from diffusers import Flux2Transformer2DModel
from diffusers.models.attention_dispatch import _AttentionBackendRegistry

from ypuddin.models.flux2 import attention as compat
from ypuddin.models.flux2.family import _set_attention_backend

CONFIG = dict(
    in_channels=128,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=24,
    timestep_guidance_channels=8,
    axes_dims_rope=(2, 2, 2, 2),
    mlp_ratio=2,
    guidance_embeds=False,
)


def public_stub(q, k, v, *, deterministic=False, **kwargs):
    raise AssertionError("CPU tests must not call a GPU kernel")


def vendor(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.3.26093")
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_func=public_stub))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_both_processors_match_upstream_forward_and_all_gradients(monkeypatch, checkpointing):
    vendor(monkeypatch)
    torch.manual_seed(52)
    model = Flux2Transformer2DModel(**CONFIG).train()
    model.set_attention_backend("native")
    reference = copy.deepcopy(model)
    before = _AttentionBackendRegistry.get_active_backend()
    calls = []

    def cpu_kernel(q, k, v):
        calls.append(tuple(q.shape))
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(compat, "_flash", cpu_kernel)
    _set_attention_backend(model, "flash_attn", "cuda:0")
    assert _AttentionBackendRegistry.get_active_backend() == before
    compat.validate_dtk_flash(model)
    if checkpointing:
        model.enable_gradient_checkpointing()
        reference.enable_gradient_checkpointing()
    states = torch.randn(1, 4, 128)
    text = torch.randn(1, 3, 24)
    outputs, gradients = [], []
    for m in (reference, model):
        x, e = states.clone().requires_grad_(), text.clone().requires_grad_()
        result = m(
            hidden_states=x,
            encoder_hidden_states=e,
            timestep=torch.tensor([0.4]),
            img_ids=torch.arange(16).reshape(4, 4),
            txt_ids=torch.arange(12).reshape(3, 4),
            return_dict=False,
        )[0]
        result.square().mean().backward()
        outputs.append(result.detach())
        gradients.append({**{n: p.grad for n, p in m.named_parameters()}, "input": x.grad, "text": e.grad})
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    assert len(calls) == (4 if checkpointing else 2)
    for name, grad in gradients[0].items():
        if grad is None:
            assert gradients[1][name] is None
        else:
            torch.testing.assert_close(gradients[1][name], grad, rtol=0, atol=0, msg=name)
    _set_attention_backend(model, "sdpa", "cuda:0")
    assert all(type(p) in compat._PROCESSORS for p in model.attn_processors.values())


def test_missing_wrapped_api_is_not_needed_and_install_is_atomic(monkeypatch):
    vendor(monkeypatch)
    model = Flux2Transformer2DModel(**CONFIG)
    compat.install_dtk_flash(model)
    original = model.attn_processors
    compat.install_dtk_flash(model)
    assert model.attn_processors == original
    processors = list(original.values())
    processors[-1]._parallel_config = object()
    with pytest.raises(ValueError, match="context parallelism"):
        compat.install_dtk_flash(model)
    assert model.attn_processors == original
    with pytest.raises(ValueError, match="attention masks"):
        processors[0](None, None, attention_mask=torch.ones(1))


def test_unusable_public_api_is_rejected_before_changing_model(monkeypatch):
    vendor(monkeypatch)
    model = Flux2Transformer2DModel(**CONFIG)
    original = model.attn_processors
    monkeypatch.setattr(sys.modules["flash_attn"], "flash_attn_func", lambda q, k, v: None)
    with pytest.raises(RuntimeError, match="deterministic backward"):
        compat.install_dtk_flash(model)
    assert model.attn_processors == original


def test_gpu_call_rejects_cpu_instead_of_falling_back():
    q = torch.zeros(1, 2, 3, 8)
    with pytest.raises(ValueError, match="HIP GPU"):
        compat._flash(q, q, q)
