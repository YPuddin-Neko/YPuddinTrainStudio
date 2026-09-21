"""CPU dispatch simulations verify gradients/fallback contracts without optional CUDA packages."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

attn = importlib.import_module("ypuddin.models.anima.vendor.attention")


@pytest.mark.parametrize("deterministic", [False, True])
def test_flash_dispatch_honors_global_determinism_and_preserves_gradients(monkeypatch, deterministic):
    seen = []

    def flash(q, k, v, *, dropout_p, causal, deterministic):
        seen.append((dropout_p, causal, deterministic))
        return q + 2 * k + 3 * v

    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_func=flash))
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        tensors = [torch.randn(1, 2, 3, 8, requires_grad=True) for _ in range(3)]
        result = attn._external_attention("flash_attn", *tensors, 0.0)
        assert result.shape == tensors[0].shape
        result.sum().backward()
        assert seen == [(0.0, False, deterministic)]
        for multiplier, tensor in enumerate(tensors, 1):
            torch.testing.assert_close(tensor.grad, torch.full_like(tensor, multiplier))
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def test_flash_deterministic_kernel_failure_is_not_silently_retried(monkeypatch):
    def unsupported(*args, **kwargs):
        assert kwargs["deterministic"] is True
        raise RuntimeError("deterministic backward is unsupported on this device")

    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_func=unsupported))
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        q = torch.randn(1, 2, 3, 8)
        with pytest.raises(RuntimeError, match="deterministic backward is unsupported"):
            attn._external_attention("flash_attn", q, q, q, 0.0)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize("backend", ["sage", "xformers", "flash_attn"])
def test_optional_backends_keep_cpu_mask_and_gqa_gradients(backend):
    q = torch.randn(2, 4, 5, 8, requires_grad=True)
    k = torch.randn(2, 2, 7, 8, requires_grad=True)
    v = torch.randn(2, 2, 7, 8, requires_grad=True)
    mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    mask[:, :, :, -2:] = False
    out = attn._sdpa(q, k, v, backend, mask)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=mask
    )
    torch.testing.assert_close(out, ref)
    actual = torch.autograd.grad(out.sum(), (q, k, v), retain_graph=True)
    expected = torch.autograd.grad(ref.sum(), (q, k, v))
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("backend", ["xformers", "flash_attn"])
def test_external_cuda_dispatch_preserves_backward_and_unsupported_kernel_falls_back(monkeypatch, backend):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    called = []

    def fake(name, q, k, v, dropout):
        called.append(name)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout)

    monkeypatch.setattr(attn, "_external_attention", fake)
    q = torch.randn(1, 2, 4, 8, dtype=torch.float16, requires_grad=True)
    attn._sdpa(q, q, q, backend).float().sum().backward()
    assert called == [backend] and q.grad.abs().sum() > 0

    def missing(*args):
        raise NotImplementedError("unsupported head dimension")

    monkeypatch.setattr(attn, "_external_attention", missing)
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.testing.assert_close(attn._sdpa(q, q, q, backend), ref)


def test_sage_never_runs_during_checkpoint_training_but_runs_in_explicit_sampling(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    calls = []

    def sage(q, k, v, **kwargs):
        calls.append("sage")
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    monkeypatch.setattr(attn, "_sageattn", sage)
    q = torch.randn(1, 2, 4, 8, dtype=torch.float16, requires_grad=True)
    out = checkpoint(lambda x: attn._sdpa(x, x, x, "sage"), q, use_reentrant=True)
    out.float().sum().backward()
    assert not calls and q.grad.abs().sum() > 0
    with attn.sampling_attention(True), torch.no_grad():
        attn._sdpa(q, q, q, "sage")
    assert calls == ["sage"]
