"""The MPS position table preserves the original CPU float64 arithmetic."""

import pytest
import torch

from ypuddin.models.krea2.vendor.krea2_mmdit import PositionalEncoding, rope, ropeapply


def reference(pos, dim, theta, ntk):
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim
    angles = torch.einsum("...n,d->...nd", pos, 1.0 / ((theta * ntk) ** scale))
    return (
        torch.stack([angles.cos(), -angles.sin(), angles.sin(), angles.cos()], -1)
        .reshape(*angles.shape, 2, 2)
        .float()
    )


@pytest.mark.parametrize("dim,theta,ntk", [(16, 1e3, 1.0), (64, 1e4, 1.5), (128, 1e6, 0.75)])
def test_cpu_rope_retains_exact_frequency_values(dim, theta, ntk):
    pos = torch.tensor([[0, 1, 127, 100000], [5, 25, 1024, 250000]], dtype=torch.float32)
    torch.testing.assert_close(rope(pos, dim, theta, ntk), reference(pos, dim, theta, ntk), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires an actual Apple MPS device")
@pytest.mark.parametrize("dim,theta,ntk", [(64, 1e3, 1.0), (128, 1e4, 1.5)])
def test_mps_rope_returns_exact_fp32_values_and_preserves_rotation_gradients(dim, theta, ntk):
    pos = torch.tensor([[0, 1, 127, 100000], [5, 25, 1024, 250000]], dtype=torch.float32)
    freqs = rope(pos.to("mps"), dim, theta, ntk)
    expected = reference(pos, dim, theta, ntk)
    assert freqs.device.type == "mps" and freqs.dtype == torch.float32
    torch.testing.assert_close(freqs.cpu(), expected, rtol=0, atol=0)
    q = torch.randn(2, 2, 4, dim, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    qm, km = [x.detach().to("mps").requires_grad_() for x in (q, k)]
    actual = ropeapply(qm, km, freqs)
    ref = ropeapply(q, k, expected)
    for a, b in zip(actual, ref, strict=True):
        torch.testing.assert_close(a.cpu(), b, rtol=1e-6, atol=1e-6)
    sum(t.square().sum() for t in actual).backward()
    sum(t.square().sum() for t in ref).backward()
    torch.mps.synchronize()
    for a, b in ((qm, q), (km, k)):
        assert a.grad is not None and bool(torch.isfinite(a.grad).all())
        torch.testing.assert_close(a.grad.cpu(), b.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires an actual Apple MPS device")
def test_actual_mps_multi_axis_position_table_matches_cpu():
    pos = torch.tensor([[[0, 0, 0], [0, 1, 128], [0, 255, 7]]], dtype=torch.float32)
    layer = PositionalEncoding(128, [32, 48, 48], theta=1e3)
    expected = layer(pos)
    actual = layer(pos.to("mps"))
    assert actual.device.type == "mps" and actual.shape == (1, 3, 64, 2, 2)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
