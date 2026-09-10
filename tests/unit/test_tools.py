import torch
from torch import nn

from ypuddin.adapters import LoKr, LoRA, TargetPreset, inject
from ypuddin.config import AdapterConfig
from ypuddin.tools import (
    extract_from_state_dicts,
    extract_lokr,
    extract_lora,
    merge_into_state_dict,
    nearest_kronecker,
    resize_lora,
)

torch.manual_seed(0)


def test_extract_lora_exact_for_low_rank_delta():
    up, down = torch.randn(32, 4), torch.randn(4, 24)
    delta = up @ down
    res = extract_lora(delta, 4)
    assert res.residual < 1e-5
    mod = LoRA.from_tensors(res.tensors)
    torch.testing.assert_close(mod.delta_weight(), delta, rtol=1e-4, atol=1e-4)


def test_nearest_kronecker_recovers_factors():
    w1, w2 = torch.randn(4, 6), torch.randn(8, 4)
    delta = torch.kron(w1, w2)  # (32, 24)
    r1, r2, residual = nearest_kronecker(delta, 4, 8, 6, 4)
    assert residual < 1e-5
    torch.testing.assert_close(torch.kron(r1, r2), delta, rtol=1e-4, atol=1e-4)


def test_extract_lokr_full_and_lowrank_roundtrip():
    # shapes follow factorization(32, 4) = (4, 8) and factorization(24, 4) = (4, 6): W1 (4,4), W2 (8,6)
    w1, w2 = torch.randn(4, 4), torch.randn(8, 6)
    delta = torch.kron(w1, w2)
    res = extract_lokr(delta, factor=4, rank="full")
    assert res.residual < 1e-5 and res.meta["shape"] == [[4, 8], [4, 6]]
    mod = LoKr.from_tensors(res.tensors, res.meta)
    torch.testing.assert_close(mod.delta_weight(), delta, rtol=1e-4, atol=1e-4)
    # low-rank W2: build a delta whose W2 is rank 2, request rank 2
    w2_lr = torch.randn(8, 2) @ torch.randn(2, 6)
    delta2 = torch.kron(w1, w2_lr)
    res2 = extract_lokr(delta2, factor=4, rank=2)
    assert res2.residual < 1e-4 and "lokr_w2_a" in res2.tensors
    mod2 = LoKr.from_tensors(res2.tensors, res2.meta)
    torch.testing.assert_close(mod2.delta_weight(), delta2, rtol=1e-3, atol=1e-3)


def test_resize_lora_reduces_rank_with_bounded_error():
    up, down = torch.randn(32, 8), torch.randn(8, 24)
    res = resize_lora(down, up, alpha=8.0, new_rank=4)
    assert res.tensors["lora_down.weight"].shape[0] == 4
    assert 0 < res.residual < 1.0


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict({"q": nn.Linear(16, 16, bias=False)}),
                        "mlp": nn.Linear(16, 32, bias=False),
                    }
                )
                for _ in range(2)
            ]
        )


def test_extract_from_state_dicts_and_merge_roundtrip():
    base = _M()
    tuned = _M()
    tuned.load_state_dict(base.state_dict())
    with torch.no_grad():
        tuned.blocks[0].attn.q.weight.add_(torch.randn(16, 16) * 0.01)
    names = [n for n, m in tuned.named_modules() if isinstance(m, nn.Linear)]
    tensors, report = extract_from_state_dicts(base.state_dict(), tuned.state_dict(), algo="lora", rank=16)
    assert set(report) == {"blocks.0.attn.q"} and report["blocks.0.attn.q"]["residual"] < 1e-5
    # a generic delta is not a Kronecker product: the LoKr extraction must report a real residual
    _, rep_lokr = extract_from_state_dicts(
        base.state_dict(), tuned.state_dict(), algo="lokr", rank="full", factor=-1
    )
    assert 0 < rep_lokr["blocks.0.attn.q"]["residual"] < 1
    merged, unmatched = merge_into_state_dict(base.state_dict(), tensors, module_names=names)
    assert not unmatched
    torch.testing.assert_close(
        merged["blocks.0.attn.q.weight"], tuned.blocks[0].attn.q.weight, rtol=1e-3, atol=1e-4
    )
    torch.testing.assert_close(merged["blocks.1.attn.q.weight"], base.blocks[1].attn.q.weight)


def test_merge_matches_adapter_forward():
    model = _M()
    preset = TargetPreset("all", include=("blocks.*",))
    aset = inject(model, AdapterConfig(algo="lokr", rank=4, alpha=2.0), preset, keep_originals=True)
    for layer in aset.layers.values():
        with torch.no_grad():
            for p in layer.adapter.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
    x = torch.randn(3, 16)
    y_adapted = model.blocks[0]["attn"]["q"](x)
    tensors, meta = aset.export_state()
    base_sd = {"blocks.0.attn.q.weight": model.blocks[0]["attn"]["q"].base.weight.clone()}
    merged, _ = merge_into_state_dict(base_sd, tensors, module_names=["blocks.0.attn.q"])
    torch.testing.assert_close(x @ merged["blocks.0.attn.q.weight"].T, y_adapted, rtol=1e-4, atol=1e-4)


def test_extract_and_merge_with_container_prefixed_checkpoints():
    """Official checkpoints wrap keys in 'net.' / 'model.diffusion_model.'; adapter keys must not."""
    base = _M()
    tuned = _M()
    tuned.load_state_dict(base.state_dict())
    with torch.no_grad():
        tuned.blocks[1].attn.q.weight.add_(torch.randn(16, 16) * 0.01)
    for prefix in ("net.", "model.diffusion_model."):
        sd_base = {prefix + k: v for k, v in base.state_dict().items()}
        sd_tuned = {prefix + k: v for k, v in tuned.state_dict().items()}
        tensors, report = extract_from_state_dicts(sd_base, sd_tuned, algo="lora", rank=16)
        assert set(report) == {"blocks.1.attn.q"}
        assert all(k.startswith("lora_unet_blocks_1_attn_q.") for k in tensors)
        merged, unmatched = merge_into_state_dict(sd_base, tensors)
        assert not unmatched and set(merged) == set(sd_base)
        torch.testing.assert_close(
            merged[prefix + "blocks.1.attn.q.weight"], tuned.blocks[1].attn.q.weight, rtol=1e-3, atol=1e-4
        )
