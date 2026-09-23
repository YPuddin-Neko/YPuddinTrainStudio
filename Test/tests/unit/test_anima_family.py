"""Anima family wrapper on CPU with tiny random weights (real weights are exercised on GPU machines)."""

import pytest
import torch
from safetensors.torch import save_file

from ypuddin.adapters import inject
from ypuddin.config import AdapterConfig, ModelConfig
from ypuddin.models import TextCond, get_family
from ypuddin.models.anima.family import AnimaLatent, load_dit
from ypuddin.models.anima.vendor.cosmos_dit import ANIMA_2B_CONFIG, Anima
from ypuddin.models.anima.vendor.qwen_image_vae_2d import AutoencoderKLQwenImage2D

TINY_CFG = dict(ANIMA_2B_CONFIG, max_img_h=64, max_img_w=64, model_channels=128, num_blocks=2, num_heads=2)


def test_broken_text_runtime_fails_before_loading_backbone(monkeypatch):
    from ypuddin.config import MemoryConfig
    from ypuddin.models.anima import family

    model = get_family("anima")
    monkeypatch.setattr(model, "validate_config", lambda cfg: [])

    def broken():
        raise RuntimeError("operator torchvision::nms does not exist")

    monkeypatch.setattr(family, "require_qwen3_runtime", broken)
    monkeypatch.setattr(
        family, "load_dit", lambda *a, **kw: pytest.fail("weights loaded before dependency check")
    )
    with pytest.raises(RuntimeError, match="torchvision::nms"):
        model.load(ModelConfig(family="anima"), MemoryConfig(), device="cpu", dtype=torch.float32)


def _tiny_checkpoint(tmp_path, prefix="net."):
    torch.manual_seed(0)
    dit = Anima(**TINY_CFG)
    sd = {prefix + k: v.contiguous() for k, v in dit.state_dict().items()}
    path = tmp_path / "tiny_anima.safetensors"
    save_file(sd, str(path))
    return dit, path


def _cond(b=1, lq=16, lt=16):
    return TextCond(
        {
            "embeds": torch.randn(b, lq, 1024),
            "attn_mask": torch.ones(b, lq, dtype=torch.bool),
            "t5_ids": torch.randint(2, 32000, (b, lt)),
            "t5_mask": torch.ones(b, lt, dtype=torch.bool),
        }
    )


@pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model."])
def test_load_dit_infers_geometry_and_matches_reference(tmp_path, prefix):
    ref, path = _tiny_checkpoint(tmp_path, prefix)
    dit, cfg = load_dit(path, device="cpu", dtype=torch.float32)
    assert cfg["model_channels"] == 128 and cfg["num_blocks"] == 2 and cfg["num_heads"] == 2
    assert all(b.device.type == "cpu" for b in dit.buffers())
    fam = get_family("anima")
    from ypuddin.models.base import LoadedModel

    loaded = LoadedModel(
        backbone=dit, text=None, latent=None, device=torch.device("cpu"), dtype=torch.float32
    )
    x = torch.randn(1, 16, 8, 8)
    t = torch.tensor([0.4])
    cond = _cond()
    out = fam.forward(loaded, x, t, cond)
    assert out.shape == x.shape and torch.isfinite(out).all()
    # identical to calling the reference module the sd-scripts way
    ref.eval()
    with torch.no_grad():
        ref_out = ref(
            x.unsqueeze(2),
            t,
            cond["embeds"],
            padding_mask=torch.zeros(1, 1, 8, 8),
            target_input_ids=cond["t5_ids"],
            target_attention_mask=cond["t5_mask"],
            source_attention_mask=cond["attn_mask"],
        ).squeeze(2)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)


def test_presets_match_expected_module_counts(tmp_path):
    _, path = _tiny_checkpoint(tmp_path)
    fam = get_family("anima")
    presets = fam.presets()
    counts = {}
    for name in presets:
        dit, _ = load_dit(path, device="cpu", dtype=torch.float32)
        aset = inject(
            dit, AdapterConfig(algo="lokr", rank=2, alpha=2.0, preset=name), presets[name], prefix="lora_unet"
        )
        counts[name] = len(aset.layers)
        assert all(
            n.startswith("lora_unet_") for n in [f"lora_unet_{k.replace('.', '_')}" for k in aset.layers]
        )
    blocks = TINY_CFG["num_blocks"]
    assert counts["attn-only"] == blocks * 8  # self+cross × (q,k,v,out)
    assert counts["attn-mlp"] == blocks * 10
    assert counts["full-linear"] == blocks * 16  # + 3 adaln × 2 linears
    assert counts["adapter-only"] == 6 * 10  # 6 adapter blocks × (2 attn × 4 proj + mlp.0 + mlp.2)
    assert counts["with-adapter"] == counts["attn-mlp"] + counts["adapter-only"]


def test_adapted_forward_and_backward(tmp_path):
    _, path = _tiny_checkpoint(tmp_path)
    fam = get_family("anima")
    dit, _ = load_dit(path, device="cpu", dtype=torch.float32)
    aset = inject(
        dit,
        AdapterConfig(algo="lokr", rank="full", alpha=1.0, factor=4, preset="attn-mlp"),
        fam.presets()["attn-mlp"],
    )
    from ypuddin.models.base import LoadedModel

    loaded = LoadedModel(
        backbone=dit, text=None, latent=None, device=torch.device("cpu"), dtype=torch.float32
    )
    dit.train()
    out = fam.forward(loaded, torch.randn(2, 16, 8, 8), torch.tensor([0.2, 0.9]), _cond(b=2))
    out.square().mean().backward()
    grads = [p.grad for p in aset.parameters()]
    assert grads and all(g is not None for g in grads)
    assert all(not p.requires_grad for n, p in dit.named_parameters() if ".adapter." not in n)


def test_meta_backbone_2b_parameter_count():
    fam = get_family("anima")
    with torch.device("meta"):
        model = fam.meta_backbone(ModelConfig(family="anima"))
    n = sum(p.numel() for p in model.parameters())
    assert 1.8e9 < n < 2.6e9, n
    layout = fam.memory_layout_meta(model)
    assert len(layout.blocks) == 28
    names = fam.linear_module_names()
    assert "blocks.0.self_attn.q_proj" in names and "llm_adapter.blocks.0.cross_attn.k_proj" in names


def test_latent_pipeline_wrapper_with_tiny_vae(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "tiny-vae.safetensors"
    path.write_bytes(b"test fixture: in-memory VAE below")
    lat = AnimaLatent(path, device="cpu", dtype=torch.float32)
    lat.vae = AutoencoderKLQwenImage2D(base_dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1).eval()
    px = torch.rand(1, 3, 64, 64) * 2 - 1
    z = lat.encode(px)
    assert z.shape == (1, 16, 8, 8) and z.dtype == torch.float32
    rec = lat.decode(z)
    assert rec.shape == (1, 3, 64, 64) and rec.min() >= -1 and rec.max() <= 1
    lat.unload()
    assert lat.vae is None


@pytest.mark.parametrize("use_2d", [True, False])
def test_lazy_vae_load_and_reload_preserve_training_rng(tmp_path, monkeypatch, use_2d):
    from ypuddin.models.anima.vendor import qwen_image_vae, qwen_image_vae_2d

    path = tmp_path / "vae.safetensors"
    path.write_bytes(b"loader boundary fixture")
    vendor = qwen_image_vae_2d if use_2d else qwen_image_vae
    monkeypatch.setattr(vendor, "load_vae", lambda *args, **kwargs: torch.nn.Linear(8, 8))
    latent = AnimaLatent(path, use_2d=use_2d)
    before = torch.get_rng_state().clone()
    latent._ensure()
    assert torch.equal(torch.get_rng_state(), before)
    latent.unload()
    latent._ensure()
    assert torch.equal(torch.get_rng_state(), before)


def test_validate_config_reports_missing_paths():
    fam = get_family("anima")
    problems = fam.validate_config(ModelConfig(family="anima", dit_path="/nope.safetensors"))
    assert len(problems) == 3 and any("does not exist" in p for p in problems)


@pytest.mark.parametrize("mode", ["block", "unsloth"])
def test_activation_checkpointing_modes_match_plain_gradients(tmp_path, mode):
    _, path = _tiny_checkpoint(tmp_path)
    fam = get_family("anima")
    from ypuddin.models.base import LoadedModel

    grads = {}
    for variant in ("plain", mode):
        torch.manual_seed(1)
        dit, _ = load_dit(path, device="cpu", dtype=torch.float32)
        if variant != "plain":
            dit.enable_gradient_checkpointing(unsloth_offload=variant == "unsloth")
            assert all(b.gradient_checkpointing for b in dit.blocks)
            assert all(b.unsloth_offload_checkpointing == (variant == "unsloth") for b in dit.blocks)
        aset = inject(
            dit,
            AdapterConfig(algo="lokr", rank=2, alpha=2.0, factor=4, preset="attn-mlp"),
            fam.presets()["attn-mlp"],
        )
        loaded = LoadedModel(
            backbone=dit, text=None, latent=None, device=torch.device("cpu"), dtype=torch.float32
        )
        dit.train()
        x = torch.randn(2, 16, 8, 8, generator=torch.Generator().manual_seed(3))
        x.requires_grad_(True)  # the trainer does this whenever checkpointing is on (offload path needs it)
        cond = _cond(b=2)
        out = fam.forward(loaded, x, torch.tensor([0.3, 0.7]), cond)
        out.square().mean().backward()
        grads[variant] = [p.grad.clone() for p in aset.parameters()]
    assert any(g.abs().sum() > 0 for g in grads[mode])  # w1 grads are zero at init (w2 starts at zero)
    for g_ref, g in zip(grads["plain"], grads[mode], strict=True):
        torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-5)


def test_resolve_attention_backend():
    from ypuddin.models.anima.family import AnimaFamily

    assert AnimaFamily.resolve_attention("auto", "cpu") == "torch"
    assert AnimaFamily.resolve_attention("sdpa", "cpu") == "torch"
    with pytest.raises(ValueError, match="requires CUDA"):
        AnimaFamily.resolve_attention("sage", "cpu")
