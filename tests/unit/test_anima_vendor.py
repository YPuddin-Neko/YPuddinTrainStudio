"""CPU-only tests for the vendored Anima DiT / Qwen-Image VAE code in ``ypuddin.models.anima.vendor``.

They build *tiny* random-weight instances (the LLM adapter keeps its real 1024-d geometry because it is not
configurable upstream), run training-style forward/backward passes, and check the contracts the rest of the
project relies on: state_dict key names, config inference from checkpoint shapes, gradient checkpointing, the
``max_img_h/w`` pure-extrapolation claim, the 3-D -> 2-D VAE weight conversion, and the attention helper.

``tests/unit/test_anima_vendor.py::test_dump_dit_key_list`` also (re)writes ``docs/reference/anima-dit-keys.txt``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
import torch

from ypuddin.models.anima.vendor import attention as vendored_attention
from ypuddin.models.anima.vendor.cosmos_dit import (
    ANIMA_2B_CONFIG,
    ANIMA_NUM_HEADS_BY_WIDTH,
    Anima,
    LLMAdapter,
    detect_key_prefix,
    get_dit_config,
    infer_dit_config,
    strip_key_prefix,
)
from ypuddin.models.anima.vendor.qwen_image_vae import AutoencoderKLQwenImage, load_safetensors
from ypuddin.models.anima.vendor.qwen_image_vae_2d import (
    AutoencoderKLQwenImage2D,
    convert_3d_state_dict_to_2d,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS_FILE = REPO_ROOT / "docs" / "reference" / "anima-dit-keys.txt"

# Tiny DiT: 128-d, 2 blocks, 2 heads (head_dim 64), real LLM adapter (1024-d, fixed upstream).
TINY_CFG = dict(ANIMA_2B_CONFIG, max_img_h=64, max_img_w=64, model_channels=128, num_blocks=2, num_heads=2)
TINY_VAE_KW = dict(base_dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1)

# Upstream RMSNorm wraps its math in torch.autocast(dtype=float32), which CPU autocast rejects with a warning.
pytestmark = pytest.mark.filterwarnings("ignore:In CPU autocast, but the target dtype is not supported")


def _tiny_batch(seed: int = 0, *, b: int = 1, hw: int = 8, l_q: int = 16, l_t: int = 16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 16, 1, hw, hw, generator=g)  # (B, C, T=1, H, W) latents
    t = torch.rand(b, generator=g)  # sigma in [0, 1]
    qwen = torch.randn(b, l_q, 1024, generator=g)  # Qwen3 last_hidden_state
    qwen_mask = torch.ones(b, l_q, dtype=torch.bool)
    qwen_mask[:, l_q - 3 :] = False  # a little padding on the source side
    qwen = qwen * qwen_mask[..., None]
    t5_ids = torch.randint(2, 32128, (b, l_t), generator=g)
    t5_mask = torch.ones(b, l_t, dtype=torch.bool)
    t5_mask[:, l_t - 4 :] = False
    t5_ids = torch.where(t5_mask, t5_ids, torch.zeros_like(t5_ids))
    padding_mask = torch.zeros(b, 1, hw, hw)
    return x, t, qwen, qwen_mask, t5_ids, t5_mask, padding_mask


@pytest.fixture(scope="module")
def tiny_dit() -> Anima:
    torch.manual_seed(0)
    return Anima(**TINY_CFG)


def _training_forward(model: Anima, batch):
    x, t, qwen, qwen_mask, t5_ids, t5_mask, padding_mask = batch
    return model(
        x,
        t,
        qwen,
        padding_mask=padding_mask,
        target_input_ids=t5_ids,
        target_attention_mask=t5_mask,
        source_attention_mask=qwen_mask,
    )


# --------------------------------------------------------------------------------------------------------- DiT


def test_tiny_dit_training_forward_backward(tiny_dit: Anima):
    model = tiny_dit.train()
    model.zero_grad(set_to_none=True)
    batch = _tiny_batch()
    x = batch[0]
    out = _training_forward(model, batch)
    assert out.shape == x.shape == (1, 16, 1, 8, 8)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    target = torch.randn_like(x) - x  # rectified-flow style target
    loss = torch.nn.functional.mse_loss(out.float(), target)
    loss.backward()

    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert "blocks.0.self_attn.q_proj.weight" in grads
    assert "blocks.1.cross_attn.k_proj.weight" in grads
    assert "llm_adapter.blocks.0.cross_attn.k_proj.weight" in grads  # adapter is trained through forward()
    assert "final_layer.linear.weight" in grads
    assert all(torch.isfinite(g).all() for g in grads.values())
    model.zero_grad(set_to_none=True)


def test_tiny_dit_batched_2d_timesteps_and_none_padding_mask(tiny_dit: Anima):
    model = tiny_dit.eval()
    x, t, qwen, qwen_mask, t5_ids, t5_mask, padding_mask = _tiny_batch(b=2, hw=16, l_q=20, l_t=12)
    with torch.no_grad():
        ref = model(
            x,
            t,
            qwen,
            padding_mask=padding_mask,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
        # (B, T) timesteps and an omitted padding mask (defaults to zeros) must give the same result
        out = model(
            x,
            t[:, None],
            qwen,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
    assert ref.shape == (2, 16, 1, 16, 16)
    torch.testing.assert_close(out, ref)


def test_sampling_path_matches_training_path(tiny_dit: Anima):
    """Pre-adapting the context (sampler) == letting forward() run the adapter (trainer)."""
    model = tiny_dit.eval()
    x, t, qwen, qwen_mask, t5_ids, t5_mask, padding_mask = _tiny_batch(seed=1)
    with torch.no_grad():
        ctx = model.llm_adapter(
            source_hidden_states=qwen,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
        ctx[~t5_mask] = 0
        assert ctx.shape == (1, 16, 1024)
        sampled = model(x, t, ctx, padding_mask=padding_mask)  # no target_input_ids -> context used as-is
        trained = model(
            x,
            t,
            qwen,
            padding_mask=padding_mask,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
    torch.testing.assert_close(sampled, trained)


def test_gradient_checkpointing_forward_backward(tiny_dit: Anima):
    model = tiny_dit.train()
    batch = _tiny_batch(seed=2)

    model.zero_grad(set_to_none=True)
    ref_out = _training_forward(model, batch)
    ref_out.square().mean().backward()
    ref_grad = model.blocks[1].mlp.layer1.weight.grad.clone()

    model.enable_gradient_checkpointing()
    assert all(b.gradient_checkpointing for b in model.blocks)
    try:
        model.zero_grad(set_to_none=True)
        # trainers mark inputs as requiring grad so non-reentrant checkpointing has something to attach to
        x = batch[0].clone().requires_grad_(True)
        out = _training_forward(model, (x, *batch[1:]))
        assert out.shape == x.shape and torch.isfinite(out).all()
        out.square().mean().backward()
        ckpt_grad = model.blocks[1].mlp.layer1.weight.grad
        assert ckpt_grad is not None and x.grad is not None
        torch.testing.assert_close(out.detach(), ref_out.detach())
        torch.testing.assert_close(ckpt_grad, ref_grad, rtol=1e-4, atol=1e-5)
    finally:
        model.disable_gradient_checkpointing()
        model.zero_grad(set_to_none=True)
    assert not any(b.gradient_checkpointing for b in model.blocks)


def test_bf16_weights_and_autocast_paths(tiny_dit: Anima):
    model = tiny_dit.eval()
    x, t, qwen, qwen_mask, t5_ids, t5_mask, padding_mask = _tiny_batch(seed=3)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(
            x,
            t,
            qwen,
            padding_mask=padding_mask,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()

    bf16 = Anima(**TINY_CFG).to(torch.bfloat16)
    bf16.load_state_dict(model.state_dict())
    with torch.no_grad():
        out = bf16(
            x.bfloat16(),
            t.bfloat16(),
            qwen.bfloat16(),
            padding_mask=padding_mask.bfloat16(),
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


def test_fp16_use_fp32_stability_path(tiny_dit: Anima):
    """fp16 activations trigger ``use_fp32`` (fp32 residual stream + AdaLN); needs autocast like sd-scripts' fp16 mode."""
    half = Anima(**TINY_CFG).half().eval()
    half.load_state_dict(tiny_dit.state_dict())
    x, t, qwen, *_ = _tiny_batch(seed=4)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        out = half(x.half(), t.half(), qwen.half(), padding_mask=torch.zeros(1, 1, 8, 8, dtype=torch.half))
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


def test_state_dict_key_patterns(tiny_dit: Anima):
    keys = set(tiny_dit.state_dict().keys())
    for expected in [
        "x_embedder.proj.1.weight",
        "t_embedder.1.linear_1.weight",
        "t_embedder.1.linear_2.weight",
        "t_embedding_norm.weight",
        "blocks.0.self_attn.q_proj.weight",
        "blocks.0.self_attn.q_norm.weight",
        "blocks.0.self_attn.output_proj.weight",
        "blocks.0.cross_attn.k_proj.weight",
        "blocks.0.mlp.layer1.weight",
        "blocks.0.mlp.layer2.weight",
        "blocks.0.adaln_modulation_self_attn.1.weight",
        "blocks.0.adaln_modulation_self_attn.2.weight",
        "blocks.0.adaln_modulation_cross_attn.1.weight",
        "blocks.0.adaln_modulation_mlp.2.weight",
        "blocks.1.self_attn.v_proj.weight",
        "final_layer.linear.weight",
        "final_layer.adaln_modulation.1.weight",
        "llm_adapter.embed.weight",
        "llm_adapter.blocks.0.cross_attn.k_proj.weight",
        "llm_adapter.blocks.5.mlp.2.bias",
        "llm_adapter.out_proj.weight",
        "llm_adapter.norm.weight",
    ]:
        assert expected in keys, expected

    # nothing bias-bearing in the DiT trunk, exactly like the official checkpoint
    assert not any(k.endswith(".bias") for k in keys if not k.startswith("llm_adapter."))
    # RoPE tables are derived, not weights: they must not leak into checkpoints (official files lack them)
    assert not any(k.startswith("pos_embedder.") for k in keys)
    assert "llm_adapter.rotary_emb.inv_freq" not in keys
    # no block-swap residue on the module
    for attr in ("enable_block_swap", "offloader", "blocks_to_swap", "prepare_block_swap_before_forward"):
        assert not hasattr(tiny_dit, attr), attr

    sd = tiny_dit.state_dict()
    assert sd["x_embedder.proj.1.weight"].shape == (128, (16 + 1) * 2 * 2 * 1)
    assert sd["final_layer.linear.weight"].shape == (2 * 2 * 1 * 16, 128)
    assert sd["blocks.0.self_attn.q_norm.weight"].shape == (64,)  # head_dim
    assert sd["blocks.0.cross_attn.k_proj.weight"].shape == (128, 1024)
    assert sd["llm_adapter.embed.weight"].shape == (32128, 1024)


def test_infer_dit_config_from_tiny_state_dict(tiny_dit: Anima):
    sd = tiny_dit.state_dict()
    cfg = infer_dit_config(sd)
    assert cfg["model_channels"] == 128
    assert cfg["in_channels"] == 16
    assert cfg["out_channels"] == 16
    assert cfg["num_blocks"] == 2
    assert cfg["num_heads"] == 2  # from q_norm head_dim (64): 128 // 64
    assert cfg["use_llm_adapter"] is True
    assert cfg["crossattn_emb_channels"] == 1024
    assert cfg["use_adaln_lora"] is True and cfg["adaln_lora_dim"] == 256
    assert cfg["mlp_ratio"] == 4.0
    assert cfg["max_img_h"] == cfg["max_img_w"] == 1024  # from ANIMA_2B_CONFIG, not from the checkpoint
    assert get_dit_config is infer_dit_config

    # The inferred config must rebuild a model with an identical state_dict layout.
    rebuilt = Anima(**dict(cfg, max_img_h=64, max_img_w=64))
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected

    # ComfyUI / official prefixes are auto-detected and stripped.
    for prefix in ("net.", "model.diffusion_model."):
        prefixed = {prefix + k: v for k, v in sd.items()}
        assert detect_key_prefix(prefixed) == prefix
        assert infer_dit_config(prefixed) == cfg
        assert strip_key_prefix(prefixed).keys() == sd.keys()
    with pytest.raises(KeyError):
        detect_key_prefix({"foo.bar": torch.zeros(1)})


def test_infer_dit_config_without_adapter_and_heads_fallback_table():
    torch.manual_seed(0)
    no_adapter = Anima(**dict(TINY_CFG, use_llm_adapter=False, num_blocks=3))
    sd = no_adapter.state_dict()
    assert not any(k.startswith("llm_adapter.") for k in sd)
    cfg = infer_dit_config(sd)
    assert cfg["use_llm_adapter"] is False and cfg["num_blocks"] == 3 and cfg["num_heads"] == 2

    # Shape-only synthetic checkpoints without q_norm keys fall back to the width table, then model_channels // 128.
    class _Shape:
        def __init__(self, *shape):
            self.shape = torch.Size(shape)

    for width, heads in [(2048, 16), (5120, 40), (1280, 10), (128, 1)]:
        fake = {
            "x_embedder.proj.1.weight": _Shape(width, 68),
            "blocks.0.mlp.layer1.weight": _Shape(width * 4, width),
            "blocks.1.mlp.layer1.weight": _Shape(width * 4, width),
        }
        cfg = infer_dit_config(fake)
        assert cfg["num_heads"] == heads == ANIMA_NUM_HEADS_BY_WIDTH.get(width, width // 128)
        assert cfg["num_blocks"] == 2 and cfg["in_channels"] == 16 and cfg["model_channels"] == width
        assert cfg["use_llm_adapter"] is False


def test_official_2b_config_matches_inferred_from_meta_model():
    """Real 2B geometry on the meta device: 685 tensors, ~2.09B params, and infer_dit_config round-trips the constant."""
    with torch.device("meta"):
        big = Anima(**ANIMA_2B_CONFIG)
    sd = big.state_dict()
    assert len(sd) == 685
    assert sum(v.numel() for v in sd.values()) == 2_091_068_928
    assert sd["x_embedder.proj.1.weight"].shape == (2048, 68)
    assert sd["blocks.27.self_attn.q_norm.weight"].shape == (128,)
    assert "blocks.28.self_attn.q_proj.weight" not in sd
    assert infer_dit_config(sd) == ANIMA_2B_CONFIG
    assert ANIMA_2B_CONFIG["max_img_h"] == ANIMA_2B_CONFIG["max_img_w"] == 1024
    assert (
        ANIMA_2B_CONFIG["rope_h_extrapolation_ratio"] == ANIMA_2B_CONFIG["rope_w_extrapolation_ratio"] == 4.0
    )


def test_max_img_size_is_pure_extrapolation(tiny_dit: Anima):
    """Raising max_img_h/w only lengthens the RoPE position table: outputs are bit-identical for inputs that fit."""
    # seq = arange(max(len_h, len_w, len_t)); use max_frames=2 so the image size is what sizes the table
    small = Anima(**dict(TINY_CFG, max_frames=2)).eval()
    large = Anima(**dict(TINY_CFG, max_frames=2, max_img_h=256, max_img_w=256)).eval()
    for m in (small, large):
        missing, unexpected = m.load_state_dict(tiny_dit.state_dict(), strict=True)
        assert (
            not missing and not unexpected
        )  # non-persistent RoPE buffers -> no shape clash on pos_embedder.seq
    assert small.pos_embedder.max_h == 32 and large.pos_embedder.max_h == 128  # max_img // patch_spatial
    assert small.pos_embedder.seq.numel() == 32 and large.pos_embedder.seq.numel() == 128
    torch.testing.assert_close(large.pos_embedder.seq[:32], small.pos_embedder.seq)
    torch.testing.assert_close(large.pos_embedder.dim_spatial_range, small.pos_embedder.dim_spatial_range)
    assert large.pos_embedder.h_ntk_factor == small.pos_embedder.h_ntk_factor

    x, t, qwen, qwen_mask, t5_ids, t5_mask, padding_mask = _tiny_batch(seed=5, hw=16)
    with torch.no_grad():
        a = small(
            x,
            t,
            qwen,
            padding_mask=padding_mask,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
        b = large(
            x,
            t,
            qwen,
            padding_mask=padding_mask,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
    assert torch.equal(a, b)

    # ... and the small table really is the binding constraint: 64 latent px -> 32 patches fits, 80 does not.
    with torch.no_grad():
        small(
            torch.randn(1, 16, 1, 64, 64),
            t,
            qwen,
            padding_mask=torch.zeros(1, 1, 64, 64),
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )
        with pytest.raises(AssertionError, match="exceed the maximum"):
            small(
                torch.randn(1, 16, 1, 80, 80),
                t,
                qwen,
                target_input_ids=t5_ids,
                target_attention_mask=t5_mask,
                source_attention_mask=qwen_mask,
            )


def test_llm_adapter_standalone_shapes_and_masking():
    torch.manual_seed(0)
    adapter = LLMAdapter(
        source_dim=1024, target_dim=1024, model_dim=1024, num_layers=1, self_attn=True
    ).eval()
    qwen = torch.randn(2, 10, 1024)
    qwen_mask = torch.ones(2, 10, dtype=torch.long)
    qwen_mask[1, 7:] = 0
    ids = torch.randint(2, 32128, (2, 6))
    t5_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        out = adapter(qwen, ids, target_attention_mask=t5_mask, source_attention_mask=qwen_mask)
        # masked source tokens must not influence the output
        qwen2 = qwen.clone()
        qwen2[1, 7:] = 123.0
        out2 = adapter(qwen2, ids, target_attention_mask=t5_mask, source_attention_mask=qwen_mask)
    assert out.shape == (2, 6, 1024)
    torch.testing.assert_close(out, out2)


def test_dump_dit_key_list(tiny_dit: Anima):
    """Write docs/reference/anima-dit-keys.txt: tiny-model keys (as built here) + official 2B shapes (meta device)."""
    tiny_sd = tiny_dit.state_dict()
    with torch.device("meta"):
        big_sd = Anima(**ANIMA_2B_CONFIG).state_dict()

    def fmt(sd):
        w = max(len(k) for k in sd)
        return "\n".join(
            f"{k.ljust(w)}  {tuple(v.shape)}  {str(v.dtype).replace('torch.', '')}" for k, v in sd.items()
        )

    lines = [
        "# Anima DiT state_dict keys (ypuddin.models.anima.vendor.cosmos_dit.Anima)",
        "# Generated by tests/unit/test_anima_vendor.py::test_dump_dit_key_list -- do not edit by hand.",
        "#",
        "# Official checkpoints prefix every key with 'net.' (anima-base-v1.0, sd-scripts saves) or",
        "# 'model.diffusion_model.' (anima-aesthetics-v1.0 / ComfyUI); strip it before load_state_dict",
        "# (cosmos_dit.strip_key_prefix / infer_dit_config auto-detect both). RoPE tables (pos_embedder.*,",
        "# llm_adapter.rotary_emb.inv_freq) are non-persistent buffers and never appear in state_dict().",
        "# The DiT trunk has no biases; only the LLM adapter (llm_adapter.*) has bias tensors.",
        "",
        f"## Section 1: tiny test model ({len(tiny_sd)} tensors) -- config: "
        + ", ".join(
            f"{k}={TINY_CFG[k]}"
            for k in (
                "model_channels",
                "num_blocks",
                "num_heads",
                "in_channels",
                "out_channels",
                "crossattn_emb_channels",
                "adaln_lora_dim",
                "use_llm_adapter",
                "max_img_h",
                "max_img_w",
            )
        ),
        "",
        fmt(tiny_sd),
        "",
        f"## Section 2: official Anima 2B geometry ({len(big_sd)} tensors, {sum(v.numel() for v in big_sd.values()):,} params) -- ANIMA_2B_CONFIG (built on the meta device)",
        "",
        fmt(big_sd),
        "",
    ]
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text("\n".join(lines), encoding="utf-8")
    text = KEYS_FILE.read_text(encoding="utf-8")
    assert "blocks.0.self_attn.q_proj.weight" in text and "llm_adapter.embed.weight" in text
    assert "blocks.27.mlp.layer2.weight" in text


# --------------------------------------------------------------------------------------------------- attention


def test_attention_helper_split_and_masked_paths_match_sdpa():
    torch.manual_seed(0)
    q = torch.randn(2, 5, 3, 8)  # (B, L_q, H, D)
    k = torch.randn(2, 9, 3, 8)  # cross-attention: L_kv != L_q
    v = torch.randn(2, 9, 3, 8)
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        .transpose(1, 2)
        .reshape(2, 5, 24)
    )

    p = vendored_attention.AttentionParams.create_attention_params("torch", False)
    torch.testing.assert_close(
        vendored_attention.attention([q.clone(), k.clone(), v.clone()], attn_params=p), ref
    )
    torch.testing.assert_close(
        vendored_attention.attention(q, k, v), ref
    )  # default params == torch, no split
    p_split = vendored_attention.AttentionParams.create_attention_params("torch", True)
    torch.testing.assert_close(
        vendored_attention.attention([q.clone(), k.clone(), v.clone()], attn_params=p_split), ref
    )
    assert p.supports_fp32 and not p.requires_same_dtype

    # masked path (img tokens + padded text tokens): equal seqlens -> trimmed and zero-padded back
    img_len, txt_len = 4, 6
    x = torch.randn(2, img_len + txt_len, 3, 8)
    txt_mask = torch.ones(2, txt_len, dtype=torch.long)
    txt_mask[:, 4:] = 0
    pm = vendored_attention.AttentionParams.create_attention_params_from_mask(
        "torch", False, img_len, txt_mask
    )
    out = vendored_attention.attention([x.clone(), x.clone(), x.clone()], attn_params=pm)
    assert out.shape == (2, img_len + txt_len, 24)
    assert torch.all(out[:, img_len + 4 :] == 0)
    valid = img_len + 4
    ref = (
        torch.nn.functional.scaled_dot_product_attention(*(x[:, :valid].transpose(1, 2),) * 3)
        .transpose(1, 2)
        .reshape(2, valid, 24)
    )
    torch.testing.assert_close(out[:, :valid], ref)

    with pytest.raises(NotImplementedError):
        vendored_attention.attention(
            [q.clone(), k.clone(), v.clone()],
            attn_params=vendored_attention.AttentionParams.create_attention_params("unknown", False),
        )
    with pytest.raises(NotImplementedError):
        Anima(**dict(TINY_CFG, num_blocks=1, use_llm_adapter=False, attn_mode="flash"))(
            torch.randn(1, 16, 1, 8, 8), torch.rand(1), torch.randn(1, 4, 1024)
        )


# --------------------------------------------------------------------------------------------------------- VAE


def test_tiny_2d_vae_encode_decode_shapes():
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage2D(**TINY_VAE_KW).eval()
    assert vae.spatial_compression_ratio == 8 and vae.z_dim == 16
    img = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        z = vae.encode_pixels_to_latents(img)
        assert z.shape == (1, 16, 8, 8) and torch.isfinite(z).all()
        rec = vae.decode_to_pixels(z)
        assert rec.shape == (1, 3, 64, 64) and rec.min() >= -1.0 and rec.max() <= 1.0
        # 5-D single-frame inputs are accepted and keep their rank; T > 1 is rejected (image-only class)
        z5 = vae.encode_pixels_to_latents(img.unsqueeze(2))
        assert z5.shape == (1, 16, 1, 8, 8)
        torch.testing.assert_close(z5.squeeze(2), z)
        assert vae.decode_to_pixels(z5).shape == (1, 3, 1, 64, 64)
        with pytest.raises(ValueError):
            vae.encode_pixels_to_latents(torch.rand(1, 3, 2, 64, 64))

    # normalisation is the per-channel (mode - mean) / std of the official constants
    with torch.no_grad():
        mode = vae.encode(img, return_dict=False)[0].mode()
    mean = torch.tensor(vae.latents_mean).view(1, 16, 1, 1)
    std = torch.tensor(vae.latents_std).view(1, 16, 1, 1)
    torch.testing.assert_close(z, (mode - mean) / std)
    assert len(vae.latents_mean) == len(vae.latents_std) == 16
    assert vae.latents_mean[0] == pytest.approx(-0.7571) and vae.latents_std[0] == pytest.approx(2.8184)


def test_tiny_3d_vae_and_2d_conversion_are_equivalent_for_single_frames():
    torch.manual_seed(0)
    vae3 = AutoencoderKLQwenImage(**TINY_VAE_KW).eval()
    vae2 = AutoencoderKLQwenImage2D(**TINY_VAE_KW).eval()
    sd2 = convert_3d_state_dict_to_2d(vae3.state_dict())
    assert not any(".time_conv." in k for k in sd2)
    info = vae2.load_state_dict(sd2, strict=True)
    assert not info.missing_keys and not info.unexpected_keys

    img = torch.rand(2, 3, 64, 48) * 2 - 1
    with torch.no_grad():
        z3 = vae3.encode_pixels_to_latents(img)  # 4-D in -> 4-D out on the 3-D class as well
        z2 = vae2.encode_pixels_to_latents(img)
        assert z3.shape == z2.shape == (2, 16, 8, 6)
        torch.testing.assert_close(z2, z3, rtol=1e-5, atol=1e-5)
        r3 = vae3.decode_to_pixels(z3)
        r2 = vae2.decode_to_pixels(z2)
        assert r3.shape == r2.shape == (2, 3, 64, 48)
        torch.testing.assert_close(r2, r3, rtol=1e-5, atol=1e-5)
        # 5-D path of the 3-D class, and the cache-free variant, agree too
        z3_5d = vae3.encode_pixels_to_latents(img.unsqueeze(2))
        assert z3_5d.shape == (2, 16, 1, 8, 6)
        torch.testing.assert_close(z3_5d.squeeze(2), z3)
        nocache = AutoencoderKLQwenImage(**TINY_VAE_KW, disable_cache=True).eval()
        nocache.load_state_dict(vae3.state_dict())
        torch.testing.assert_close(nocache.encode_pixels_to_latents(img), z3)


def test_official_vae_key_layout_and_defaults():
    """Default constructor == official config; parameter names follow the diffusers layout the checkpoint uses."""
    with torch.device("meta"):
        vae = AutoencoderKLQwenImage()
    sd = vae.state_dict()
    assert vae.encoder.dim == 96 and vae.z_dim == 16 and vae.temperal_downsample == [False, True, True]
    for key in (
        "encoder.conv_in.weight",
        "encoder.down_blocks.0.conv1.weight",
        "encoder.mid_block.attentions.0.to_qkv.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
        "decoder.up_blocks.0.upsamplers.0.time_conv.weight",
        "decoder.up_blocks.3.resnets.2.conv2.weight",
        "decoder.norm_out.gamma",
        "decoder.conv_out.bias",
    ):
        assert key in sd, key
    assert sd["encoder.conv_in.weight"].shape == (96, 3, 3, 3, 3)
    assert sd["quant_conv.weight"].shape == (32, 32, 1, 1, 1)
    assert sd["decoder.norm_out.gamma"].shape == (96, 1, 1, 1)
    # the 2-D class consumes the same checkpoint after conversion
    with torch.device("meta"):
        sd2 = AutoencoderKLQwenImage2D().state_dict()
    conv = convert_3d_state_dict_to_2d(sd)
    assert conv.keys() == sd2.keys()
    assert all(conv[k].shape == sd2[k].shape for k in sd2)


def test_load_safetensors_helper(tmp_path):
    from safetensors.torch import save_file

    p = tmp_path / "w.safetensors"
    save_file({"a": torch.arange(4, dtype=torch.float32), "b": torch.ones(2, 2)}, str(p))
    sd = load_safetensors(str(p), device="cpu")
    assert set(sd) == {"a", "b"} and torch.equal(sd["a"], torch.arange(4, dtype=torch.float32))
    sd = load_safetensors(str(p), device=torch.device("cpu"), disable_mmap=True, dtype=torch.bfloat16)
    assert sd["b"].dtype == torch.bfloat16 and sd["a"].tolist() == [0.0, 1.0, 2.0, 3.0]


# ------------------------------------------------------------------------------------------ vendoring hygiene


def test_vendored_files_have_no_sd_scripts_dependency():
    vendor_dir = REPO_ROOT / "ypuddin" / "models" / "anima" / "vendor"
    files = sorted(vendor_dir.glob("*.py"))
    assert {f.name for f in files} >= {
        "__init__.py",
        "attention.py",
        "cosmos_dit.py",
        "qwen_image_vae.py",
        "qwen_image_vae_2d.py",
    }
    banned = re.compile(
        r"^\s*(from|import)\s+(library|torchvision|transformers|accelerate|diffusers)\b", re.M
    )
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not banned.search(text), f"{f.name} imports a non-vendored dependency"
        if f.name != "__init__.py":
            assert text.splitlines()[0].startswith(
                "# Vendored from kohya-ss/sd-scripts (Apache-2.0) at commit 4e62430"
            ), f.name
    notice = (vendor_dir / "NOTICE.md").read_text(encoding="utf-8")
    for name in (
        "cosmos_dit.py",
        "attention.py",
        "qwen_image_vae.py",
        "qwen_image_vae_2d.py",
        "qwen3_06b",
        "t5_old",
        "Apache",
    ):
        assert name in notice
    assets = REPO_ROOT / "ypuddin" / "models" / "anima" / "assets"
    for rel in (
        "qwen3_06b/config.json",
        "qwen3_06b/tokenizer.json",
        "qwen3_06b/tokenizer_config.json",
        "qwen3_06b/vocab.json",
        "qwen3_06b/merges.txt",
        "t5_old/config.json",
        "t5_old/spiece.model",
        "t5_old/tokenizer.json",
    ):
        assert (assets / rel).is_file(), rel


def test_no_unexpected_warnings_from_import():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import importlib

        import ypuddin.models.anima.vendor.cosmos_dit as m

        importlib.reload(m)


def test_sage_mode_falls_back_to_sdpa_off_cuda():
    """attn_mode='sage' must produce SDPA results wherever sageattention cannot run (CPU, masks, dropout)."""
    from ypuddin.models.anima.vendor.attention import AttentionParams, attention

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 6, 4, 8) for _ in range(3))
    ref = attention([q, k, v], attn_params=AttentionParams.create_attention_params("torch", False))
    out = attention([q, k, v], attn_params=AttentionParams.create_attention_params("sage", False))
    torch.testing.assert_close(out, ref)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    ref_m = attention(
        [q, k, v], attn_params=AttentionParams.create_attention_params_from_mask("torch", False, 0, mask)
    )
    out_m = attention(
        [q, k, v], attn_params=AttentionParams.create_attention_params_from_mask("sage", False, 0, mask)
    )
    torch.testing.assert_close(out_m, ref_m)
