"""Krea 2 family: vendored SingleStreamDiT geometry inference, checkpoint variants (bare / prefixed / fp8_scaled),
forward contract (patchify, image-first positions, text padding mask), presets on the real geometry, resolution-aware
preview shift, GQA expansion in the shared attention helper and the Qwen3-VL text pipeline on a reduced model."""

from __future__ import annotations

import shutil
import weakref
from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file

from ypuddin.adapters.frozen import FrozenLinear, quantize_fp8
from ypuddin.adapters.inject import inject
from ypuddin.adapters.rules import resolve_targets
from ypuddin.config import AdapterConfig, ObjectiveConfig, TrainConfig
from ypuddin.models import get_family
from ypuddin.models.anima.vendor import attention as vendored_attention
from ypuddin.models.base import LoadedModel, TextCond
from ypuddin.models.krea2.family import MU_RANGE, MU_TOKENS, Krea2Family, load_dit
from ypuddin.models.krea2.vendor.krea2_mmdit import (
    KREA2_CONFIG,
    SingleMMDiTConfig,
    SingleStreamDiT,
    infer_config,
)

TINY = SingleMMDiTConfig(
    features=64,
    tdim=16,
    txtdim=32,
    heads=4,
    kvheads=2,
    multiplier=4,
    layers=2,
    patch=2,
    channels=16,
    txtheads=2,
    txtkvheads=1,
    txtlayers=3,
)


def _tiny_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {k: v.contiguous() for k, v in SingleStreamDiT(TINY).state_dict().items()}


def _fp8_scaled(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Comfy-Org style: every block Linear weight in fp8 + ``.scale_weight``; norms / modulation / embeddings stay."""
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if (
            k.startswith("blocks.")
            and k.endswith(".weight")
            and v.ndim == 2
            and "norm" not in k
            and ".mod." not in k
        ):
            q, s = quantize_fp8(v, "fp8_e4m3")
            out[k] = q
            out[k[: -len(".weight")] + ".scale_weight"] = s
        else:
            out[k] = v
    out["scaled_fp8"] = torch.zeros((), dtype=torch.float8_e4m3fn)
    return out


def _cond(b: int = 2, n: int = 7, layers: int = 3, dim: int = 32, valid=(7, 4)) -> TextCond:
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, v in enumerate(valid):
        mask[i, :v] = True
    return TextCond({"embeds": torch.randn(b, n, layers, dim), "attn_mask": mask})


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("krea2_ckpt")
    sd = _tiny_state_dict()
    save_file(sd, str(root / "bare.safetensors"))
    save_file({"model.diffusion_model." + k: v for k, v in sd.items()}, str(root / "comfy.safetensors"))
    save_file(_fp8_scaled(sd), str(root / "fp8.safetensors"))
    return root


# ------------------------------------------------------------------------------------------------ geometry / loading
def test_infer_config_round_trips_tiny_and_official_geometry():
    assert infer_config(_tiny_state_dict()) == TINY
    with torch.device("meta"):
        official = SingleStreamDiT(KREA2_CONFIG)
    assert infer_config(dict(official.state_dict())) == KREA2_CONFIG
    n_params = sum(p.numel() for p in official.parameters())
    assert 12.5e9 < n_params < 13.0e9  # the public 12.9B checkpoint
    assert sum(isinstance(m, torch.nn.Linear) for m in official.modules()) == 264


def test_load_dit_accepts_bare_and_comfy_prefixed_keys(checkpoints):
    m_bare, cfg_bare = load_dit(checkpoints / "bare.safetensors", device="cpu", dtype=torch.float32)
    m_comfy, cfg_comfy = load_dit(checkpoints / "comfy.safetensors", device="cpu", dtype=torch.float32)
    assert cfg_bare == cfg_comfy == TINY.__dict__
    for (k1, v1), (k2, v2) in zip(m_bare.state_dict().items(), m_comfy.state_dict().items(), strict=True):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)
    assert not any(p.requires_grad for p in m_bare.parameters())
    assert not any(p.device.type == "meta" for p in m_bare.parameters())
    assert not any(b.device.type == "meta" for b in m_bare.buffers())


def test_fp8_scaled_checkpoint_loads_into_frozen_linears_and_matches_bf16_model(checkpoints):
    fam = get_family("krea2")
    ref, _ = load_dit(checkpoints / "bare.safetensors", device="cpu", dtype=torch.float32)
    fp8, _ = load_dit(checkpoints / "fp8.safetensors", device="cpu", dtype=torch.float32)
    frozen = {n: m for n, m in fp8.named_modules() if isinstance(m, FrozenLinear)}
    assert len(frozen) == 2 * 8 and all(m.is_fp8 for m in frozen.values())
    assert "blocks.0.attn.wq" in frozen and "blocks.1.mlp.down" in frozen
    assert not isinstance(fp8.get_submodule("first"), FrozenLinear)  # untouched: not quantized in the file
    # the scale of the checkpoint is used verbatim (not re-derived)
    torch.testing.assert_close(
        frozen["blocks.0.attn.wq"].weight_scale, quantize_fp8(ref.blocks[0].attn.wq.weight, "fp8_e4m3")[1]
    )
    x, t = torch.randn(2, 16, 8, 12), torch.rand(2)
    cond = _cond()
    out_ref = fam.forward(LoadedModel(ref, None, None, torch.device("cpu"), torch.float32), x, t, cond)
    out_fp8 = fam.forward(LoadedModel(fp8, None, None, torch.device("cpu"), torch.float32), x, t, cond)
    assert (out_fp8 - out_ref).norm() / out_ref.norm() < 2e-2


def test_missing_scale_for_fp8_weight_is_an_error(tmp_path):
    sd = _fp8_scaled(_tiny_state_dict())
    del sd["blocks.0.attn.wq.scale_weight"]
    save_file(sd, str(tmp_path / "broken.safetensors"))
    with pytest.raises(RuntimeError, match="scale"):
        load_dit(tmp_path / "broken.safetensors", device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("scale_shape", [(), (1,)])
def test_fp8_scales_release_safetensors_views_when_weights_move(tmp_path, monkeypatch, scale_shape):
    """A scalar's old view base must not retain the whole mapped checkpoint after a move."""
    import safetensors.torch

    from ypuddin.memory import BlockSwapper

    state = _fp8_scaled(_tiny_state_dict())
    for key in state:
        if key.endswith(".scale_weight"):
            state[key] = state[key].reshape(scale_shape)
    path = tmp_path / "mapped-fp8.safetensors"
    save_file(state, str(path))
    del state
    original_load = safetensors.torch.load_file
    source_scales = []

    def observe_load(*args, **kwargs):
        tensors = original_load(*args, **kwargs)
        for key, tensor in tensors.items():
            if key.endswith(".scale_weight"):
                # Observe the real mapped Tensor and its view chain without
                # keeping any of them alive ourselves.
                while tensor is not None:
                    source_scales.append(weakref.ref(tensor))
                    tensor = tensor._base
        return tensors

    monkeypatch.setattr(safetensors.torch, "load_file", observe_load)
    model, _ = load_dit(path, device="cpu", dtype=torch.float32)
    frozen = {name: layer for name, layer in model.named_modules() if isinstance(layer, FrozenLinear)}
    assert len(frozen) == 16 and source_scales
    expected = {
        name: (layer.weight.view(torch.uint8).clone(), layer.weight_scale.clone(), layer.dequant().clone())
        for name, layer in frozen.items()
    }
    identities = {name: (id(layer.weight), id(layer.weight_scale)) for name, layer in frozen.items()}
    adapters = inject(model, AdapterConfig(algo="lora", rank=4, alpha=4), Krea2Family().presets()["attn-mlp"])
    for name, layer in adapters.layers.items():
        if name in frozen:
            assert layer.base is frozen[name]
    swapper = BlockSwapper(model.blocks, num_swap=len(model.blocks), device="cpu")
    try:
        # Exercise the same in-place Tensor rebind used by a CPU -> CUDA move,
        # including the non-swapped modules, without requiring a GPU.
        for tensor in list(model.parameters()) + list(model.buffers()):
            tensor.data = tensor.detach().clone()
        assert all(reference() is None for reference in source_scales), (
            "FP8 scales still retain the source safetensors view after all model weights moved"
        )
        for name, layer in frozen.items():
            assert layer.weight_scale._base is None
            assert layer.precision == "fp8_e4m3" and layer.weight.dtype == torch.float8_e4m3fn
            assert layer.weight_scale.dtype == torch.float32 and layer.weight_scale.shape == ()
            assert (id(layer.weight), id(layer.weight_scale)) == identities[name]
            assert layer in tuple(model.modules())
            weight, scale, dequantized = expected[name]
            torch.testing.assert_close(layer.weight.view(torch.uint8), weight, rtol=0, atol=0)
            torch.testing.assert_close(layer.weight_scale, scale, rtol=0, atol=0)
            torch.testing.assert_close(layer.dequant(), dequantized, rtol=0, atol=0)
        for index in swapper.swapped_idx:
            swapper.ensure(index)
        swapper.release_all()
    finally:
        swapper.remove()


@pytest.mark.parametrize("filename", ["bare.safetensors", "fp8.safetensors"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_checkpoint_storage_budget_uses_header_and_real_loaded_dtypes(
    checkpoints, monkeypatch, filename, dtype
):
    import safetensors.torch

    from ypuddin.models.krea2.family import _checkpoint_storage_bytes

    path = checkpoints / filename
    model, _ = load_dit(path, device="cpu", dtype=dtype)
    actual = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))

    def forbidden(*args, **kwargs):
        pytest.fail("The materialization budget must not read the checkpoint payload")

    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    assert _checkpoint_storage_bytes(path, dtype) == actual


def test_storage_budget_matches_loader_scale_and_nonstandard_dtype_conversion(tmp_path):
    from ypuddin.models.krea2.family import _checkpoint_storage_bytes, _read_state_dict

    dtypes = (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        torch.float16,
        torch.float64,
        torch.complex64,
        torch.int64,
        torch.uint8,
        torch.bool,
    )
    state = {f"extra_{index}": torch.zeros(3, dtype=dtype) for index, dtype in enumerate(dtypes)}
    state["model.diffusion_model.layer.weight_scale"] = torch.tensor([0.125], dtype=torch.bfloat16)
    state["scaled_fp8"] = torch.zeros((), dtype=torch.float8_e4m3fn)
    path = tmp_path / "budget.safetensors"
    save_file(state, str(path))
    weights, scales = _read_state_dict(path, torch.bfloat16)
    actual = sum(tensor.numel() * tensor.element_size() for tensor in [*weights.values(), *scales.values()])
    assert _checkpoint_storage_bytes(path, torch.bfloat16) == actual


def test_load_dit_releases_temporary_state_dicts_before_device_move(checkpoints, monkeypatch):
    from ypuddin.models.krea2 import family as module

    destroyed = []

    class TrackedDict(dict):
        def __init__(self, label, contents):
            super().__init__(contents)
            self.label = label

        def __del__(self):
            destroyed.append(self.label)

    read, move = module._read_state_dict, SingleStreamDiT.to

    def read_tracked(*args, **kwargs):
        weights, scales = read(*args, **kwargs)
        return TrackedDict("weights", weights), TrackedDict("scales", scales)

    def move_observed(self, *args, **kwargs):
        assert sorted(destroyed) == ["scales", "weights"]
        return move(self, *args, **kwargs)

    monkeypatch.setattr(module, "_read_state_dict", read_tracked)
    monkeypatch.setattr(SingleStreamDiT, "to", move_observed)
    model, _ = load_dit(checkpoints / "fp8.safetensors", device="cpu", dtype=torch.float32)
    assert sum(isinstance(layer, FrozenLinear) for layer in model.modules()) == 16


def test_missing_tensor_is_an_error(tmp_path):
    sd = _tiny_state_dict()
    del sd["blocks.1.mlp.down.weight"]
    save_file(sd, str(tmp_path / "short.safetensors"))
    with pytest.raises(RuntimeError, match="missing"):
        load_dit(tmp_path / "short.safetensors", device="cpu", dtype=torch.float32)


# ------------------------------------------------------------------------------------------------ forward contract
def test_forward_shape_padding_invariance_and_patch_order(checkpoints):
    fam = get_family("krea2")
    dit, _ = load_dit(checkpoints / "bare.safetensors", device="cpu", dtype=torch.float32)
    loaded = LoadedModel(dit, None, None, torch.device("cpu"), torch.float32)
    torch.manual_seed(1)
    x, t = torch.randn(2, 16, 8, 12), torch.rand(2)
    cond = _cond()
    out = fam.forward(loaded, x, t, cond)
    assert out.shape == x.shape and torch.isfinite(out).all()
    # extra padded text tokens (masked out) must not change the prediction
    pad = TextCond(
        {
            "embeds": torch.cat([cond["embeds"], torch.randn(2, 5, 3, 32)], dim=1),
            "attn_mask": torch.cat([cond["attn_mask"], torch.zeros(2, 5, dtype=torch.bool)], dim=1),
        }
    )
    torch.testing.assert_close(fam.forward(loaded, x, t, pad), out, atol=1e-5, rtol=1e-4)
    # the second sample only has 4 valid tokens: replacing its padding values changes nothing either
    cond2 = TextCond({"embeds": cond["embeds"].clone(), "attn_mask": cond["attn_mask"]})
    cond2["embeds"][1, 4:] = 123.0
    torch.testing.assert_close(fam.forward(loaded, x, t, cond2), out, atol=1e-5, rtol=1e-4)
    # patchify / unpatchify are exact inverses in the family's layout
    b, c, h, w = x.shape
    hp, wp = h // 2, w // 2
    tokens = x.reshape(b, c, hp, 2, wp, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, hp * wp, c * 4)
    back = tokens.reshape(b, hp, wp, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
    torch.testing.assert_close(back, x)
    # and match the reference ``rearrange`` convention (c ph pw) of the original pipeline
    from einops import rearrange

    torch.testing.assert_close(tokens, rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2))
    with pytest.raises(ValueError, match="patch"):
        fam.forward(loaded, torch.randn(1, 16, 7, 8), torch.rand(1), _cond(1, 3, valid=(3,)))


def test_batch_independence_with_ragged_text(checkpoints):
    """Sample 1 must see exactly what it would see alone (image-first ordering + varlen mask)."""
    fam = get_family("krea2")
    dit, _ = load_dit(checkpoints / "bare.safetensors", device="cpu", dtype=torch.float32)
    loaded = LoadedModel(dit, None, None, torch.device("cpu"), torch.float32)
    torch.manual_seed(2)
    x, t = torch.randn(2, 16, 8, 8), torch.tensor([0.3, 0.8])
    cond = _cond(valid=(7, 3))
    both = fam.forward(loaded, x, t, cond)
    single = fam.forward(loaded, x[1:], t[1:], cond.select([1]))
    torch.testing.assert_close(both[1:], single, atol=1e-5, rtol=1e-4)


def test_lokr_training_step_with_block_checkpointing_on_fp8_base(checkpoints):
    fam = get_family("krea2")
    dit, _ = load_dit(checkpoints / "fp8.safetensors", device="cpu", dtype=torch.float32)
    dit.enable_gradient_checkpointing()
    aset = inject(
        dit,
        AdapterConfig(algo="lokr", rank=4, alpha=4, factor=4),
        fam.presets()["attn-mlp"],
        prefix="lora_unet",
    )
    assert len(aset.layers) == 16 and all(layer.base.is_fp8 for layer in aset.layers.values())
    dit.train()
    loaded = LoadedModel(dit, None, None, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 16, 8, 12, requires_grad=True)
    out = fam.forward(loaded, x, torch.rand(2), _cond())
    out.pow(2).mean().backward()
    assert all(
        any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.adapter.parameters())
        for layer in aset.layers.values()
    )
    assert not any(
        p.grad is not None for n, p in dit.named_parameters() if "lokr" not in n and "adapter" not in n
    )


# ------------------------------------------------------------------------------------------------ presets / planning
def test_presets_on_official_geometry_and_default():
    fam = Krea2Family()
    names = fam.linear_module_names()
    assert len(names) == 264
    counts = {
        n: len(resolve_targets(names, AdapterConfig(algo="lora", rank=32, alpha=32), p))
        for n, p in fam.presets().items()
    }
    assert counts == {"all-linear": 264, "attn-mlp": 224, "attn-only": 140, "attn-mlp-text": 259}
    assert fam.default_preset() == "attn-mlp"
    assert fam.spec.adapter_prefix == "lora_unet" and "online_text" not in fam.spec.capabilities
    with torch.device("meta"):
        backbone = fam.meta_backbone(type("Cfg", (), {"dit_path": None})())
    assert all(parameter.is_meta for parameter in backbone.parameters())
    layout = fam.memory_layout_meta(backbone)
    assert len(layout.blocks) == 28 and layout.block_param_bytes > 0
    assert "txtfusion*" in layout.keep_high_precision and "*mod*" in layout.keep_high_precision


def test_meta_backbone_reads_geometry_from_checkpoint_header(checkpoints):
    from ypuddin.config import ModelConfig

    fam = Krea2Family()
    cfg = ModelConfig(family="krea2", dit_path=str(checkpoints / "fp8.safetensors"))
    with torch.device("meta"):  # the planner builds on meta; only the header of the file is read
        meta = fam.meta_backbone(cfg)
    assert meta.config == TINY and next(meta.parameters()).device.type == "meta"
    with torch.device("meta"):
        assert (
            fam.meta_backbone(ModelConfig(family="krea2", dit_path="/nonexistent.safetensors")).config
            == KREA2_CONFIG
        )


def test_sampling_shift_follows_mu_interpolation():
    import math

    fam = Krea2Family()
    assert fam.spec.sampling.shift is None
    assert fam.sampling_shift(MU_TOKENS[0]) == pytest.approx(math.exp(MU_RANGE[0]))
    assert fam.sampling_shift(MU_TOKENS[1]) == pytest.approx(math.exp(MU_RANGE[1]))
    # 1024x1024 image = 64x64 latent = 32x32 = 1024 tokens
    mid = fam.sampling_shift(1024)
    assert math.exp(MU_RANGE[0]) < mid < math.exp(MU_RANGE[1])
    # a resolution_shift objective with custom endpoints drives the preview shift too
    obj = ObjectiveConfig(
        timestep_sampling="resolution_shift", res_shift_tokens=(256, 6400), res_shift_mu=(0.0, 0.0)
    )
    assert fam.sampling_shift(1024, obj) == pytest.approx(1.0)
    # families with a fixed shift keep it
    assert get_family("anima").sampling_shift(1024) == 3.0


def test_config_accepts_krea2_family_and_res_shift_fields():
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "krea2"},
            "dataset": {"sources": [{"path": "x"}]},
            "objective": {
                "timestep_sampling": "resolution_shift",
                "res_shift_tokens": [256, 6400],
                "res_shift_mu": [0.5, 1.15],
            },
        }
    )
    assert cfg.model.family == "krea2" and cfg.objective.res_shift_tokens == (256, 6400)
    schema = TrainConfig.model_json_schema()
    fam_schema = schema["$defs"]["ModelConfig"]["properties"]["family"]
    assert "krea2" in fam_schema["enum"]
    with pytest.raises(ValueError):
        TrainConfig.model_validate({"model": {"family": "flux3"}, "dataset": {"sources": [{"path": "x"}]}})


# ------------------------------------------------------------------------------------------------ GQA attention
def test_attention_helper_expands_grouped_kv_heads():
    torch.manual_seed(0)
    q = torch.randn(2, 6, 4, 8)  # 4 query heads
    k = torch.randn(2, 6, 2, 8)  # 2 kv heads (GQA)
    v = torch.randn(2, 6, 2, 8)
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.repeat_interleave(2, dim=2).transpose(1, 2),
            v.repeat_interleave(2, dim=2).transpose(1, 2),
        )
        .transpose(1, 2)
        .reshape(2, 6, 32)
    )
    p = vendored_attention.AttentionParams.create_attention_params("torch", False)
    torch.testing.assert_close(
        vendored_attention.attention([q.clone(), k.clone(), v.clone()], attn_params=p), ref
    )
    # masked (image-first + ragged padded text) path with GQA: valid rows equal attention over the valid prefix only
    mask = torch.ones(2, 3, dtype=torch.long)
    mask[1, 2:] = 0
    pm = vendored_attention.AttentionParams.create_attention_params_from_mask("torch", False, 3, mask)
    out = vendored_attention.attention([q.clone(), k.clone(), v.clone()], attn_params=pm)
    assert out.shape == (2, 6, 32)
    torch.testing.assert_close(out[0], ref[0])
    ref1 = (
        torch.nn.functional.scaled_dot_product_attention(
            q[1:, :5].transpose(1, 2),
            k[1:, :5].repeat_interleave(2, dim=2).transpose(1, 2),
            v[1:, :5].repeat_interleave(2, dim=2).transpose(1, 2),
        )
        .transpose(1, 2)
        .reshape(5, 32)
    )
    torch.testing.assert_close(out[1, :5], ref1)
    # equal seqlens -> the trimmed path, padding rows zeroed
    mask_eq = torch.ones(2, 3, dtype=torch.long)
    mask_eq[:, 2:] = 0
    pe = vendored_attention.AttentionParams.create_attention_params_from_mask("torch", False, 3, mask_eq)
    out_eq = vendored_attention.attention([q.clone(), k.clone(), v.clone()], attn_params=pe)
    assert torch.all(out_eq[:, 5:] == 0)
    torch.testing.assert_close(out_eq[1, :5], ref1)


# ------------------------------------------------------------------------------------------------ text pipeline
@pytest.fixture(scope="module")
def tiny_qwen3vl(tmp_path_factory):
    pytest.importorskip("transformers")
    from ypuddin.models.anima.text import ASSETS

    if not (ASSETS / "qwen3_06b").exists():
        pytest.skip("anima assets not vendored")
    from Test.tests.conftest import make_tiny_qwen3vl

    return make_tiny_qwen3vl(tmp_path_factory.mktemp("qwen3vl_tiny"))


def test_krea2_text_prompt_template_and_cache_entries(tiny_qwen3vl):
    from ypuddin.models.krea2.text import PREFIX, SUFFIX, Krea2Text

    tp = Krea2Text(tiny_qwen3vl["hf"], dtype=torch.float32, device="cpu", max_len=32, select_layers=(1, 2, 3))
    assert tp.prefix_len == 34 and tp.suffix_ids.tolist() == [151645, 198, 151644, 77091, 198]
    ids, mask = tp._tokenize(["a cat"])
    assert ids.shape == (1, 32 + 34) and mask.shape == ids.shape  # max_len + prefix - suffix + suffix
    decoded = tp.tokenizer.decode(ids[0][mask[0]])
    assert decoded.startswith(PREFIX) and decoded.endswith("a cat" + SUFFIX)
    assert not mask[0, tp.prefix_len + 2 : -len(tp.suffix_ids)].any()  # padding between caption and suffix
    entries = tp.encode_for_cache(["a cat", "a very long caption about a dog sitting on a mat in the sun"])
    assert entries[0]["embeds"].shape == (2 + 5, 3, 32) and entries[0]["embeds"].dtype == torch.bfloat16
    assert entries[1]["embeds"].shape[0] > entries[0]["embeds"].shape[0]
    cond = tp.cond_from_cache(entries, "cpu")
    assert cond["embeds"].shape == (2, entries[1]["embeds"].shape[0], 3, 32)
    assert cond["attn_mask"].sum(1).tolist() == [7, entries[1]["embeds"].shape[0]]
    # the trimmed entry is exactly the valid slice of the full padded encoding
    ids_full, mask_full = tp._tokenize(["a cat"])
    out = tp.encoder(input_ids=ids_full, attention_mask=mask_full, output_hidden_states=True)
    stack = torch.stack([out.hidden_states[i] for i in (1, 2, 3)], dim=2)[0, tp.prefix_len :]
    valid = stack[mask_full[0, tp.prefix_len :]]
    torch.testing.assert_close(entries[0]["embeds"].float(), valid.to(torch.bfloat16).float())
    # right padding keeps the prefix at the front even when a caption is truncated
    ids_long, mask_long = tp._tokenize(["word " * 200])
    assert mask_long.all() and ids_long[0, -5:].tolist() == tp.suffix_ids.tolist()
    assert ids_long[0, : tp.prefix_len].tolist() == tp.tokenizer(PREFIX)["input_ids"]


def test_krea2_text_single_file_matches_hf_directory(tiny_qwen3vl):
    from ypuddin.models.krea2.text import Krea2Text

    ref = Krea2Text(
        tiny_qwen3vl["hf"], dtype=torch.float32, device="cpu", max_len=16, select_layers=(1, 2, 3)
    )
    single = Krea2Text(
        tiny_qwen3vl["single"],
        tokenizer_path=tiny_qwen3vl["hf"],
        dtype=torch.float32,
        device="cpu",
        max_len=16,
        select_layers=(1, 2, 3),
    )
    a = ref.encode_for_cache(["hello world"])[0]["embeds"]
    b = single.encode_for_cache(["hello world"])[0]["embeds"]
    torch.testing.assert_close(a, b)
    assert not hasattr(single.encoder, "visual")  # only the decoder is instantiated
    assert all(not value.is_meta for value in (*single.encoder.parameters(), *single.encoder.buffers()))
    for name, parameter in single.encoder.named_parameters():
        torch.testing.assert_close(parameter, ref.encoder.get_parameter(name), rtol=0, atol=0)
    for name, buffer in single.encoder.named_buffers():
        torch.testing.assert_close(buffer, ref.encoder.get_buffer(name), rtol=0, atol=0)
    single.unload()
    assert single.encoder is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_krea2_text_single_file_preserves_eager_decoder_results(tiny_qwen3vl, dtype):
    from ypuddin.models.krea2.text import Krea2Text

    options = dict(
        tokenizer_path=tiny_qwen3vl["hf"], dtype=dtype, device="cpu", max_len=16, select_layers=(1, 2, 3)
    )
    expected = Krea2Text(tiny_qwen3vl["single"], **options)
    # The eager fixture has the exact stored weights and CPU-initialized RoPE.
    # Convert its buffers as the old single-file loader did; HF from_pretrained
    # keeps some buffers fp32, so bf16 HF equivalence is a different contract.
    expected.encoder = deepcopy(tiny_qwen3vl["model"].model.language_model).to(dtype)
    expected.encoder.config.use_cache = False
    expected.encoder.requires_grad_(False).eval()
    actual = Krea2Text(tiny_qwen3vl["single"], **options)
    reference = expected.encode_for_cache(["hello world", "a cat"])
    result = actual.encode_for_cache(["hello world", "a cat"])
    for a, b in zip(result, reference, strict=True):
        torch.testing.assert_close(a["embeds"], b["embeds"], rtol=0, atol=0)
    for name, buffer in actual.encoder.named_buffers():
        assert not buffer.is_meta
        torch.testing.assert_close(buffer, expected.encoder.get_buffer(name), rtol=0, atol=0)


def test_krea2_text_fp8_scaled_single_file_dequantizes(tiny_qwen3vl, tmp_path):
    from safetensors.torch import load_file

    from ypuddin.models.krea2.text import Krea2Text

    sd = load_file(str(tiny_qwen3vl["single"]))
    fp8: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("model.layers.") and k.endswith("proj.weight"):
            q, s = quantize_fp8(v, "fp8_e4m3")
            fp8[k] = q
            fp8[k[: -len(".weight")] + ".scale_weight"] = s
        else:
            fp8[k] = v
    fp8["scaled_fp8"] = torch.zeros((), dtype=torch.float8_e4m3fn)
    d = tmp_path / "fp8"
    d.mkdir()
    save_file(fp8, str(d / "te.safetensors"))
    shutil.copy(tiny_qwen3vl["hf"] / "config.json", d / "config.json")
    ref = Krea2Text(
        tiny_qwen3vl["hf"], dtype=torch.float32, device="cpu", max_len=16, select_layers=(1, 2, 3)
    )
    tp = Krea2Text(
        d / "te.safetensors",
        tokenizer_path=tiny_qwen3vl["hf"],
        dtype=torch.float32,
        device="cpu",
        max_len=16,
        select_layers=(1, 2, 3),
    )
    a = ref.encode_for_cache(["hello world"])[0]["embeds"].float()
    b = tp.encode_for_cache(["hello world"])[0]["embeds"].float()
    assert (a - b).norm() / a.norm() < 5e-2


def test_krea2_text_single_file_geometry_mismatch_is_explained(tiny_qwen3vl, tmp_path, monkeypatch):
    from transformers import Qwen3VLTextModel

    from ypuddin.models.krea2.text import Krea2Text

    original_init = Qwen3VLTextModel.__init__
    constructed = []

    def check_meta_init(self, config):
        # Fail before allocating any 4B tensors if the loader regresses to CPU init.
        assert torch.empty(0).is_meta
        original_init(self, config)
        constructed.append(sum(parameter.numel() for parameter in self.parameters()))
        assert all(parameter.is_meta for parameter in self.parameters())

    monkeypatch.setattr(Qwen3VLTextModel, "__init__", check_meta_init)
    lone = tmp_path / "lone.safetensors"
    shutil.copy(tiny_qwen3vl["single"], lone)  # no config.json next to it -> 4B geometry assumed
    tp = Krea2Text(lone, tokenizer_path=tiny_qwen3vl["hf"], dtype=torch.float32, device="cpu", max_len=16)
    with pytest.raises(RuntimeError, match="config.json"):
        tp.encode_for_cache(["x"])
    assert len(constructed) == 1 and constructed[0] > 3_000_000_000
    assert tp.encoder is None


@pytest.mark.parametrize("defect", ["missing", "unexpected"])
def test_krea2_text_single_file_rejects_incomplete_checkpoint(tiny_qwen3vl, tmp_path, defect):
    from safetensors.torch import load_file

    from ypuddin.models.krea2.text import Krea2Text

    state = load_file(str(tiny_qwen3vl["single"]))
    if defect == "missing":
        del state["model.embed_tokens.weight"]
    else:
        state["model.unexpected.weight"] = torch.ones(1)
    path = tmp_path / "incomplete.safetensors"
    save_file(state, str(path))
    shutil.copy(tiny_qwen3vl["hf"] / "config.json", tmp_path / "config.json")
    text = Krea2Text(path, tokenizer_path=tiny_qwen3vl["hf"], device="cpu", select_layers=(1, 2, 3))
    with pytest.raises(RuntimeError, match=f"{defect}=\\['"):
        text.encode_for_cache(["x"])
    assert text.encoder is None
