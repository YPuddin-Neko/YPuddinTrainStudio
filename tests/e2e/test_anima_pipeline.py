"""The whole Anima path through the real Trainer on CPU with reduced-size components.

Everything is the production code path (AnimaFamily.load -> real transformers Qwen3 -> LLM adapter ->
DiT -> VAE -> caches -> LoKr -> optimizer -> validation -> sampling -> export); only the *sizes* are
tiny: a 2-layer / 64-wide Qwen3 saved as an HF directory, a 2-block DiT checkpoint with a 2-layer
adapter, and a 16-channel-base Qwen-Image VAE installed through the vendored loader hook.
"""

import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from ypuddin.config import TrainConfig
from ypuddin.models.anima import vendor
from ypuddin.models.anima.text import ASSETS
from ypuddin.models.anima.vendor.cosmos_dit import ANIMA_2B_CONFIG, Anima
from ypuddin.models.anima.vendor.qwen_image_vae_2d import AutoencoderKLQwenImage2D
from ypuddin.train import Trainer

pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not (ASSETS / "qwen3_06b").exists(), reason="anima assets not vendored")

QWEN_HIDDEN = 64
TINY_DIT = dict(
    ANIMA_2B_CONFIG,
    max_img_h=64,
    max_img_w=64,
    model_channels=128,
    num_blocks=2,
    num_heads=2,
    crossattn_emb_channels=QWEN_HIDDEN,
    llm_adapter_source_dim=QWEN_HIDDEN,
    llm_adapter_dim=QWEN_HIDDEN,
    llm_adapter_layers=2,
    llm_adapter_heads=2,
)


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    root = tmp_path_factory.mktemp("anima_tiny")
    torch.manual_seed(0)
    # text encoder: real Qwen3 architecture, tiny dims, full vocabulary so the real tokenizer ids fit
    qcfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=QWEN_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    qdir = root / "qwen3_tiny"
    Qwen3ForCausalLM(qcfg).save_pretrained(qdir)
    for f in (ASSETS / "qwen3_06b").iterdir():
        if f.name != "config.json":
            shutil.copy(f, qdir / f.name)
    # DiT: official key names / prefix, reduced geometry
    dit = Anima(**TINY_DIT)
    dit_path = root / "anima_tiny.safetensors"
    save_file({"net." + k: v.contiguous() for k, v in dit.state_dict().items()}, str(dit_path))
    vae_path = root / "vae_placeholder.safetensors"
    save_file({"placeholder": torch.zeros(1)}, str(vae_path))
    return {"qwen": qdir, "dit": dit_path, "vae": vae_path}


@pytest.fixture
def tiny_vae_loader(monkeypatch):
    def load_vae(vae_path, input_channels=3, device="cpu", **_):
        torch.manual_seed(1)
        return AutoencoderKLQwenImage2D(base_dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1).to(
            device
        )

    monkeypatch.setattr(vendor.qwen_image_vae_2d, "load_vae", load_vae)


def _cfg(tiny_models, image_dataset, out, **overrides) -> TrainConfig:
    base = {
        "model": {
            "family": "anima",
            "dit_path": str(tiny_models["dit"]),
            "text_encoder_path": str(tiny_models["qwen"]),
            "vae_path": str(tiny_models["vae"]),
            "dtype": "fp32",
        },
        "dataset": {
            "sources": [{"path": str(image_dataset)}],
            "resolutions": [64],
            "bucket_step": 16,
            "batch_size": 2,
            "num_workers": 0,
            "caption": {"trigger_word": "ypd", "shuffle": True, "caption_dropout": 0.1},
        },
        "adapter": {"algo": "lokr", "rank": "full", "alpha": 1.0, "factor": 4, "preset": "with-adapter"},
        "optimizer": {"type": "adamw", "lr": 1e-3},
        "loop": {"epochs": 1, "grad_accum": 1, "mixed_precision": "no", "seed": 5},
        "memory": {"activation_checkpointing": "block"},
        "checkpoint": {"output_dir": str(out), "name": "anima", "save_every_epochs": 1},
        "validation": {"enabled": True, "split_ratio": 0.25, "every_epochs": 1, "timesteps": [0.3, 0.7]},
        "sampling": {
            "enabled": True,
            "every_epochs": 1,
            "prompts": [
                {"prompt": "ypd, 1girl, smile", "negative": "", "width": 64, "height": 64, "steps": 2}
            ],
            "width": 64,
            "height": 64,
            "cfg": 3.0,
        },
    }
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return TrainConfig.model_validate(base)


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("text_mode", ["online", "cached"])
def test_anima_trainer_end_to_end_on_cpu(tiny_models, tiny_vae_loader, image_dataset, tmp_path, text_mode):
    out = tmp_path / text_mode
    cfg = _cfg(tiny_models, image_dataset, out, dataset={"text_encoding": text_mode})
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    events = _events(out / "events.jsonl")
    types = [e["type"] for e in events]
    assert types[-1] == "run.finished"
    prepared = next(e for e in events if e["type"] == "run.prepared")
    assert prepared["text_mode"] == text_mode and prepared["total_steps"] == trainer.progress.step > 0
    steps = [e for e in events if e["type"] == "step"]
    assert all(torch.isfinite(torch.tensor(s["loss"])) for s in steps)
    val = [e for e in events if e["type"] == "validation"]
    assert val and set(val[0]["per_t"]) == {"0.3", "0.7"}
    assert [e for e in events if e["type"] == "sample.saved"]
    # geometry really came from the checkpoint
    dit_cfg = trainer.loaded.extra["dit_config"]
    assert (
        dit_cfg["model_channels"] == 128
        and dit_cfg["llm_adapter_source_dim"] == QWEN_HIDDEN
        and dit_cfg["llm_adapter_layers"] == 2
    )
    # adapters cover DiT blocks *and* the LLM adapter
    layers = list(trainer.adapters.layers)
    assert any(n.startswith("blocks.0.self_attn.q_proj") for n in layers)
    assert any(n.startswith("llm_adapter.blocks.0.cross_attn") for n in layers)
    # exported file: kohya-style Anima keys, LoKr tensors, metadata
    final = out / "anima-final.safetensors"
    tensors = load_file(str(final))
    assert "lora_unet_blocks_0_self_attn_q_proj.lokr_w1" in tensors
    assert "lora_unet_blocks_0_self_attn_q_proj.lokr_w2" in tensors
    assert "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.lokr_w1" in tensors
    assert all(k.startswith("lora_unet_") for k in tensors)
    from safetensors import safe_open

    with safe_open(str(final), framework="pt") as f:
        meta = f.metadata()
    assert (
        meta["modelspec.architecture"].startswith("anima") and meta["ss_network_module"] == "ypuddin.adapters"
    )
    # samples decode to real 64x64 PNGs through the (tiny) VAE
    from PIL import Image

    pngs = sorted((out / "samples").glob("*.png"))
    assert pngs and Image.open(pngs[0]).size == (64, 64)
    if text_mode == "cached":
        assert list((out / "cache" / "text").rglob("*.safetensors"))
        assert trainer.loaded.text.encoder is None  # encoder was unloaded after caching


def test_anima_convert_roundtrip_comfyui(tiny_models, tiny_vae_loader, image_dataset, tmp_path):
    from ypuddin.adapters.convert import comfy_to_kohya, kohya_to_comfy
    from ypuddin.models import get_family

    out = tmp_path / "run"
    cfg = _cfg(
        tiny_models,
        image_dataset,
        out,
        adapter={
            "algo": "lora",
            "rank": 4,
            "alpha": 4,
            "preset": "attn-mlp",
            "rules": [{"match": "blocks.*.mlp.*", "algo": "lokr", "rank": "full", "factor": 4}],
        },
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    tensors = load_file(str(out / "anima-final.safetensors"))
    names = get_family("anima").linear_module_names()
    comfy = kohya_to_comfy(tensors, names)
    # LoRA modules become PEFT-style diffusion_model.* keys, alpha travelling with them
    assert "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight" in comfy
    assert "diffusion_model.blocks.0.self_attn.q_proj.alpha" in comfy
    # LoKr modules (ComfyUI reads kohya LoKr keys natively) stay untouched, including their alpha
    assert "lora_unet_blocks_0_mlp_layer1.lokr_w1" in comfy and "lora_unet_blocks_0_mlp_layer1.alpha" in comfy
    assert len(comfy) == len(tensors)
    back = comfy_to_kohya(comfy)
    assert set(back) == set(tensors)
    assert all(torch.equal(back[k], tensors[k]) for k in tensors)


def test_anima_merge_into_prefixed_base_checkpoint(tiny_models, tiny_vae_loader, image_dataset, tmp_path):
    """`ypuddin merge` must resolve kohya keys against the official ``net.``-prefixed base file."""
    from ypuddin.models import get_family
    from ypuddin.models.anima.family import load_dit
    from ypuddin.tools import merge_into_state_dict

    out = tmp_path / "run"
    cfg = _cfg(
        tiny_models,
        image_dataset,
        out,
        checkpoint={"save_dtype": "fp32"},  # compare merge against the in-memory adapters exactly
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    tensors = load_file(str(out / "anima-final.safetensors"))
    base = load_file(str(tiny_models["dit"]))
    assert all(k.startswith("net.") for k in base)
    for names in (None, get_family("anima").linear_module_names()):
        merged, unmatched = merge_into_state_dict(base, tensors, module_names=names)
        assert not unmatched, unmatched[:3]
        assert set(merged) == set(base)
        layer = trainer.adapters.layers["blocks.0.self_attn.q_proj"]
        expected = base["net.blocks.0.self_attn.q_proj.weight"].float() + layer.adapter.delta_weight().float()
        torch.testing.assert_close(
            merged["net.blocks.0.self_attn.q_proj.weight"].float(), expected, rtol=1e-5, atol=1e-6
        )
        changed = [k for k in base if not torch.equal(base[k], merged[k])]
        assert len(changed) == len(trainer.adapters.layers)
    merged_path = tmp_path / "merged.safetensors"
    save_file({k: v.contiguous() for k, v in merged.items()}, str(merged_path))
    dit, _ = load_dit(merged_path, device="cpu", dtype=torch.float32)  # still a valid Anima checkpoint
    assert sum(p.numel() for p in dit.parameters()) == sum(v.numel() for v in base.values())
