"""The whole Krea 2 path through the real Trainer on CPU with reduced-size components.

Production code path end to end (Krea2Family.load -> real transformers Qwen3-VL decoder -> 12-layer-style hidden
state stack -> SingleStreamDiT -> Qwen-Image VAE -> caches -> LoKr -> optimizer -> validation -> preview sampling
with the resolution-aware shift -> export -> merge into the fp8_scaled base); only the *sizes* are tiny.
"""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tests.conftest import make_tiny_qwen3vl
from ypuddin.adapters.frozen import quantize_fp8
from ypuddin.config import TrainConfig
from ypuddin.models.anima import vendor
from ypuddin.models.anima.text import ASSETS
from ypuddin.models.anima.vendor.qwen_image_vae_2d import AutoencoderKLQwenImage2D
from ypuddin.models.krea2.vendor.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from ypuddin.train import Trainer

pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not (ASSETS / "qwen3_06b").exists(), reason="anima assets not vendored")

TEXT_HIDDEN = 32
TINY_DIT = SingleMMDiTConfig(
    features=64,
    tdim=16,
    txtdim=TEXT_HIDDEN,
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


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("krea2_tiny")
    qwen = make_tiny_qwen3vl(root / "qwen3vl", hidden=TEXT_HIDDEN, layers=4)
    torch.manual_seed(0)
    sd = {k: v.contiguous() for k, v in SingleStreamDiT(TINY_DIT).state_dict().items()}
    bf16_path = root / "krea2_tiny.safetensors"
    save_file(sd, str(bf16_path))
    # Comfy-Org fp8_scaled layout: fp8 block Linears + scale_weight, everything else untouched, marker tensor
    fp8: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if (
            k.startswith("blocks.")
            and k.endswith(".weight")
            and v.ndim == 2
            and "norm" not in k
            and ".mod." not in k
        ):
            q, s = quantize_fp8(v, "fp8_e4m3")
            fp8[k] = q
            fp8[k[: -len(".weight")] + ".scale_weight"] = s
        else:
            fp8[k] = v
    fp8["scaled_fp8"] = torch.zeros((), dtype=torch.float8_e4m3fn)
    fp8_path = root / "krea2_tiny_fp8_scaled.safetensors"
    save_file(fp8, str(fp8_path))
    vae_path = root / "vae_placeholder.safetensors"
    save_file({"placeholder": torch.zeros(1)}, str(vae_path))
    return {
        "qwen_dir": qwen["hf"],
        "qwen_single": qwen["single"],
        "dit": bf16_path,
        "dit_fp8": fp8_path,
        "vae": vae_path,
    }


@pytest.fixture
def tiny_vae_loader(monkeypatch):
    def load_vae(vae_path, input_channels=3, device="cpu", **_):
        torch.manual_seed(1)
        return AutoencoderKLQwenImage2D(base_dim=16, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1).to(
            device
        )

    monkeypatch.setattr(vendor.qwen_image_vae_2d, "load_vae", load_vae)


def _cfg(tiny_models, image_dataset, out, *, dit="dit", text="qwen_dir", **overrides) -> TrainConfig:
    base = {
        "model": {
            "family": "krea2",
            "dit_path": str(tiny_models[dit]),
            "text_encoder_path": str(tiny_models[text]),
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
        "adapter": {"algo": "lokr", "rank": "full", "alpha": 1.0, "factor": 4, "preset": "attn-mlp"},
        "objective": {
            "timestep_sampling": "resolution_shift",
            "res_shift_tokens": [256, 6400],
            "res_shift_mu": [0.5, 1.15],
        },
        "optimizer": {"type": "adamw", "lr": 1e-3},
        "loop": {"epochs": 1, "grad_accum": 1, "mixed_precision": "no", "seed": 5},
        "memory": {"activation_checkpointing": "block"},
        "checkpoint": {"output_dir": str(out), "name": "krea2", "save_every_epochs": 1},
        "validation": {"enabled": True, "split_ratio": 0.25, "every_epochs": 1, "timesteps": [0.3, 0.7]},
        "sampling": {
            "enabled": True,
            "every_epochs": 1,
            "prompts": [
                {"prompt": "ypd, 1girl, smile", "negative": "", "width": 64, "height": 64, "steps": 2}
            ],
            "width": 64,
            "height": 64,
        },
    }
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return TrainConfig.model_validate(base)


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("resolution_mode", ["bucket", "native"])
def test_krea2_trainer_end_to_end_on_cpu(
    tiny_models, tiny_vae_loader, image_dataset, tmp_path, resolution_mode
):
    out = tmp_path / "run"
    cfg = _cfg(
        tiny_models,
        image_dataset,
        out,
        dataset={
            "resolution_mode": resolution_mode,
            "native_max_pixels": 8192,
            "cache_latents": resolution_mode != "native",
            "masked_loss": resolution_mode == "native",
        },
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    events = _events(out / "events.jsonl")
    types = [e["type"] for e in events]
    assert types[-1] == "run.finished"
    prepared = next(e for e in events if e["type"] == "run.prepared")
    assert prepared["text_mode"] == "cached"  # no online_text capability -> auto resolves to cached
    assert prepared["total_steps"] == trainer.progress.step > 0
    steps = [e for e in events if e["type"] == "step"]
    assert all(torch.isfinite(torch.tensor(s["loss"])) for s in steps)
    val = [e for e in events if e["type"] == "validation"]
    assert val and set(val[0]["per_t"]) == {"0.3", "0.7"}
    assert [e for e in events if e["type"] == "sample.saved"]
    # geometry came from the checkpoint; the text pipeline picked 3 of the 4 decoder layers to match txtlayers
    assert (
        trainer.loaded.extra["dit_config"]["txtlayers"] == 3
        and trainer.loaded.extra["dit_config"]["features"] == 64
    )
    assert (
        trainer.loaded.text.select_layers == (2, 3, 4) and trainer.loaded.text.encoder is None
    )  # unloaded after caching
    cache_files = list((out / "cache" / "text").rglob("*.safetensors"))
    assert cache_files
    with safe_open(str(cache_files[0]), framework="pt") as f:
        embeds = f.get_tensor("embeds") if "embeds" in f.keys() else f.get_tensor(next(iter(f.keys())))
    assert embeds.dtype == torch.bfloat16 and embeds.shape[-2:] == (3, TEXT_HIDDEN)
    # adapters: main-block attention + SwiGLU only (attn-mlp), 16 layers for 2 blocks
    layers = list(trainer.adapters.layers)
    assert len(layers) == 16 and all(n.startswith("blocks.") for n in layers)
    assert (
        "blocks.0.attn.wq" in layers
        and "blocks.1.mlp.down" in layers
        and not any("txtfusion" in n for n in layers)
    )
    # exported file: musubi/ComfyUI-compatible kohya keys, LoKr tensors, metadata
    final = out / "krea2-final.safetensors"
    tensors = load_file(str(final))
    assert "lora_unet_blocks_0_attn_wq.lokr_w1" in tensors and "lora_unet_blocks_0_attn_wq.lokr_w2" in tensors
    assert all(k.startswith("lora_unet_") for k in tensors)
    with safe_open(str(final), framework="pt") as f:
        meta = f.metadata()
    assert (
        meta["modelspec.architecture"].startswith("krea2") and meta["ss_network_module"] == "ypuddin.adapters"
    )
    from PIL import Image

    pngs = sorted((out / "samples").glob("*.png"))
    assert pngs and Image.open(pngs[0]).size == (64, 64)


def test_krea2_fp8_scaled_base_and_single_file_text_encoder(
    tiny_models, tiny_vae_loader, image_dataset, tmp_path
):
    """Comfy-Org style downloads: fp8_scaled DiT + single-file Qwen3-VL. The base stays fp8 with checkpoint scales,
    training touches only the adapters, and ``merge`` writes scales back under the ComfyUI key name."""
    from ypuddin.adapters.frozen import FrozenLinear
    from ypuddin.models.krea2.family import load_dit
    from ypuddin.tools import merge_into_state_dict

    out = tmp_path / "run"
    cfg = _cfg(
        tiny_models,
        image_dataset,
        out,
        dit="dit_fp8",
        text="qwen_single",
        checkpoint={"save_dtype": "fp32"},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    assert all(
        isinstance(layer.base, FrozenLinear) and layer.base.is_fp8
        for layer in trainer.adapters.layers.values()
    )
    tensors = load_file(str(out / "krea2-final.safetensors"))
    base = load_file(str(tiny_models["dit_fp8"]))
    merged, unmatched = merge_into_state_dict(
        base, tensors, module_names=trainer.family.linear_module_names()
    )
    assert not unmatched
    assert set(merged) == set(base)  # scale_weight keys reused, no stray weight_scale
    assert merged["blocks.0.attn.wq.weight"].dtype == torch.float8_e4m3fn
    layer = trainer.adapters.layers["blocks.0.attn.wq"]
    expected = layer.base.dequant(torch.float32) + layer.adapter.delta_weight().float()
    got = merged["blocks.0.attn.wq.weight"].float() * merged["blocks.0.attn.wq.scale_weight"].float()
    assert (got - expected).norm() / expected.norm() < 2e-2  # re-quantized in fp8
    merged_path = tmp_path / "merged_fp8.safetensors"
    save_file({k: v.contiguous() for k, v in merged.items()}, str(merged_path))
    dit, _ = load_dit(
        merged_path, device="cpu", dtype=torch.float32
    )  # still a valid fp8_scaled Krea 2 checkpoint
    assert sum(isinstance(m, FrozenLinear) for m in dit.modules()) == 16


def test_krea2_rejects_online_text_and_mismatched_text_width(
    tiny_models, tiny_vae_loader, image_dataset, tmp_path
):
    cfg = _cfg(
        tiny_models,
        image_dataset,
        tmp_path / "a",
        dataset={"text_encoding": "online"},
        sampling={"enabled": False},
    )
    with pytest.raises(ValueError, match="online"):
        Trainer(cfg, device="cpu").run()
    wrong = make_tiny_qwen3vl(tmp_path / "wrong", hidden=48, layers=4)
    cfg = _cfg(tiny_models, image_dataset, tmp_path / "b", sampling={"enabled": False})
    cfg.model.text_encoder_path = str(wrong["hf"])
    with pytest.raises(ValueError, match="hidden size 48"):
        Trainer(cfg, device="cpu").run()
