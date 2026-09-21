import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_image_dataset(
    root: Path,
    n: int = 12,
    *,
    seed: int = 0,
    sizes=((96, 64), (64, 96), (80, 80), (128, 48)),
    captions=True,
    mask_every: int = 0,
) -> Path:
    """Synthetic RGB images (some RGBA) with tag captions; deterministic."""
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tags = [
        "1girl",
        "solo",
        "smile",
        "red_hair",
        "blue_eyes",
        "outdoors",
        "night",
        "cat_ears",
        "school_uniform",
        "sword",
    ]
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = (i * 37) % 255
        arr[..., 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[..., 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        if i % 5 == 4:
            a = np.full((h, w, 1), 255, dtype=np.uint8)
            a[: h // 2] = 0
            Image.fromarray(np.concatenate([arr, a], -1), "RGBA").save(root / f"img_{i:03d}.png")
        else:
            Image.fromarray(arr).save(root / f"img_{i:03d}.jpg", quality=90)
        if captions:
            chosen = rng.sample(tags, k=4)
            (root / f"img_{i:03d}.txt").write_text(", ".join(chosen), encoding="utf-8")
        if mask_every and i % mask_every == 0:
            m = np.zeros((h, w), dtype=np.uint8)
            m[:, : w // 2] = 255
            Image.fromarray(m).save(root / f"img_{i:03d}.mask.png")
    return root


@pytest.fixture
def image_dataset(tmp_path):
    return make_image_dataset(tmp_path / "data")


def make_tiny_qwen3vl(root: Path, *, hidden: int = 32, layers: int = 4, seed: int = 0):
    """A reduced Qwen3-VL (real architecture, full vocabulary) saved as an HF directory with the vendored Qwen3
    tokenizer, plus a ComfyUI-style single ``safetensors`` file (bare ``model.`` decoder keys, ``visual.`` tower)
    with the model's ``config.json`` next to it. Returns ``{"hf", "single", "model"}``."""
    import shutil

    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from ypuddin.models.anima.text import ASSETS

    torch.manual_seed(seed)
    cfg = Qwen3VLConfig(
        text_config=dict(
            vocab_size=151936,
            hidden_size=hidden,
            intermediate_size=2 * hidden,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=4096,
            tie_word_embeddings=True,
            rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default"},
            rope_theta=5_000_000,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=hidden,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
    )
    model = Qwen3VLForConditionalGeneration(cfg)
    hf_dir = root / "hf"
    model.save_pretrained(hf_dir)
    for f in (ASSETS / "qwen3_06b").iterdir():
        if f.name != "config.json":
            shutil.copy(f, hf_dir / f.name)
    single: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.startswith("model.language_model."):
            single["model." + k[len("model.language_model.") :]] = v.contiguous()
        elif k.startswith("model.visual."):
            single["visual." + k[len("model.visual.") :]] = v.contiguous()
        else:
            single[k] = v.contiguous()
    comfy_dir = root / "comfy"
    comfy_dir.mkdir()
    save_file(single, str(comfy_dir / "qwen3vl_tiny.safetensors"))
    shutil.copy(hf_dir / "config.json", comfy_dir / "config.json")
    return {"hf": hf_dir, "single": comfy_dir / "qwen3vl_tiny.safetensors", "model": model}
