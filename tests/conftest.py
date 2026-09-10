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
