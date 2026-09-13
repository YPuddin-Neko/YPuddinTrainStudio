"""Conditioning describes the real CPU pixel transform, including latent cache hits."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from ypuddin.config import CaptionConfig, DatasetSourceConfig, TrainConfig
from ypuddin.data import build_data
from ypuddin.data.buckets import Bucket
from ypuddin.data.cache import LatentCache, build_latent_cache
from ypuddin.data.dataset import Item, TrainDataset, collate
from ypuddin.data.images import pil_to_tensor
from ypuddin.data.index import ImageRecord, content_hash
from ypuddin.data.native import collate_native
from ypuddin.models import get_family


def make_item(root, name, source_size, target_size, *, image_fit="crop", scale=None, no_upscale=False):
    width, height = source_size
    y, x = np.indices((height, width))
    image = Image.fromarray(np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8))
    path = root / f"{name}.png"
    image.save(path)
    record = ImageRecord(str(path), 0, content_hash(path), width, height, None, None, False)
    item = Item(
        record,
        Bucket(*target_size, 0),
        DatasetSourceConfig(path=str(root)),
        CaptionConfig(),
        False,
        1.0,
        native_scale=scale,
        image_fit=image_fit,
        no_upscale=no_upscale,
    )
    return item, image


def make_dataset(items, cache=None, *, flip=False):
    return TrainDataset(
        items,
        latent_spec=get_family("toy").spec.latent,
        latent_cache=cache,
        flip=flip,
        masked_loss=False,
        seed=7,
    )


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize(
    "source_size,target_size,image_fit,scale,no_upscale,resized_size,offset",
    [
        ((131, 81), (64, 48), "crop", None, False, (78, 48), (7, 0)),
        ((81, 131), (48, 64), "crop", None, False, (48, 78), (0, 7)),
        ((101, 79), (96, 64), "crop", 1.0, False, (101, 79), (2, 7)),
        ((301, 179), (144, 80), "crop", 0.5, False, (150, 90), (3, 5)),
        ((128, 32), (64, 64), "pad", None, False, (64, 16), (0, 24)),
        ((101, 79), (112, 80), "pad", 1.0, False, (101, 79), (5, 0)),
        ((37, 29), (64, 64), "pad", None, True, (37, 29), (13, 17)),
        ((301, 179), (160, 96), "pad", 0.5, False, (150, 90), (5, 3)),
    ],
    ids=[
        "wide-crop",
        "tall-crop",
        "native-crop",
        "native-downscale",
        "pad",
        "native-pad",
        "small-pad",
        "downscale-pad",
    ],
)
def test_geometry_matches_actual_pixels_and_cached_latents(
    tmp_path, flip, source_size, target_size, image_fit, scale, no_upscale, resized_size, offset
):
    item, image = make_item(
        tmp_path, "image", source_size, target_size, image_fit=image_fit, scale=scale, no_upscale=no_upscale
    )
    cache = LatentCache(tmp_path / "latents")
    dataset = make_dataset([item], cache, flip=flip)
    if flip:
        dataset.set_epoch(next(epoch for epoch in range(100) if _flips_at_epoch(dataset, epoch)))
    online = dataset[0]
    left, top = offset
    width, height = target_size
    expected_geometry = {
        "original_size": tuple(reversed(source_size)),
        "crop_top_left": (0, 0) if image_fit == "pad" else (top, left),
        "target_size": (height, width),
    }
    assert online["geometry"] == expected_geometry
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    if image_fit == "pad":
        rw, rh = resized_size
        expected = Image.fromarray(
            np.pad(
                np.asarray(resized),
                ((top, height - rh - top), (left, width - rw - left), (0, 0)),
                mode="edge",
            )
        )
        assert online["mask"].sum() == rw * rh
    else:
        expected = resized.crop((left, top, left + width, top + height))
    if flip:
        expected = ImageOps.mirror(expected)
    torch.testing.assert_close(online["pixels"], pil_to_tensor(expected))

    # This small CPU encoder exercises the real safetensors cache path, not a GPU/VAE claim.
    def encode(pixels):
        return torch.nn.functional.avg_pool2d(pixels, 8)

    written = build_latent_cache(
        [(dataset.cache_key(item, flip), {"pixels": online["pixels"]})],
        cache,
        encode,
        device="cpu",
        dtype=torch.float32,
    )
    assert written == 1
    cached = dataset[0]
    assert "latents" in cached and "pixels" not in cached
    assert cached["geometry"] == online["geometry"]
    torch.testing.assert_close(cached["latents"], encode(online["pixels"].unsqueeze(0))[0])
    if "mask" in online:
        torch.testing.assert_close(cached["mask"], online["mask"])
    cached_batch, online_batch = collate([cached]), collate([online])
    for key, expected_pair in expected_geometry.items():
        assert cached_batch["geometry"][key].dtype == torch.int64
        assert cached_batch["geometry"][key].tolist() == [list(expected_pair)]
        torch.testing.assert_close(cached_batch["geometry"][key], online_batch["geometry"][key])


def _flips_at_epoch(dataset, epoch):
    dataset.set_epoch(epoch)
    return dataset._rng(0).random() < 0.5


def test_geometry_uses_oriented_original_size_from_real_index(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (101, 79), "white").save(source / "image.jpg", exif=exif)
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(source)}],
                "resolution_mode": "native",
                "image_fit": "crop",
                "cache_latents": False,
            },
        }
    )
    sample = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache").train[0]
    assert sample["geometry"] == {
        "original_size": (101, 79),
        "crop_top_left": (2, 7),
        "target_size": (96, 64),
    }
    assert sample["pixels"].shape[-2:] == (96, 64)


def test_native_collation_keeps_each_images_geometry_when_grouping_by_canvas(tmp_path):
    items = [
        make_item(tmp_path, "wide", (101, 79), (96, 64), scale=1.0)[0],
        make_item(tmp_path, "tall", (79, 101), (64, 96), scale=1.0)[0],
        make_item(tmp_path, "wide2", (105, 83), (96, 64), scale=1.0)[0],
    ]
    dataset = make_dataset(items)
    samples = [dataset[i] for i in range(3)]
    batch = collate_native(samples, max_pixels=96 * 64 * 2)
    assert [part["index"].tolist() for part in batch["microbatches"]] == [[0, 2], [1]]
    wide, tall = [part["geometry"] for part in batch["microbatches"]]
    assert wide["original_size"].tolist() == [[79, 101], [83, 105]]
    assert wide["crop_top_left"].tolist() == [[7, 2], [9, 4]]
    assert wide["target_size"].tolist() == [[64, 96], [64, 96]]
    assert tall["original_size"].tolist() == [[101, 79]]
    assert tall["crop_top_left"].tolist() == [[2, 7]]
    assert tall["target_size"].tolist() == [[96, 64]]
    assert all(tensor.dtype == torch.int64 for part in (wide, tall) for tensor in part.values())
