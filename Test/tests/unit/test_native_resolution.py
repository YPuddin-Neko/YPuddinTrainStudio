import math
import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageOps

from ypuddin.config import TrainConfig
from ypuddin.data import build_data
from ypuddin.data.index import probe_image
from ypuddin.data.native import NativeBatchSampler, collate_native, microbatch_indices, native_size
from ypuddin.models import get_family
from ypuddin.train import Trainer
from ypuddin.train.plan import plan
from ypuddin.train.state import Progress


def test_native_geometry_retains_scale_and_respects_budget():
    size = native_size(1001, 777, align=16, max_pixels=1024**2, max_side=4096)
    assert (size.width, size.height, size.downscaled) == (992, 768, False)
    rng = random.Random(21)
    for _ in range(500):
        width, height = rng.randint(32, 8192), rng.randint(32, 8192)
        try:
            result = native_size(width, height, align=16, max_pixels=1024**2, max_side=2048)
        except ValueError:
            assert min(width, height) / max(width, height) < 16 / 2048
            continue
        assert result.width % 16 == result.height % 16 == 0
        assert result.width <= width and result.height <= height
        assert result.width * result.height <= 1024**2
        assert max(result.width, result.height) <= 2048
        assert 0 <= width * result.scale - result.width < 16.000001
        assert 0 <= height * result.scale - result.height < 16.000001
    with pytest.raises(ValueError, match="exceeds native"):
        native_size(2048, 1024, align=16, max_pixels=1024**2, max_side=4096, overflow="error")
    with pytest.raises(ValueError, match="smaller"):
        native_size(15, 512, align=16, max_pixels=1024**2, max_side=4096)


def test_native_plan_recomputes_source_shapes_when_pixel_limit_changes(tmp_path):
    root = tmp_path / "pictures"
    root.mkdir()
    for index, size in enumerate(((1896, 2656), (3000, 2448))):
        Image.new("RGB", size).save(root / f"{index}.png")
    cfg = TrainConfig.model_validate({"model": {"family": "toy"}, "dataset": {
        "sources": [{"path": str(root)}], "resolution_mode": "native",
        "native_max_pixels": 1024**2, "native_max_side": 4096,
    }})
    limited = plan(cfg, device="cpu")
    assert limited["native"]["downscaled"] == 2
    assert {(bucket["w"], bucket["h"]) for bucket in limited["buckets"]} == {(864, 1200), (1120, 912)}
    cfg.dataset.native_max_pixels = 4096**2
    preserved = plan(cfg, device="cpu")
    assert preserved["native"]["downscaled"] == 0
    assert {(bucket["w"], bucket["h"]) for bucket in preserved["buckets"]} == {(1888, 2656), (2992, 2448)}
    assert preserved["native"]["max_pixels"] == 4096**2


def test_native_sampler_keeps_singleton_shapes_and_resumes_without_repetition():
    shapes = [(64 + 16 * i, 64) for i in range(11)]
    sampler = NativeBatchSampler(shapes, 4, seed=9)
    batches = sampler.plan()
    assert [len(batch) for batch in batches] == [4, 4, 3]
    assert sorted(i for batch in batches for i in batch) == list(range(11))
    sampler.set_epoch(2)
    sampler.set_position(1)
    resumed = NativeBatchSampler(shapes, 4, seed=123)
    resumed.load_state_dict(sampler.state_dict())
    assert list(resumed) == sampler.plan(2)[1:]
    assert sampler.plan(0) != sampler.plan(2)


def test_native_microbatches_limit_simultaneous_pixels_and_preserve_every_image():
    shapes = [(64, 64)] * 5 + [(96, 64)] + [(64, 96)]
    groups = microbatch_indices(shapes, 8192)
    assert sorted(i for group in groups for i in group) == list(range(7))
    assert all(len({shapes[i] for i in group}) == 1 for group in groups)
    assert all(sum(math.prod(shapes[i]) for i in group) <= 8192 for group in groups)
    with pytest.raises(ValueError, match="exceeds"):
        microbatch_indices([(128, 128)], 8192)


def test_native_data_and_plan_use_original_sizes_without_multires_fanout(image_dataset, tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(image_dataset), "repeats": 2, "resolutions": [32, 48]}],
                "resolution_mode": "native",
                "resolutions": [64, 1024],
                "native_max_pixels": 8192,
                "batch_size": 5,
                "masked_loss": True,
                "cache_latents": False,
                "num_workers": 0,
            },
        }
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    assert len(bundle.train) == 24
    assert {item.bucket.key for item in bundle.train.items} == {(96, 64), (64, 96), (80, 80), (128, 48)}
    output = plan(cfg, device="cpu")
    assert output["ok"] and output["native"]["logical_batches"] == 5
    assert output["steps_per_epoch"] == 5
    assert output["native"]["downscaled"] == 0
    samples = [bundle.train[i] for i in range(5)]
    batch = collate_native(samples, max_pixels=8192)
    assert len(batch["caption"]) == 5
    for part in batch["microbatches"]:
        assert part["pixels"].shape[-2:] == tuple(reversed(part["bucket"]))
        assert part["pixels"].shape[0] * math.prod(part["bucket"]) <= 8192
        if "mask" in part:
            assert part["mask"].shape[-2:] == part["pixels"].shape[-2:]
    cfg.dataset.native_overflow = "error"
    cfg.dataset.native_max_pixels = 1024
    failed = plan(cfg, device="cpu")
    assert not failed["ok"] and any(error["loc"] == "dataset.native_max_pixels" for error in failed["errors"])


def test_native_mask_alignment_matches_image_crop(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    Image.new("RGB", (101, 79), "white").save(source / "a.png")
    mask = Image.new("L", (101, 79), 0)
    mask.paste(255, (40, 0, 101, 79))
    mask.save(source / "a.mask.png")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(source)}],
                "resolution_mode": "native",
                "masked_loss": True,
                "cache_latents": False,
            },
        }
    )
    sample = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache").train[0]
    assert sample["pixels"].shape[-2:] == sample["mask"].shape == (64, 96)
    assert sample["mask"][:, :38].eq(0).all() and sample["mask"][:, 38:].eq(1).all()

    # A differently sized mask must be fixed explicitly, not independently
    # resized/cropped and silently shifted away from the intended image region.
    Image.new("L", (64, 64), 255).save(source / "a.mask.png")
    preview = plan(cfg, device="cpu")
    assert not preview["ok"]
    assert any(error["loc"] == "dataset.masked_loss" for error in preview["errors"])


def test_native_uses_exif_orientation_and_does_not_reuse_bucket_transform_cache(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    image = Image.new("RGB", (101, 79), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source / "a.jpg", exif=exif)
    assert probe_image(source / "a.jpg")[:2] == (79, 101)
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(source)}],
                "resolution_mode": "native",
                "cache_latents": False,
            },
        }
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    item = bundle.train.items[0]
    assert item.bucket.key == (64, 96)
    assert bundle.train[0]["pixels"].shape[-2:] == (96, 64)
    native_key = bundle.train.cache_key(item, False)
    item.native_scale = None
    assert native_key != bundle.train.cache_key(item, False)


def test_native_downscale_uses_one_geometry_for_alpha_sidecar_and_flipped_rgb(tmp_path):
    from ypuddin.data.images import load_mask, pil_to_tensor, to_native

    source = tmp_path / "source.png"
    alpha = Image.new("L", (301, 179), 0)
    alpha.paste(255, (85, 25, 225, 150))
    Image.merge("RGBA", (alpha, alpha, alpha, alpha)).save(source)
    alpha.save(tmp_path / "source.mask.png")
    size = native_size(301, 179, align=16, max_pixels=16384, max_side=4096)
    assert size.downscaled
    resized = alpha.resize((round(301 * size.scale), round(179 * size.scale)), Image.BILINEAR)
    left, top = (resized.width - size.width) // 2, (resized.height - size.height) // 2
    expected = ImageOps.mirror(resized.crop((left, top, left + size.width, top + size.height)))
    expected_tensor = (pil_to_tensor(expected.convert("RGB"))[0] + 1) / 2
    for path, channel in [(str(tmp_path / "source.mask.png"), None), (None, alpha)]:
        actual = load_mask(path, channel, size.width, size.height, native_scale=size.scale, flip=True)
        torch.testing.assert_close(actual, expected_tensor)
    transformed = to_native(
        alpha, size.width, size.height, scale=size.scale, flip=True, resample=Image.BILINEAR
    )
    assert transformed.tobytes() == expected.tobytes()


def test_native_gradient_weights_each_image_equally_across_sizes_and_tail(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "dataset": {"resolution_mode": "native", "batch_size": 4},
            "loop": {"grad_accum": 2},
            "checkpoint": {"output_dir": str(tmp_path / "native-gradient")},
        }
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.progress = Progress(total_steps=1)
    trainer.sampler = NativeBatchSampler([(32, 32)] * 7, 4)
    trainer.bundle = SimpleNamespace(train=SimpleNamespace(set_epoch=lambda epoch: None))
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer.swapper = None
    trainer.emit = lambda *args, **kwargs: None
    trainer._epoch_hooks = lambda epoch: None
    groups = [[1.0, 2.0, 3.0], [4.0], [5.0], [6.0, 7.0]]
    trainer.loader = [
        {"caption": [""] * 4, "microbatches": [{"caption": [""] * len(g), "target": g} for g in groups[:2]]},
        {"caption": [""] * 3, "microbatches": [{"caption": [""] * len(g), "target": g} for g in groups[2:]]},
    ]
    trainer.compute_loss = lambda batch: (
        (parameter - torch.tensor(batch["target"])).square().mean(),
        None,
        None,
    )
    observed = []

    def step(loss, elapsed):
        observed.append((parameter.grad.item(), loss))
        trainer.progress.step += 1

    trainer._optimizer_step = step
    trainer._run_native_epoch()
    assert observed[0][0] == pytest.approx(-8.0)
    assert observed[0][1] == pytest.approx(20.0)
    assert trainer.progress.samples_seen == 7
