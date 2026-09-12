"""Whole-image transforms are shared by planning, masks, caches and actual training."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from ypuddin.config import TrainConfig, config_hash
from ypuddin.data import build_data
from ypuddin.data.buckets import fit_pad
from ypuddin.data.cache import LatentCache
from ypuddin.data.dataset import cache_latents
from ypuddin.data.images import to_padded, valid_image_mask
from ypuddin.data.index import dataset_fingerprint
from ypuddin.data.native import native_size
from ypuddin.models import get_family
from ypuddin.objectives.flow import Objective
from ypuddin.train import Trainer
from ypuddin.train.plan import plan


def make_source(root, size=(128, 32), color="white"):
    root.mkdir(parents=True)
    image = Image.new("RGB", size, color)
    for box, value in [
        ((0, 0, 8, 8), "red"),
        ((size[0] - 8, 0, size[0], 8), "blue"),
        ((0, size[1] - 8, 8, size[1]), "green"),
        ((size[0] - 8, size[1] - 8, size[0], size[1]), "yellow"),
    ]:
        image.paste(value, box)
    image.save(root / "image.png")
    (root / "image.txt").write_text("whole image")
    return root


def config(source, out, **dataset):
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(source)}],
                "resolutions": [64],
                "bucket_step": 16,
                "aspect_ratio_limit": 1,
                "image_fit": "pad",
                "num_workers": 0,
                "cache_latents": False,
                "masked_loss": False,
                **dataset,
            },
            "loop": {"epochs": 4, "mixed_precision": "no", "seed": 7},
            "adapter": {"algo": "lora", "rank": 2, "alpha": 2},
            "checkpoint": {
                "output_dir": str(out),
                "name": "toy",
                "save_every_epochs": None,
                "save_state_every_steps": 2,
            },
            "validation": {"enabled": False},
            "sampling": {"enabled": False},
        }
    )


@pytest.mark.parametrize("flip", [False, True])
def test_padding_keeps_every_pixel_and_all_four_corners_at_native_scale(tmp_path, flip):
    source = make_source(tmp_path / "source", (101, 79))
    original = Image.open(source / "image.png")
    result = to_padded(original, 112, 80, max_scale=1, flip=flip)
    mask = valid_image_mask(101, 79, 112, 80, max_scale=1, flip=flip)
    expected = Image.new("RGB", (112, 80))
    expected.paste(original, (5, 0))
    if flip:
        expected = ImageOps.mirror(expected)
    valid = mask.bool().numpy()
    np.testing.assert_array_equal(np.asarray(result)[valid], np.asarray(expected)[valid])
    assert mask.sum() == 101 * 79
    assert torch.equal(mask, mask.round())
    if not flip:
        # Replicated edges provide context, but they are not valid image pixels.
        np.testing.assert_array_equal(np.asarray(result)[:, 0], np.asarray(result)[:, 5])
        assert mask[:, :5].eq(0).all()


def test_native_padding_accounts_for_alignment_in_budget_and_never_crops():
    exact = native_size(101, 79, align=16, max_pixels=112 * 80, max_side=128, image_fit="pad")
    assert (exact.width, exact.height, exact.scale) == (112, 80, 1)
    with pytest.raises(ValueError, match="including alignment padding"):
        native_size(101, 79, align=16, max_pixels=101 * 79, max_side=128, image_fit="pad", overflow="error")
    rng = random.Random(19)
    for _ in range(300):
        w, h = rng.randint(1, 8192), rng.randint(1, 8192)
        size = native_size(w, h, align=16, max_pixels=1024**2, max_side=2048, image_fit="pad")
        rw, rh, left, top, right, bottom = fit_pad(w, h, size.width, size.height, max_scale=size.scale)
        assert size.width % 16 == size.height % 16 == 0
        assert size.width * size.height <= 1024**2 and max(size.width, size.height) <= 2048
        assert 0 < size.scale <= 1 and rw <= w and rh <= h
        assert 0 <= left < right <= size.width and 0 <= top < bottom <= size.height
        assert size.width - rw < 16 and size.height - rh < 16


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("user_mask", [False, True])
def test_plan_online_cache_and_current_user_mask_use_same_geometry(tmp_path, native, user_mask):
    source = make_source(tmp_path / "data")
    mask = Image.new("L", (128, 32), 0)
    mask.paste(255, (64, 0, 128, 32))
    mask.save(source / "image.mask.png")
    cfg = config(
        source,
        tmp_path / "out",
        resolution_mode="native" if native else "bucket",
        native_max_pixels=2048,
        native_max_side=128,
        masked_loss=user_mask,
        cache_latents=True,
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    dataset = bundle.train
    item = dataset.items[0]
    online = dataset[0]
    assert "mask" in online  # independent of user_mask
    assert online["mask"].sum() > 0 and online["mask"].min() == 0
    expected_valid = valid_image_mask(128, 32, *item.bucket.key, max_scale=item.max_scale, flip=False)
    if user_mask:
        expected_user = to_padded(
            mask, *item.bucket.key, max_scale=item.max_scale, mask=True, resample=Image.BILINEAR
        )
        expected_valid *= torch.from_numpy(np.asarray(expected_user).copy()).float() / 255
    torch.testing.assert_close(online["mask"], expected_valid)
    calls = []

    def encode(pixels):
        calls.append(pixels.clone())
        return torch.nn.functional.avg_pool2d(pixels, 8)

    assert cache_latents(bundle, encode, device="cpu", dtype=torch.float32) == 1
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], online["pixels"])
    cached = dataset[0]
    assert "latents" in cached and "pixels" not in cached
    torch.testing.assert_close(cached["mask"], online["mask"])
    torch.testing.assert_close(cached["latents"], encode(online["pixels"].unsqueeze(0))[0])
    info = plan(cfg, device="cpu")["image_fit"]
    rectangle = info["items"][0]
    actual = ((online["pixels"] + 1) * 127.5).round().byte()
    left, top, right, bottom = (rectangle[k] for k in ("left", "top", "right", "bottom"))
    for x, y, expected in (
        (left + 1, top + 1, (255, 0, 0)),
        (right - 2, top + 1, (0, 0, 255)),
        (left + 1, bottom - 2, (0, 128, 0)),
        (right - 2, bottom - 2, (255, 255, 0)),
    ):
        assert torch.all((actual[:, y, x].int() - torch.tensor(expected)).abs() <= 20)
    actual_valid = valid_image_mask(128, 32, *item.bucket.key, max_scale=item.max_scale, flip=False)
    assert info["cropped_images"] == 0
    assert info["padding_pixels"] == item.bucket.area - actual_valid.sum().item()
    assert info["items"][0]["padding_pixels"] == info["padding_pixels"]
    if user_mask:
        mask.paste(255, (0, 0, 128, 32))
        mask.save(source / "image.mask.png")
        torch.testing.assert_close(dataset[0]["mask"], actual_valid)
        Image.new("L", (16, 16)).save(source / "image.mask.png")
        with pytest.raises(ValueError, match="do not match"):
            dataset[0]


def test_compute_loss_ignores_padded_latents_with_user_masks_disabled(tmp_path):
    source = make_source(tmp_path / "data")
    cfg = config(source, tmp_path / "out")
    trainer = Trainer(cfg, device="cpu")
    x0 = torch.zeros(1, 4, 8, 8)
    valid = torch.zeros(1, 64, 64)
    valid[:, 24:40] = 1
    latent_valid = torch.nn.functional.interpolate(valid[:, None], (8, 8), mode="area")
    prediction = torch.nn.Parameter((1 - latent_valid).expand_as(x0).clone() * 100)
    trainer.loaded = SimpleNamespace(dtype=torch.float32)
    trainer.family = SimpleNamespace(
        forward=lambda *_: prediction, spec=SimpleNamespace(latent=SimpleNamespace(patch=1))
    )
    trainer.objective = Objective(cfg.objective)
    trainer.objective.prepare = lambda *_args, **_kwargs: (x0, torch.zeros_like(x0), x0)
    trainer._latents = lambda _batch: x0
    trainer._text_cond = lambda _captions: None
    batch = {"caption": ["image"], "mask": valid, "weight": torch.ones(1)}
    loss, per_image, _ = trainer.compute_loss(batch, t_override=torch.tensor([0.5]))
    assert loss.item() == per_image.item() == 0
    loss.backward()
    assert prediction.grad.eq(0).all()
    with torch.no_grad():
        prediction.add_(latent_valid)
    loss, per_image, _ = trainer.compute_loss(batch, t_override=torch.tensor([0.5]))
    assert loss.item() == per_image.item() == 1
    loss.backward()
    assert prediction.grad[(1 - latent_valid).expand_as(prediction).bool()].eq(0).all()
    assert prediction.grad[latent_valid.expand_as(prediction).bool()].gt(0).all()


def test_legacy_crop_cache_and_fingerprint_are_byte_compatible(tmp_path):
    source = make_source(tmp_path / "data")
    cfg = config(source, tmp_path / "out", image_fit="crop", cache_latents=True)
    legacy = cfg.to_dict()
    legacy["dataset"].pop("image_fit")
    restored = TrainConfig.model_validate(legacy)
    assert restored.dataset.image_fit == "crop"
    bundle = build_data(restored, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    ds = restored.dataset
    expected = dataset_fingerprint(
        bundle.records,
        ds.sources,
        settings={
            "dataset": ds.model_dump(
                mode="json",
                exclude={
                    "sources",
                    "cache_dir",
                    "num_workers",
                    "image_fit",
                    "resolution_mode",
                    "native_max_pixels",
                    "native_max_side",
                    "native_overflow",
                },
            ),
            "validation": restored.validation.model_dump(mode="json", exclude={"sources"}),
            "validation_content": [],
        },
    )
    assert bundle.plan.fingerprint == expected
    item = bundle.train.items[0]
    assert bundle.train.cache_key(item, False) == LatentCache.key(
        item.record.content_hash, *item.bucket.key, get_family("toy").spec.latent.fingerprint, False
    )
    cfg.dataset.image_fit = "pad"
    padded = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    assert padded.plan.fingerprint != expected
    assert padded.train.cache_key(padded.train.items[0], False) != bundle.train.cache_key(item, False)


@pytest.mark.parametrize("native,cached", [(False, False), (False, True), (True, False), (True, True)])
def test_real_toy_training_validation_and_exact_resume_keep_validity(tmp_path, native, cached):
    source = make_source(tmp_path / "data")
    val = make_source(tmp_path / "validation", color="gray")
    cfg = config(
        source,
        tmp_path / "reference",
        resolution_mode="native" if native else "bucket",
        native_max_pixels=2048,
        native_max_side=128,
        cache_latents=cached,
    )
    cfg.validation.enabled = True
    cfg.validation.sources = [{"path": str(val)}]
    cfg.validation.split_ratio = 0
    cfg.validation.timesteps = [0.5]
    trainer = Trainer(cfg, device="cpu")
    compute = trainer.compute_loss
    calls = []

    def checked(batch, **kwargs):
        assert "mask" in batch and batch["mask"].min() == 0
        assert batch["mask"].sum() > 0
        calls.append("validation" if "t_override" in kwargs else "train")
        return compute(batch, **kwargs)

    trainer.compute_loss = checked
    assert trainer.run() == "finished"
    assert set(calls) == {"train", "validation"}
    assert not cfg.dataset.masked_loss
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed_cfg.checkpoint.resume = str(tmp_path / "reference" / "state-2")
    resumed_cfg.dataset.cache_dir = str(tmp_path / "reference" / "cache")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    assert resumed.progress.extra["loss_sum"] == trainer.progress.extra["loss_sum"]
    assert resumed.progress.extra["loss_count"] == trainer.progress.step
    assert resumed.progress.extra["loss_mean_scope"] == "run"
    expected, _ = trainer.adapters.export_state()
    actual, _ = resumed.adapters.export_state()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_legacy_state_resumes_after_config_gains_explicit_crop_default(tmp_path):
    source = make_source(tmp_path / "data")
    cfg = config(source, tmp_path / "reference", image_fit="crop")
    original = Trainer(cfg, device="cpu")
    assert original.run() == "finished"
    state_path = tmp_path / "reference/state-2/state.json"
    state = json.loads(state_path.read_text())
    old_config = cfg.to_dict()
    old_config["dataset"].pop("image_fit")
    state["config_hash"] = config_hash(old_config)
    for key in ("loss_sum", "loss_count", "loss_mean_scope"):
        state["progress"]["extra"].pop(key)
    state_path.write_text(json.dumps(state))
    old_config["checkpoint"].update(output_dir=str(tmp_path / "resumed"), resume=str(state_path.parent))
    restored = Trainer(TrainConfig.model_validate(old_config), device="cpu")
    assert restored.run() == "finished"
    assert restored.progress.extra["loss_count"] == 2
    assert restored.progress.extra["loss_mean_scope"] == "since_resume"
    expected, _ = original.adapters.export_state()
    actual, _ = restored.adapters.export_state()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_new_defaults_preserve_frames_and_legacy_family_changes_keep_crop(tmp_path):
    from fastapi.testclient import TestClient

    from ypuddin.server import create_app
    from ypuddin.server.family_config import change_config_family

    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    with TestClient(app) as client:
        for family in ("anima", "krea2"):
            defaults = client.get(f"/api/config/defaults?family={family}").json()
            assert defaults["dataset"]["image_fit"] == "pad"
            assert defaults["dataset"]["resolutions"] == [1024]
        project = client.post("/api/projects", json={"name": "Whole frame"}).json()
        saved = client.get(f"/api/projects/{project['id']}/config").json()
        assert saved["dataset"]["image_fit"] == "pad"
        saved["dataset"].pop("image_fit")
        changed = change_config_family(app.state.ctx, saved, "krea2")
        assert changed["dataset"]["image_fit"] == "crop"
        source = make_source(tmp_path / "api-data")
        cfg = config(source, tmp_path / "run").to_dict()
        result = client.post("/api/plan", json={"config": cfg})
        assert result.status_code == 200, result.text
        assert result.json()["image_fit"]["padded_images"] == 1
        assert result.json()["image_fit"]["items"][0]["cropped_pixels"] == 0
    app.state.ctx.db.close()
