import io
import json
from dataclasses import replace

import pytest
import torch
from PIL import Image, ImageOps

from ypuddin.config import DatasetSourceConfig, TrainConfig
from ypuddin.data import build_data
from ypuddin.data.image_metadata import alpha_channel, has_alpha
from ypuddin.data.images import load_alpha, load_rgb
from ypuddin.data.index import IndexDB, content_hash, probe_image, record_content_key, scan_sources
from ypuddin.models import get_family
from ypuddin.server.routes_dataset_masks import _load


@pytest.mark.parametrize(
    "mode,key,opaque",
    [
        ("RGB", (10, 20, 30), (40, 50, 60)),
        ("L", 10, 40),
        ("P", 0, 1),
    ],
)
def test_png_color_keys_agree_across_index_rgb_loading_inspection_and_mask_editor(
    tmp_path, mode, key, opaque
):
    image = Image.new(mode, (4, 2), opaque)
    if mode == "P":
        image.putpalette([10, 20, 30, 40, 50, 60] + [0] * 762)
    image.putpixel((0, 0), key)
    path = tmp_path / "keyed.png"
    image.save(path, transparency=key)
    assert probe_image(path) == (4, 2, True)
    rgb, alpha = load_rgb(str(path))
    assert rgb.getpixel((0, 0)) == (255, 255, 255)
    assert alpha.getpixel((0, 0)) == 0 and alpha.getpixel((1, 0)) == 255
    assert load_alpha(str(path)).tobytes() == alpha.tobytes()
    editor_mask, info = _load(path, tmp_path / "keyed.mask.png")
    assert info.source == "alpha" and info.coverage == 7 / 8
    assert editor_mask.tobytes() == alpha.tobytes()
    with Image.open(path) as opened:
        assert has_alpha(opened)
        assert sum(alpha_channel(opened).histogram()[:255]) == 1


@pytest.mark.parametrize("key", [12, 1234, 65535])
def test_16bit_grayscale_png_compares_color_key_before_rgba_conversion(tmp_path, key):
    image = Image.new("I;16", (2, 1), 60000)
    image.putpixel((0, 0), key)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", transparency=key)
    original = buffer.getvalue()
    with Image.open(io.BytesIO(original)) as opened:
        assert opened.mode in ("I", "I;16")
        assert opened.info["transparency"] == key
        assert alpha_channel(opened).tobytes() == bytes([0, 255])
        integer_image = opened.convert("I")
        assert integer_image.info["transparency"] == key
        assert alpha_channel(integer_image).tobytes() == bytes([0, 255])
    path = tmp_path / "gray16.png"
    path.write_bytes(original)
    assert probe_image(path) == (2, 1, True)
    assert load_alpha(str(path)).tobytes() == bytes([0, 255])
    rgb, alpha = load_rgb(str(path))
    assert rgb.getpixel((0, 0)) == (255, 255, 255)
    assert alpha.tobytes() == bytes([0, 255])
    mask, info = _load(path, tmp_path / "gray16.mask.png")
    assert mask.tobytes() == bytes([0, 255])
    assert info.source == "alpha" and info.coverage == 0.5
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_fully_opaque_alpha_remains_distinct_from_transparent_pixels(tmp_path, mode):
    path = tmp_path / "opaque.png"
    Image.new(mode, (3, 2), (10, 20, 30, 255) if mode == "RGBA" else (10, 255)).save(path)
    assert probe_image(path)[2]
    assert load_alpha(str(path)).getextrema() == (255, 255)
    with Image.open(path) as image:
        assert sum(alpha_channel(image).histogram()[:255]) == 0


def test_exif_orientation_preserves_color_key_and_pixel_positions(tmp_path):
    path = tmp_path / "oriented.png"
    image = Image.new("RGB", (4, 2), (30, 40, 50))
    image.putpixel((0, 0), (10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, transparency=(10, 20, 30), exif=exif)
    with Image.open(path) as image:
        expected = ImageOps.exif_transpose(image).convert("RGBA").getchannel("A")
    assert probe_image(path) == (2, 4, True)
    assert load_alpha(str(path)).tobytes() == expected.tobytes()
    rgb, alpha = load_rgb(str(path))
    assert rgb.size == alpha.size == (2, 4)
    assert alpha.tobytes() == expected.tobytes()


def test_old_index_alpha_flags_are_reprobed_without_changing_the_image(tmp_path):
    path = tmp_path / "keyed.png"
    Image.new("RGB", (4, 2), (10, 20, 30)).save(path, transparency=(10, 20, 30))
    before = path.read_bytes()
    stat = path.stat()
    db = IndexDB(tmp_path / "index.sqlite")
    db.store(
        str(path),
        stat.st_mtime,
        stat.st_size,
        content_hash(path),
        4,
        2,
        False,
        stat_signature=json.dumps(
            ("exif-size-v1", stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino, stat.st_dev)
        ),
    )
    db.commit()
    assert scan_sources([DatasetSourceConfig(path=str(tmp_path))], index_db=db)[0].has_alpha
    db.close()
    assert path.read_bytes() == before


def test_color_key_fix_invalidates_only_previously_incorrect_latents_and_resume_identity(tmp_path):
    Image.new("RGB", (64, 64), (10, 20, 30)).save(tmp_path / "keyed.png", transparency=(10, 20, 30))
    Image.new("RGBA", (64, 64), (10, 20, 30, 128)).save(tmp_path / "alpha.png")
    Image.new("I;16", (64, 64), 1234).save(tmp_path / "gray16-keyed.png", transparency=1234)
    cfg = TrainConfig.model_validate(
        {"model": {"family": "toy"}, "dataset": {"sources": [{"path": str(tmp_path)}], "resolutions": [64]}}
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    for index, item in enumerate(bundle.train.items):
        old_record = replace(item.record, color_key_transparency=False)
        old_item = replace(item, record=old_record)
        changed = item.record.path.endswith("keyed.png")
        assert (record_content_key(item.record) != record_content_key(old_record)) is changed
        assert (bundle.train.cache_key(item, False) != bundle.train.cache_key(old_item, False)) is changed
        bundle.latent_cache.put(bundle.train.cache_key(old_item, False), {"latents": torch.zeros(3, 64, 64)})
        sample = bundle.train[index]
        assert ("pixels" in sample) is changed
        assert ("latents" in sample) is not changed
