"""Bucket coverage and image-retention contracts, independent of learned quality."""

import math
from itertools import product

import pytest

from ypuddin.data.buckets import BucketManager, fit_crop, fit_pad


@pytest.mark.parametrize(
    "base,step,ratio,tolerance",
    [
        (512, 64, 2, 0.1),
        (1024, 64, 2, 0.1),
        (1536, 64, 2, 0.1),
        (2048, 128, 2, 0.1),
        (768, 32, 3, 0.2),
        (1024, 64, 4, 0.5),
        (128, 8, 1, 0.1),
        (256, 16, 1.5, 0),
        (1000, 64, 2, 0.1),
    ],
)
def test_grid_equals_exhaustive_search_including_transposes(base, step, ratio, tolerance):
    manager = BucketManager([base], align=8, step=step, aspect_ratio_limit=ratio, area_tolerance=tolerance)
    limit = math.ceil(base * math.sqrt((1 + tolerance) * ratio) / step) * step
    expected = {
        (w, h)
        for w, h in product(range(step, limit + 1, step), repeat=2)
        if base**2 * (1 - tolerance) <= w * h <= base**2 * (1 + tolerance)
        and max(w / h, h / w) <= ratio + 1e-9
    }
    assert expected
    actual = {b.key for b in manager.buckets[base]}
    assert actual == expected
    assert actual == {(h, w) for w, h in actual}


def test_default_grid_restores_all_four_missing_shapes():
    buckets = BucketManager([1024]).buckets[1024]
    assert len(buckets) == 37
    assert {(704, 1344), (768, 1472), (832, 1152), (896, 1280)} <= {b.key for b in buckets}


@pytest.mark.parametrize("image_fit", ["crop", "pad"])
def test_portrait_avoids_previous_four_percent_crop_or_padding(image_fit):
    manager = BucketManager([1024])
    b = manager.assign(522, 1000, 1024, image_fit=image_fit)
    assert b.key == (768, 1472)
    geometry = fit_crop if image_fit == "crop" else fit_pad
    rw, rh, *_ = geometry(522, 1000, *b.key)
    assert (rw, rh) == ((768, 1472) if image_fit == "crop" else (768, 1471))
    old_rw, old_rh, *_ = geometry(522, 1000, 704, 1408)
    assert abs(old_rw * old_rh - 704 * 1408) / max(old_rw * old_rh, 704 * 1408) > 0.04


@pytest.mark.parametrize("image_fit", ["crop", "pad"])
def test_selected_shape_minimizes_actual_pixel_loss_across_all_legal_buckets(image_fit):
    manager = BucketManager([1024])
    for width, height in [(522, 1000), (1031, 1000), (1001, 777), (187, 2048), (2013, 1027), (4096, 4096)]:
        b = manager.assign(width, height, 1024, image_fit=image_fit)
        # The oracle checks the achieved pixel geometry, including integer rounding.
        transform = fit_crop if image_fit == "crop" else fit_pad

        def loss(bucket, width=width, height=height, transform=transform):
            rw, rh, *_ = transform(width, height, *bucket.key)
            return abs(rw * rh - bucket.area) / max(rw * rh, bucket.area)

        assert loss(b) == min(loss(candidate) for candidate in manager.buckets[1024])


@pytest.mark.parametrize("image_fit,no_upscale", product(["crop", "pad"], [False, True]))
def test_rotating_image_rotates_selected_bucket(image_fit, no_upscale):
    manager = BucketManager([1024], no_upscale=no_upscale)
    for width, height in [(522, 1000), (1031, 1000), (1001, 777), (41, 197), (4096, 2139)]:
        b = manager.assign(width, height, 1024, image_fit=image_fit)
        rotated = manager.assign(height, width, 1024, image_fit=image_fit)
        assert b.key == (rotated.height, rotated.width)


@pytest.mark.parametrize("image_fit", ["crop", "pad"])
def test_equal_framing_prefers_original_scale_then_more_retained_resolution(image_fit):
    manager = BucketManager([1536])
    # All three square buckets are legal; exact framing must not cause needless
    # enlargement or shrinking when the source already fits one of them.
    for source in (1472, 1536, 1600):
        assert manager.assign(source, source, 1536, image_fit=image_fit).key == (source, source)
    assert manager.assign(3000, 3000, 1536, image_fit=image_fit).key == (1600, 1600)


@pytest.mark.parametrize("image_fit", ["crop", "pad"])
def test_no_upscale_applies_before_selection_and_never_enlarges_pixels(image_fit):
    manager = BucketManager([1024], no_upscale=True)
    for width, height in [(522, 1000), (101, 79), (256, 256), (17, 511), (4096, 1536)]:
        b = manager.assign(width, height, 1024, image_fit=image_fit)
        rw, rh, *_ = (
            fit_crop(width, height, *b.key)
            if image_fit == "crop"
            else fit_pad(width, height, *b.key, max_scale=1)
        )
        assert rw <= width and rh <= height
        assert b.width % 16 == b.height % 16 == 0
    assert manager.assign(256, 256, 1024, image_fit=image_fit).key == (256, 256)


def test_sub_alignment_no_upscale_requires_padding():
    manager = BucketManager([1024], no_upscale=True)
    with pytest.raises(ValueError, match="保留完整画面"):
        manager.assign(3, 71, 1024)
    b = manager.assign(3, 71, 1024, image_fit="pad")
    rw, rh, *_ = fit_pad(3, 71, *b.key, max_scale=1)
    assert rw <= 3 and rh <= 71


def test_empty_band_keeps_historical_aligned_square_fallback():
    assert BucketManager([32]).assign(128, 128, 32).key == (64, 64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step": 0},
        {"align": 0},
        {"aspect_ratio_limit": float("inf")},
        {"aspect_ratio_limit": 0.5},
        {"area_tolerance": 1},
    ],
)
def test_invalid_grid_constraints_fail_before_enumeration(kwargs):
    with pytest.raises(ValueError):
        BucketManager([1024], **kwargs)
