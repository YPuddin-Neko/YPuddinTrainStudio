"""Balance estimates use real post-validation items, never indexed source totals."""

import shutil
from dataclasses import replace

import pytest
from PIL import Image

from ypuddin.config import TrainConfig
from ypuddin.data.dataset import build_data
from ypuddin.models import get_family
from ypuddin.server.models import Plan
from ypuddin.train.plan import plan


def pictures(root, count, offset):
    root.mkdir()
    for i in range(count):
        Image.new("RGB", (64, 64), (offset + i, 10, 20)).save(root / f"{i}.png")
        (root / f"{i}.txt").write_text(f"caption {offset + i}")
    return root


def fixture_config(tmp_path):
    first = pictures(tmp_path / "first", 3, 10)
    second = pictures(tmp_path / "second", 2, 30)
    reg = pictures(tmp_path / "regularization", 1, 50)
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    shutil.copy(first / "0.png", heldout / "validation.png")
    (first / "broken.png").write_bytes(b"not an image")
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [
                    {"path": str(first), "repeats": 3, "resolutions": [64, 80]},
                    {"path": str(second), "repeats": 1},
                    {"path": str(reg), "repeats": 4, "is_reg": True, "prior_weight": 0.2},
                ],
                "resolutions": [64],
                "bucket_step": 16,
            },
            "validation": {"enabled": True, "sources": [{"path": str(heldout)}]},
        }
    )


def test_balance_matches_real_training_items_and_separates_regularization(tmp_path):
    cfg = fixture_config(tmp_path)
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    result = plan(cfg)
    rows = result["source_balance"]
    assert [row["images"] for row in rows] == [2, 2, 1]
    assert [row["repeated_images"] for row in rows] == [6, 2, 4]
    assert [row["resolution_variants"] for row in rows] == [2, 1, 1]
    assert [row["items"] for row in rows] == [12, 2, 4]
    assert [row["is_reg"] for row in rows] == [False, False, True]
    assert sum(row["items"] for row in rows) == len(bundle.train) == result["items"] == 18
    assert sum(row["images"] for row in rows) == 5
    assert Plan.model_validate(result).model_dump()["source_balance"] == rows


def test_repeats_update_balance_and_native_mode_does_not_multiply_sizes(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg.dataset.sources[1].repeats = 5
    result = plan(cfg)
    assert [row["items"] for row in result["source_balance"]] == [12, 10, 4]
    cfg.dataset.resolution_mode = "native"
    native = plan(cfg)
    assert [row["resolution_variants"] for row in native["source_balance"]] == [1, 1, 1]
    assert [row["items"] for row in native["source_balance"]] == [6, 10, 4]


def test_balance_is_before_distributed_tail_dropping(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg.loop.gpu_count = 2
    cfg.dataset.batch_size = 5
    result = plan(cfg)
    assert sum(row["items"] for row in result["source_balance"]) == result["items"]
    assert result["distributed"]["tail_policy"] == "drop_incomplete_rank_group"


def test_incomplete_model_and_sampling_settings_keep_a_truthful_data_preview(tmp_path):
    raw = fixture_config(tmp_path).to_dict()
    raw["model"]["dit_path"] = None
    raw["sampling"]["enabled"] = True
    raw["sampling"]["prompts"] = []
    result = plan(raw)
    assert not result["ok"]
    assert [row["items"] for row in result["source_balance"]] == [12, 2, 4]
    assert "params" not in result  # No weights were loaded to create this data preview.


def test_incomplete_preview_still_enforces_family_caption_capabilities(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path)
    family = get_family("toy")
    monkeypatch.setattr(family, "spec", replace(family.spec, caption_formats=("txt",)))
    source = cfg.dataset.sources[0]
    from pathlib import Path

    (Path(source.path) / "1.json").write_text('{"tags":42}')
    raw = cfg.to_dict()
    raw["sampling"]["enabled"] = True
    raw["sampling"]["prompts"] = []
    assert plan(raw)["source_balance"][0]["items"] == 12  # Auto reads TXT.
    raw["dataset"]["sources"][0]["caption_ext"] = ".json"
    rejected = plan(raw)
    assert rejected["source_balance"] is None
    assert any("does not support JSON" in issue["msg"] for issue in rejected["errors"])


@pytest.mark.parametrize("incomplete", [False, True])
def test_invalid_source_never_fabricates_zero_balance(tmp_path, incomplete):
    cfg = fixture_config(tmp_path).to_dict()
    cfg["dataset"]["sources"][0]["path"] = str(tmp_path / "missing")
    if incomplete:
        cfg["sampling"].update(enabled=True, prompts=[])
    result = plan(cfg)
    assert not result["ok"] and result["source_balance"] is None
