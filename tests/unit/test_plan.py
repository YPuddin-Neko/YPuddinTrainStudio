import math
import shutil

import pytest

from ypuddin.config import TrainConfig
from ypuddin.data import BucketBatchSampler, build_data
from ypuddin.models import get_family
from ypuddin.train.plan import plan


def config_for(path):
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(path), "repeats": 2, "resolutions": [80]}],
                "resolutions": [64],
                "bucket_step": 16,
                "batch_size": 3,
            },
            "loop": {"epochs": 3, "grad_accum": 2, "max_steps": 17},
        }
    )


def test_plan_matches_training_selection_and_step_count(image_dataset, tmp_path):
    validation = tmp_path / "heldout"
    validation.mkdir()
    shutil.copy(image_dataset / "img_000.jpg", validation / "renamed.jpg")
    raw = config_for(image_dataset).to_dict()
    raw["validation"] = {
        "enabled": True,
        "split_ratio": 0.25,
        "sources": [{"path": str(validation), "resolutions": [96]}],
    }
    cfg = TrainConfig.model_validate(raw)
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    result = plan(cfg, index_db_path=tmp_path / "plan.sqlite")
    assert result["ok"], result["errors"]
    sampler = BucketBatchSampler(bundle.train.bucket_keys(), cfg.dataset.batch_size)
    steps = math.ceil(len(sampler) / cfg.loop.grad_accum)
    assert result["images"] == bundle.plan.images
    assert result["items"] == len(bundle.train)
    assert result["validation_images"] == len(bundle.validation)
    assert result["steps_per_epoch"] == steps
    assert result["total_steps"] == min(3 * steps, 17)


@pytest.mark.parametrize("resolutions", [[], [0], [16384]])
def test_plan_reports_invalid_source_resolutions(image_dataset, resolutions):
    cfg = config_for(image_dataset)
    cfg.dataset.sources[0].resolutions = resolutions
    result = plan(cfg)
    assert result["ok"] is False
    assert any(error["loc"] == "dataset.sources.0.resolutions" for error in result["errors"])


def test_plan_reports_bad_bucket_alignment(image_dataset):
    cfg = config_for(image_dataset)
    cfg.dataset.bucket_step = 9
    result = plan(cfg)
    assert result["ok"] is False
    assert any(error["loc"] == "dataset.bucket_step" for error in result["errors"])


def test_plan_rejects_missing_empty_and_fully_held_out_data(image_dataset, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for source in (tmp_path / "missing", empty):
        result = plan(config_for(source))
        assert not result["ok"] and result["total_steps"] == 0
        assert result["errors"][0]["loc"].startswith("dataset.sources")
    assert not plan({"model": {"family": "toy"}})["ok"]
    cfg = config_for(image_dataset).to_dict()
    cfg["validation"] = {"enabled": True, "sources": [{"path": str(image_dataset)}]}
    result = plan(cfg)
    assert not result["ok"] and result["total_steps"] == 0
    assert "no training images remain" in result["errors"][0]["msg"]


def test_plan_maps_config_errors_without_raising():
    result = plan({"model": {"family": "toy"}, "dataset": {"batch_size": 0}})
    assert not result["ok"]
    assert result["errors"][0]["loc"] == "dataset.batch_size"


def test_plan_hardware_checks_only_apply_with_execution_device(image_dataset):
    cfg = config_for(image_dataset)
    cfg.memory.base_precision = "fp8_e4m3"
    assert plan(cfg)["ok"]  # Offline planning can describe a target CUDA workstation.
    for device in ("mps", "cpu"):
        result = plan(cfg, device=device)
        assert not result["ok"]
        assert any(error["loc"] == "memory.base_precision" for error in result["errors"])
    cfg.memory.base_precision = "auto"
    cfg.model.attention = "sage"
    assert any(error["loc"] == "model.attention" for error in plan(cfg, device="mps")["errors"])
    cfg.model.attention = "auto"
    cfg.memory.compile = True
    cfg.memory.blocks_to_swap = 1
    assert any(error["loc"] == "memory.compile" for error in plan(cfg)["errors"])


def test_plan_mps_uses_fp32_and_does_not_subtract_unified_memory_swap(image_dataset):
    cfg = config_for(image_dataset)
    cfg.dataset.sources[0].resolutions = [512]
    cfg.memory.blocks_to_swap = 1
    cuda = plan(cfg, device="cuda")
    mps = plan(cfg, device="mps")
    assert cuda["ok"] and mps["ok"]
    assert mps["memory"]["effective_dtype"] == "fp32"
    assert mps["memory"]["swapped_mb"] == 0
    assert mps["memory"]["peak_mb_estimate"] > cuda["memory"]["peak_mb_estimate"]
    assert any(warning["code"] == "device.mps_fp32" for warning in mps["warnings"])
