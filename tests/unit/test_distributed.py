"""Multi-device admission, bucket ownership and plan contract."""

import math
from types import SimpleNamespace

import pytest

from ypuddin.config import TrainConfig
from ypuddin.data import BucketBatchSampler, build_data
from ypuddin.data.native import NativeBatchSampler
from ypuddin.models import get_family
from ypuddin.train import Trainer, train
from ypuddin.train.distributed import DistributedTrainer
from ypuddin.train.plan import plan


def test_bucket_shards_keep_shapes_and_do_not_repeat_tail_images():
    shapes = [(64, 64)] * 7 + [(96, 64)] * 5 + [(64, 96)] * 3
    for epoch in (0, 1, 9):
        global_sampler = BucketBatchSampler(shapes, 2, seed=13)
        global_plan = global_sampler.plan(epoch)
        shards = [
            BucketBatchSampler(shapes, 2, seed=13, rank=rank, world_size=3).plan(epoch) for rank in range(3)
        ]
        assert len({len(shard) for shard in shards}) == 1
        for rank, shard in enumerate(shards):
            assert shard == global_plan[rank : len(global_plan) - len(global_plan) % 3 : 3]
            assert all(len({shapes[index] for index in batch}) == 1 for batch in shard)
        indices = [index for shard in shards for batch in shard for index in batch]
        assert len(indices) == len(set(indices))


def test_distributed_plan_matches_actual_rank_batches(image_dataset, tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolutions": [64],
                "bucket_step": 16,
                "batch_size": 2,
            },
            "loop": {"gpu_count": 2, "grad_accum": 2, "epochs": 3},
        }
    )
    result = plan(cfg, device="cuda")
    assert result["ok"], result["errors"]
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    sampler = BucketBatchSampler(bundle.train.bucket_keys(), 2, world_size=2)
    assert result["steps_per_epoch"] == math.ceil(len(sampler) / 2)
    assert result["total_steps"] == 3 * result["steps_per_epoch"]
    assert result["distributed"]["effective_batch_size"] == 8
    assert result["distributed"]["batches_per_rank"] == len(sampler)


@pytest.mark.parametrize(
    "change",
    [
        {"memory": {"blocks_to_swap": 1}},
        {"memory": {"compile": True}},
        {"memory": {"activation_checkpointing": "block"}},
    ],
)
def test_incompatible_multi_gpu_settings_rejected_during_config(change):
    with pytest.raises(ValueError, match="多卡"):
        TrainConfig.model_validate({"model": {"family": "toy"}, "loop": {"gpu_count": 2}, **change})


def test_multi_gpu_never_falls_back_to_single_trainer(image_dataset, tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "loop": {"gpu_count": 2},
            "dataset": {"sources": [{"path": str(image_dataset)}]},
            "checkpoint": {"output_dir": str(tmp_path / "out")},
        }
    )
    with pytest.raises(ValueError, match="requires torchrun"):
        train(cfg, device="cpu")
    with pytest.raises(ValueError, match="requires torchrun"):
        Trainer(cfg, device="cpu").run()


def test_native_multi_gpu_plan_matches_actual_rank_batches(image_dataset, tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolution_mode": "native",
                "batch_size": 5,
                "native_max_pixels": 16384,
            },
            "loop": {"gpu_count": 2, "grad_accum": 2},
        }
    )
    result = plan(cfg, device="cpu")
    assert result["ok"], result["errors"]
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "native-cache")
    keys = bundle.train.bucket_keys()
    full_plan = NativeBatchSampler(keys, 5, seed=cfg.loop.seed).plan()
    shards = [NativeBatchSampler(keys, 5, seed=cfg.loop.seed, world_size=2, rank=i).plan() for i in (0, 1)]
    assert len(shards[0]) == len(shards[1]) == result["distributed"]["batches_per_rank"]
    assert set(sum(shards[0], [])).isdisjoint(sum(shards[1], []))
    assert result["distributed"]["dropped_samples"] == sum(map(len, full_plan[2 * len(shards[0]) :]))
    assert result["steps_per_epoch"] == math.ceil(len(shards[0]) / cfg.loop.grad_accum)


@pytest.mark.parametrize(
    "saved, message",
    [
        ({}, "same saved world size"),
        ({"world_size": 3}, "same saved world size"),
        ({"world_size": 2, "batch_size": 9, "grad_accum": 2}, "unchanged per-device batch"),
        ({"world_size": 2, "batch_size": 1, "grad_accum": 2, "ranks": [{}]}, "missing per-rank"),
    ],
)
def test_exact_resume_rejects_changed_distributed_shape(saved, message):
    trainer = object.__new__(DistributedTrainer)
    trainer.distributed = SimpleNamespace(world_size=2)
    trainer.cfg = SimpleNamespace(dataset=SimpleNamespace(batch_size=1), loop=SimpleNamespace(grad_accum=2))
    with pytest.raises(ValueError, match=message):
        trainer._restore_checkpoint_rng({"distributed": saved})
