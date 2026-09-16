"""Real CPU training proves LR closure identity survives exact resume."""

import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.checkpoint_assertions import assert_checkpoint_value_exact
from tests.conftest import make_image_dataset
from tests.e2e.test_toy_training import _cfg
from ypuddin.config import config_hash, load_config, write_config
from ypuddin.train import Trainer
from ypuddin.train.distributed import DistributedContext, DistributedTrainer
from ypuddin.train.scheduler_contract import read_resume_scheduler_contract
from ypuddin.train.state import load_checkpoint


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _config(images, output, *, mode="adapter", optimizer="adamw"):
    return _cfg(
        images,
        output,
        training={"mode": mode},
        optimizer={"type": optimizer, "lr": 1e-6 if optimizer == "automagic" else 0.001},
        scheduler={"type": "constant" if optimizer == "automagic" else "cosine", "warmup_steps": 0},
        loop={"epochs": None, "max_steps": 4, "deterministic": True},
        checkpoint={"save_every_epochs": None, "save_state_every_steps": 2},
        validation={"enabled": False},
        sampling={"enabled": False},
    )


@pytest.fixture
def reference(image_dataset, tmp_path):
    cfg = _config(image_dataset, tmp_path / "reference")
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    return cfg, trainer.run_dir / "state-2"


def _resuming(cfg, checkpoint, output):
    cfg = cfg.model_copy(deep=True)
    cfg.checkpoint.resume = str(checkpoint)
    cfg.checkpoint.output_dir = str(output)
    return cfg


def _snapshot(trainer):
    return copy.deepcopy(
        {
            "weights": trainer.adapters.training_state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": None if trainer.scheduler is None else trainer.scheduler.state_dict(),
            "progress": trainer.progress.to_dict(),
            "sampler": trainer.sampler.state_dict(),
            "rng": trainer._capture_local_checkpoint_rng(),
        }
    )


def _assert_checkpoint_exact(actual, expected):
    for key in ("training", "optimizer", "scheduler", "progress", "sampler", "rng"):
        a, b = actual[key], expected[key]
        if key == "progress":
            a, b = a.to_dict(), b.to_dict()
        assert_checkpoint_value_exact(a, b, key)


@pytest.mark.parametrize("mode", ["adapter", "full"])
@pytest.mark.parametrize("optimizer", ["adamw", "automagic"])
def test_toy_continuous_and_resumed_scheduler_and_all_training_state_exact(
    image_dataset, tmp_path, mode, optimizer
):
    cfg = _config(image_dataset, tmp_path / "reference", mode=mode, optimizer=optimizer)
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    checkpoint = trainer.run_dir / "state-2"
    contract = read_resume_scheduler_contract(checkpoint).contract
    assert contract["config"] == cfg.scheduler.model_dump(mode="json")
    assert contract["total_steps"] == 4
    assert contract["training_kind"] == ("full-model" if mode == "full" else "adapter")
    assert contract["scheduler_class"] == (
        None if optimizer == "automagic" else "torch.optim.lr_scheduler.LambdaLR"
    )
    assert contract["managed_by_optimizer"] == ("automagic" if optimizer == "automagic" else None)
    resumed = Trainer(_resuming(cfg, checkpoint, tmp_path / "resumed"), device="cpu")
    assert resumed.run() == "finished"
    _assert_checkpoint_exact(
        load_checkpoint(resumed.run_dir / "state-4"), load_checkpoint(trainer.run_dir / "state-4")
    )


@pytest.mark.parametrize("change", ["formula", "total_steps"])
def test_changed_recipe_is_rejected_before_loading_or_mutating_any_live_state(
    reference, tmp_path, monkeypatch, change
):
    import ypuddin.train.trainer as module

    cfg, checkpoint = reference
    current = cfg.model_copy(deep=True)
    current.checkpoint.output_dir = str(tmp_path / "prepared")
    if change == "formula":
        current.scheduler.type = "linear"
    else:
        current.loop.max_steps = 6
    trainer = Trainer(current, device="cpu")
    trainer.prepare()
    before = _snapshot(trainer)
    monkeypatch.setattr(module, "load_checkpoint", lambda _: pytest.fail("tensor state was loaded"))
    try:
        with pytest.raises(ValueError, match="学习率调度配置|总训练步数"):
            trainer._resume(str(checkpoint))
        assert_checkpoint_value_exact(_snapshot(trainer), before)
    finally:
        trainer._close_logs()
        trainer.emitter.close()


def _remove_contract(checkpoint):
    path = checkpoint / "state.json"
    metadata = json.loads(path.read_text())
    del metadata["progress"]["extra"]["scheduler_contract"]
    path.write_text(json.dumps(metadata))
    return metadata


def test_legacy_same_directory_resume_captures_original_config_before_overwrite(reference):
    cfg, checkpoint = reference
    expected = load_checkpoint(checkpoint.parent / "state-4")
    metadata = _remove_contract(checkpoint)
    assert config_hash(load_config(checkpoint.parent / "config.toml")) == metadata["config_hash"]
    resumed = Trainer(_resuming(cfg, checkpoint, checkpoint.parent), device="cpu")
    assert resumed.run() == "finished"
    # The resume path changes the saved config hash: re-reading this replacement
    # file as if it were historical would have rejected a valid legacy resume.
    assert config_hash(load_config(checkpoint.parent / "config.toml")) != metadata["config_hash"]
    _assert_checkpoint_exact(load_checkpoint(checkpoint.parent / "state-4"), expected)


@pytest.mark.parametrize("budget", ["max_steps", "epochs"])
def test_rejected_legacy_same_directory_budget_change_preserves_config_and_can_retry(reference, budget):
    cfg, checkpoint = reference
    expected = load_checkpoint(checkpoint.parent / "state-4")
    _remove_contract(checkpoint)
    config_path = checkpoint.parent / "config.toml"
    original = config_path.read_bytes()
    current = _resuming(cfg, checkpoint, checkpoint.parent)
    if budget == "max_steps":
        current.loop.max_steps = 6
    else:
        current.loop.epochs = 2
        current.loop.max_steps = None
    rejected = Trainer(current, device="cpu")
    with pytest.raises(ValueError, match="总训练步数"):
        rejected.run()
    assert config_path.read_bytes() == original
    retry = Trainer(_resuming(cfg, checkpoint, checkpoint.parent), device="cpu")
    assert retry.run() == "finished"
    _assert_checkpoint_exact(load_checkpoint(checkpoint.parent / "state-4"), expected)


@pytest.mark.parametrize("damage", ["missing", "changed", "current_hash"])
def test_legacy_config_must_be_original_and_is_never_replaced_before_rejection(
    reference, tmp_path, monkeypatch, damage
):
    cfg, checkpoint = reference
    _remove_contract(checkpoint)
    current = _resuming(cfg, checkpoint, checkpoint.parent)
    original = checkpoint.parent / "config.toml"
    if damage == "missing":
        original.unlink()
    else:
        changed = current if damage == "current_hash" else cfg.model_copy(deep=True)
        if damage == "changed":
            changed.scheduler.type = "linear"
        write_config(changed, original)
    before = original.read_bytes() if original.exists() else None
    trainer = Trainer(current, device="cpu")
    monkeypatch.setattr(trainer, "_seed_all", lambda: pytest.fail("model preparation began"))
    with pytest.raises(ValueError, match="旧训练状态.*无法验证精确恢复"):
        trainer.prepare_data()
    assert (original.read_bytes() if original.exists() else None) == before
    trainer.emitter.close()


def test_captured_legacy_recipe_cannot_be_reused_for_changed_checkpoint_metadata(reference):
    _, checkpoint = reference
    metadata = _remove_contract(checkpoint)
    captured = read_resume_scheduler_contract(checkpoint)
    metadata["progress"]["total_steps"] += 1
    (checkpoint / "state.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="预检后训练状态已改变"):
        read_resume_scheduler_contract(checkpoint, captured=captured)


def test_save_cannot_relabel_a_constructed_scheduler_after_config_mutation(image_dataset, tmp_path):
    cfg = _config(image_dataset, tmp_path / "prepared")
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    before = trainer.scheduler.lr_lambdas[0](3)
    trainer.cfg.scheduler.type = "linear"
    assert trainer.scheduler.lr_lambdas[0](3) == before
    try:
        with pytest.raises(ValueError, match="学习率调度配置"):
            trainer.save_state("wrong")
        assert not (trainer.run_dir / "state-wrong").exists()
    finally:
        trainer._close_logs()
        trainer.emitter.close()


@pytest.mark.parametrize("saved_scheduler", [{}, {"unexpected": 1}])
def test_scheduler_presence_is_checked_before_weights_or_optimizer_mutate(
    image_dataset, tmp_path, saved_scheduler
):
    managed = bool(saved_scheduler)
    cfg = _config(image_dataset, tmp_path / "reference", optimizer="automagic" if managed else "adamw")
    reference = Trainer(cfg, device="cpu")
    assert reference.run() == "finished"
    checkpoint = reference.run_dir / "state-2"
    torch.save(saved_scheduler, checkpoint / "scheduler.pt")
    current = cfg.model_copy(deep=True)
    current.checkpoint.output_dir = str(tmp_path / "prepared")
    trainer = Trainer(current, device="cpu")
    trainer.prepare()
    before = _snapshot(trainer)
    try:
        with pytest.raises(ValueError, match="保存的学习率调度器状态"):
            trainer._resume(str(checkpoint))
        assert_checkpoint_value_exact(_snapshot(trainer), before)
    finally:
        trainer._close_logs()
        trainer.emitter.close()


def test_optimizer_managed_lr_cannot_be_restored_into_an_external_scheduler(
    image_dataset, tmp_path, monkeypatch
):
    import ypuddin.train.trainer as module

    cfg = _config(image_dataset, tmp_path / "reference", optimizer="automagic")
    reference = Trainer(cfg, device="cpu")
    assert reference.run() == "finished"
    current = cfg.model_copy(deep=True)
    current.optimizer.type = "adamw"
    current.checkpoint.output_dir = str(tmp_path / "prepared")
    trainer = Trainer(current, device="cpu")
    trainer.prepare()
    assert trainer.scheduler is not None and reference.scheduler is None
    before = _snapshot(trainer)
    monkeypatch.setattr(module, "load_checkpoint", lambda _: pytest.fail("tensor state was loaded"))
    try:
        with pytest.raises(ValueError, match="学习率管理方式"):
            trainer._resume(str(reference.run_dir / "state-2"))
        assert_checkpoint_value_exact(_snapshot(trainer), before)
    finally:
        trainer._close_logs()
        trainer.emitter.close()


def _ddp_legacy_worker(rank, directory):
    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        cfg = _config(directory / "images", directory / "shared")
        cfg.loop.gpu_count = 2
        cfg.dataset.cache_dir = str(directory / "cache")
        context = DistributedContext(rank, 2, rank, torch.device("cpu"), "gloo")
        reference = DistributedTrainer(cfg, context=context)
        assert reference.run() == "finished"
        expected = copy.deepcopy(load_checkpoint(reference.run_dir / "state-4"))
        checkpoint = reference.run_dir / "state-2"
        if rank == 0:
            _remove_contract(checkpoint)
        dist.barrier()
        resumed = DistributedTrainer(_resuming(cfg, checkpoint, reference.run_dir), context=context)
        assert resumed.run() == "finished"
        _assert_checkpoint_exact(load_checkpoint(reference.run_dir / "state-4"), expected)
        (directory / f"rank-{rank}.json").write_text(
            json.dumps({"passed": True, "step": resumed.progress.step})
        )
    finally:
        dist.destroy_process_group()


def test_real_two_rank_ddp_legacy_same_directory_resume_does_not_deadlock(tmp_path):
    make_image_dataset(tmp_path / "images", n=8, sizes=((64, 64),))
    mp.spawn(_ddp_legacy_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)] == [
        {"passed": True, "step": 4},
        {"passed": True, "step": 4},
    ]


def _fsdp_save_guard_worker(rank, directory):
    from torch.distributed.device_mesh import init_device_mesh

    from tests.unit.test_sharded_state import _model, _step
    from tests.unit.test_sharded_state import _snapshot as local_snapshot
    from ypuddin.config import TrainConfig
    from ypuddin.data import BucketBatchSampler
    from ypuddin.models import get_family
    from ypuddin.optim import build_scheduler
    from ypuddin.train.scheduler_contract import scheduler_recipe
    from ypuddin.train.sharded import ShardedTrainer
    from ypuddin.train.state import Progress
    from ypuddin.train.training_modes import FullTrainingSet

    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "sharded-rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        cfg = TrainConfig.model_validate(
            {
                "model": {"family": "toy", "dtype": "fp32"},
                "training": {"mode": "full", "train_backbone": True},
                "loop": {"gpu_count": 2, "distributed_strategy": "fsdp", "max_steps": 4},
                "memory": {"activation_checkpointing": "block"},
                "checkpoint": {"output_dir": str(directory / "shared")},
            }
        )
        context = DistributedContext(rank, 2, rank, torch.device("cpu"), "gloo")
        trainer = ShardedTrainer(cfg, context=context)
        model = _model(init_device_mesh("cpu", (2,)), torch.float32)
        trainer.adapters = FullTrainingSet({"backbone": model})
        trainer.optimizer = torch.optim.AdamW(trainer.adapters.parameters(), lr=0.01, foreach=False)
        trainer.scheduler = build_scheduler(cfg.scheduler.model_copy(deep=True), trainer.optimizer, 4)
        trainer._scheduler_contract = scheduler_recipe(cfg, 4)
        trainer.family = get_family("toy")
        trainer.bundle = SimpleNamespace(plan=SimpleNamespace(fingerprint="dataset"))
        trainer.model_identity = "model"
        trainer.sampler = BucketBatchSampler([(64, 64)] * 8, 1, rank=rank, world_size=2)
        trainer.progress = Progress(step=1, total_steps=4, batch_in_epoch=1)
        _step(model, trainer.optimizer)
        trainer.scheduler.step()
        checkpoint = trainer.save_state("valid")
        dist.barrier()  # Include rank zero's completed event write in file snapshots.
        metadata = json.loads((checkpoint / "state.json").read_text())
        assert metadata["format"] == 4 and metadata["strategy"] == "fsdp2"
        assert metadata["scheduler_contract"] == {
            "config": trainer._scheduler_contract["config"],
            "total_steps": 4,
        }
        assert "scheduler_contract" not in metadata["progress"]["extra"]

        def files():
            return {
                str(path.relative_to(trainer.run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in trainer.run_dir.rglob("*")
                if path.is_file()
            }

        checked = []
        # Both rank-zero and peer-only failures must reach both ranks, including
        # an attempted overwrite of an existing complete checkpoint.
        for changed_rank, change, tag in (
            (0, "formula", "valid"),
            (1, "formula", "bad-formula"),
            (1, "total", "bad-total"),
            (0, "class", "bad-class"),
        ):
            original_type = trainer.cfg.scheduler.type
            original_scheduler = trainer.scheduler
            if rank == changed_rank:
                if change == "formula":
                    trainer.cfg.scheduler.type = "linear"
                elif change == "total":
                    trainer.progress.total_steps = 6
                else:
                    trainer.scheduler = None
            before = {
                "state": local_snapshot(model, trainer.optimizer),
                "scheduler": None
                if trainer.scheduler is None
                else copy.deepcopy(trainer.scheduler.state_dict()),
                "progress": trainer.progress.to_dict(),
                "rng": trainer._capture_local_checkpoint_rng(),
            }
            before_files = files()
            with pytest.raises(
                ValueError, match="分片训练状态检查失败.*学习率|分片训练状态检查失败.*总训练步数"
            ):
                trainer.save_state(tag)
            after = {
                "state": local_snapshot(model, trainer.optimizer),
                "scheduler": None if trainer.scheduler is None else trainer.scheduler.state_dict(),
                "progress": trainer.progress.to_dict(),
                "rng": trainer._capture_local_checkpoint_rng(),
            }
            assert_checkpoint_value_exact(after, before)
            assert files() == before_files
            assert not list(trainer.run_dir.glob(".*.tmp-*"))
            trainer.cfg.scheduler.type = original_type
            trainer.scheduler = original_scheduler
            trainer.progress.total_steps = 4
            checked.append(f"rank{changed_rank}-{change}")
        # Rejection leaves all ranks usable; a subsequent valid save still works.
        assert trainer.save_state("after-rejection").joinpath("complete.json").is_file()
        (directory / f"sharded-rank-{rank}.json").write_text(json.dumps(checked))
        trainer.emitter.close()
    finally:
        dist.destroy_process_group()


def test_two_rank_sharded_save_rejects_single_rank_scheduler_drift_before_io(tmp_path):
    """Actual DTensor optimizer/collective IO; no CUDA FSDP kernels are needed."""
    context = mp.spawn(_fsdp_save_guard_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        assert context.join(timeout=50) or context.join(timeout=10), "sharded save validation timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in (0, 1):
        assert json.loads((tmp_path / f"sharded-rank-{rank}.json").read_text()) == [
            "rank0-formula",
            "rank1-formula",
            "rank1-total",
            "rank0-class",
        ]
