"""Capacity-critical placement and strategy admission for full-model sharding."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import Shard

from ypuddin.config import TrainConfig
from ypuddin.config.io import config_hash, read_config_file, write_config
from ypuddin.models import get_family
from ypuddin.train.sharded import ShardedTrainer, parameter_shard, sharding_groups
from ypuddin.train.state import Progress
from ypuddin.train.training_modes import FullTrainingSet


def _config(**changes):
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "training": {"mode": "full", "train_backbone": True},
            "loop": {"gpu_count": 2, "distributed_strategy": "fsdp"},
            "memory": {"activation_checkpointing": "block"},
            **changes,
        }
    )


def test_fsdp_accepts_full_block_checkpointing_without_changing_ddp_defaults():
    assert _config().loop.distributed_strategy == "fsdp"
    assert TrainConfig().loop.distributed_strategy == "ddp"
    with pytest.raises(ValueError, match="多卡数据并行"):
        _config(loop={"gpu_count": 2, "distributed_strategy": "ddp"})


@pytest.mark.parametrize(
    "change, message",
    [
        ({"loop": {"gpu_count": 1, "distributed_strategy": "fsdp"}}, "至少需要两张"),
        ({"training": {"mode": "adapter"}}, "主模型全量微调"),
        ({"training": {"mode": "full", "train_text_encoder": True}}, "暂不支持同时训练"),
        ({"optimizer": {"type": "adam"}}, "请选择 AdamW"),
        ({"loop": {"gpu_count": 2, "distributed_strategy": "fsdp", "ema": True}}, "EMA"),
    ],
)
def test_fsdp_incompatible_configs_rejected_before_training(change, message):
    with pytest.raises(ValueError, match=message):
        _config(**change)


def test_shard_axis_handles_krea_projector_without_empty_rank_or_uneven_columns():
    assert parameter_shard(nn.Parameter(torch.empty(7, 5)), 2) == Shard(0)
    assert parameter_shard(nn.Parameter(torch.empty(1, 12)), 2) == Shard(1)
    with pytest.raises(ValueError, match="无法分到"):
        parameter_shard(nn.Parameter(torch.empty(1, 7)), 2)


def test_native_krea_blocks_include_text_refiners_and_keep_root_separate():
    from ypuddin.models.krea2.vendor.krea2_mmdit import KREA2_CONFIG, SingleStreamDiT

    with torch.device("meta"):
        backbone = SingleStreamDiT(KREA2_CONFIG)
    groups = sharding_groups(backbone, backbone.blocks)
    names = [name for name, _ in groups]
    assert len(names) == len(set(names)) == 32  # 28 main blocks plus 4 text refiners
    assert sum(name.startswith("blocks.") for name in names) == 28
    assert sum(name.startswith("txtfusion.") for name in names) == 4
    assert "" not in names
    assert sum(p.numel() for p in backbone.parameters()) == 12_820_073_036


def test_optimizer_rebind_uses_replaced_parameters():
    module = nn.Linear(3, 2)
    training = FullTrainingSet({"backbone": module})
    old_weight = module.weight
    module.weight = nn.Parameter(module.weight.detach().clone())
    training.rebind_parameters()
    groups = training.param_groups(1e-3, 0)
    assert any(p is module.weight for g in groups for p in g["params"])
    assert all(p is not old_weight for g in groups for p in g["params"])


def _serialized_preparation_worker(rank, directory):
    import ypuddin.train.sharded as module

    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        checkpoint = directory / "state-2"
        original = _config(checkpoint={"output_dir": str(directory)})
        if rank == 0:
            checkpoint.mkdir()
            write_config(original, directory / "config.toml")
            (checkpoint / "state.json").write_text(
                json.dumps(
                    {
                        "format": 4,
                        "strategy": "fsdp2",
                        "training_kind": "full-model",
                        "config_hash": config_hash(original),
                        "progress": {"total_steps": 8},
                    }
                )
            )
            (checkpoint / "complete.json").write_text('{"format": 4}')
        dist.barrier()
        trainer = object.__new__(ShardedTrainer)
        trainer.cfg = _config(
            checkpoint={"resume": str(checkpoint), "output_dir": str(directory)},
            scheduler={"type": "constant"},
            memory={"activation_checkpointing": "none"},
        )
        trainer.device = torch.device("cpu")
        trainer.distributed = SimpleNamespace(rank=rank, world_size=2)
        trainer.is_primary = rank == 0
        trainer.family = get_family("toy")
        trainer.run_dir = directory
        trainer._stop = None
        trainer._preparing = False
        read_contract = module.read_sharded_scheduler_contract

        def record_read(path):
            contract = read_contract(path)
            (directory / f"read-{rank}.json").write_text(json.dumps(contract))
            return contract

        def prepare_without_loading_models(self):
            # Preserve the real local capability method while avoiding CUDA
            # allocations; the real owner/control collectives use CPU below.
            self.device = torch.device("cuda")
            try:
                self._check_capabilities()
            finally:
                self.device = torch.device("cpu")
            assert all((directory / f"read-{i}.json").exists() for i in range(2))
            if self.is_primary:
                write_config(self.cfg, directory / "config.toml")
            else:
                assert (directory / "owner-0.json").is_file()
                assert read_config_file(directory / "config.toml")["scheduler"]["type"] == "constant"
            (directory / f"owner-{rank}.json").write_text(json.dumps(self._resume_scheduler_contract))

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, "read_sharded_scheduler_contract", record_read)
            patch.setattr(module.Trainer, "prepare_data", prepare_without_loading_models)
            # Keep actual DistributedTrainer.prepare_data, all broadcasts,
            # control-request all_reduces and the preflight all_gather intact.
            trainer.prepare_data()
        assert trainer._resume_scheduler_contract == {
            "config": original.scheduler.model_dump(mode="json"),
            "total_steps": 8,
        }
    finally:
        dist.destroy_process_group()


def test_legacy_scheduler_preflight_precedes_real_serialized_owner_preparation(tmp_path):
    context = mp.spawn(_serialized_preparation_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        assert context.join(timeout=50) or context.join(timeout=10), "serialized preparation timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    contracts = [json.loads((tmp_path / f"owner-{rank}.json").read_text()) for rank in range(2)]
    assert contracts[0] == contracts[1]
    assert contracts[0]["config"]["type"] == "cosine"
    assert read_config_file(tmp_path / "config.toml")["scheduler"]["type"] == "constant"


def test_resume_checks_current_scheduler_and_step_budget_before_loading(tmp_path, monkeypatch):
    import ypuddin.train.sharded as module

    trainer = object.__new__(ShardedTrainer)
    trainer.cfg = _config(scheduler={"type": "constant"}, checkpoint={"output_dir": str(tmp_path)})
    model = nn.Linear(3, 2)
    trainer.optimizer = torch.optim.AdamW(model.parameters())
    trainer.adapters = SimpleNamespace(modules={"backbone": model})
    trainer.distributed = SimpleNamespace(world_size=2)
    trainer.bundle = SimpleNamespace(plan=SimpleNamespace(fingerprint="dataset"))
    trainer.model_identity = "model"
    trainer.progress = Progress(total_steps=9)
    trainer._resume_scheduler_contract = {"config": {"type": "cosine"}, "total_steps": 8}

    def reject_before_load(path, **kwargs):
        assert kwargs["expected_scheduler_config"] == trainer.cfg.scheduler.model_dump(mode="json")
        assert kwargs["expected_total_steps"] == 9
        assert kwargs["legacy_scheduler_contract"] is trainer._resume_scheduler_contract
        raise ValueError("scheduler preflight rejected")

    monkeypatch.setattr(module, "load_sharded_checkpoint", reject_before_load)
    with pytest.raises(ValueError, match="scheduler preflight rejected"):
        trainer._resume(tmp_path / "state-2")
