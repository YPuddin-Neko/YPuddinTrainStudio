"""Capacity-critical placement and strategy admission for full-model sharding."""

import pytest
import torch
from torch import nn
from torch.distributed.tensor import Shard

from ypuddin.config import TrainConfig
from ypuddin.train.sharded import parameter_shard, sharding_groups
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
