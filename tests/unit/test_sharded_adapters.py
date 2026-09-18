"""Real two-process FSDP: frozen base shards, adapter export and cold restore."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from ypuddin.adapters import inject
from ypuddin.adapters.rules import TargetPreset
from ypuddin.config import AdapterConfig
from ypuddin.train.sharded import parameter_shard, sharding_groups
from ypuddin.train.sharded_adapters import adapter_modules, gathered_adapter_export, shardable_frozen_weights
from ypuddin.train.sharded_state import load_sharded_checkpoint, save_sharded_checkpoint
from ypuddin.train.state import Progress


def worker(rank, directory, algo, dtype):
    directory = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=50),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))

        def build(shard=True):
            torch.manual_seed(41)
            model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 8))
            model.to(dtype)
            model.requires_grad_(False)
            adapters = inject(
                model, AdapterConfig(algo=algo, rank=2, alpha=2), TargetPreset("all", ("*",), (), "all")
            )
            shardable_frozen_weights(model)
            frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
            if shard:
                for _, group in [*sharding_groups(model, []), ("", model)]:
                    fully_shard(
                        group,
                        mesh=mesh,
                        reshard_after_forward=True,
                        shard_placement_fn=lambda p: parameter_shard(p, 2),
                    )
            optimizer = torch.optim.AdamW(adapters.parameters(), lr=0.01)
            return model, adapters, optimizer, frozen

        model, adapters, optimizer, frozen = build()

        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 + rank
            model(x.to(dtype)).float().square().mean().backward()
            optimizer.step()

        step(model, optimizer)
        modules = adapter_modules(adapters)
        save_sharded_checkpoint(
            directory / "state",
            modules=modules,
            optimizer=optimizer,
            scheduler=None,
            sampler_state={},
            progress=Progress(),
            rng={},
            batch_size=1,
            grad_accum=1,
            training_kind="adapter",
        )
        step(model, optimizer)
        expected = gathered_adapter_export(adapters)
        plain, portable, _, _ = build(shard=False)
        portable.load_state(expected)
        model.eval()
        plain.eval()
        x = torch.arange(16, dtype=dtype).reshape(2, 8) / 16
        with torch.no_grad():
            torch.testing.assert_close(model(x), plain(x), rtol=0, atol=0)
        model.reshard()
        for name, p in model.named_parameters():
            assert isinstance(p, DTensor), name
            if not p.requires_grad:
                assert torch.equal(p.full_tensor(), frozen[name]), name
        restored, aset, opt, _ = build()
        load_sharded_checkpoint(
            directory / "state",
            modules=adapter_modules(aset),
            optimizer=opt,
            expected_training_kind="adapter",
        )
        step(restored, opt)
        actual = gathered_adapter_export(aset)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert torch.equal(actual[key], expected[key]), key
        (directory / f"rank{rank}.json").write_text(json.dumps({"passed": True, "tensors": len(actual)}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_real_fsdp_adapter_shards_and_resume(tmp_path, algo, dtype):
    mp.spawn(worker, args=(str(tmp_path), algo, dtype), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f"rank{rank}.json").read_text())["passed"] for rank in range(2))


def test_fp8_rejected_before_any_frozen_buffer_is_mutated():
    from ypuddin.adapters.frozen import FrozenLinear

    model = nn.Sequential(
        FrozenLinear(torch.ones(4, 4)), FrozenLinear(torch.ones(4, 4), precision="fp8_e4m3")
    )
    with pytest.raises(ValueError, match="FP8"):
        shardable_frozen_weights(model)
    assert "weight" in model[0]._buffers and not dict(model[0].named_parameters())


@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_adapter_shard_memory_counts_frozen_base_without_optimizer_states(algo):
    from ypuddin.config import TrainConfig
    from ypuddin.train.plan import _fsdp_memory

    model = nn.Sequential(nn.Linear(32, 32, dtype=torch.bfloat16))
    model.requires_grad_(False)
    adapters = inject(
        model, AdapterConfig(algo=algo, rank=2, alpha=2), TargetPreset("all", ("*",), (), "all")
    )
    shardable_frozen_weights(model)
    cfg = TrainConfig.model_validate(
        {
            "loop": {"gpu_count": 2, "distributed_strategy": "fsdp"},
            "adapter": {"algo": algo},
            "memory": {"activation_checkpointing": "none"},
        }
    )
    estimate = _fsdp_memory(model, [], cfg)
    assert sum(estimate["local_trainable_bytes_by_rank"]) == sum(
        p.numel() * p.element_size() for p in adapters.parameters()
    )
    assert sum(estimate["local_parameter_bytes_by_rank"]) == sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    assert estimate["optimizer_state_bytes_by_rank"] == [
        2 * b for b in estimate["local_trainable_bytes_by_rank"]
    ]
    assert estimate["replicated_buffer_bytes"] == 0
