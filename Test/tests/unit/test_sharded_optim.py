"""Real CPU DTensor collectives exercise Adafactor's unchanged update and resume."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers.optimization import Adafactor

from ypuddin.optim.sharded import prepare_sharded_optimizer, sharded_optimizer_state_bytes


def _worker(rank: int, rendezvous: str):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        for shape, shard_dim in [((7, 5), 0), ((3, 7, 5), 0), ((7,), 0), ((1, 12), 1), ((7, 6), 1)]:
            for beta1 in [None, 0.9]:
                generator = torch.Generator().manual_seed(87)
                initial = torch.randn(shape, generator=generator)
                plain = torch.nn.Parameter(initial.clone())
                parameter = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [Shard(shard_dim)]))
                settings = dict(lr=1e-3, beta1=beta1, scale_parameter=False, relative_step=False)
                scalar = torch.nn.Parameter(torch.tensor(0.7))
                plain_scalar = torch.nn.Parameter(scalar.detach().clone())
                reference = Adafactor([plain, plain_scalar], **settings)
                optimizer = prepare_sharded_optimizer(Adafactor([parameter, scalar], **settings))
                assert prepare_sharded_optimizer(optimizer) is optimizer
                for step in range(3):
                    gradient = torch.randn(shape, generator=generator)
                    plain.grad = gradient.clone()
                    parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(shard_dim)])
                    scalar.grad = torch.tensor(float(rank + 1))
                    dist.all_reduce(scalar.grad)
                    scalar.grad.div_(2)
                    plain_scalar.grad = scalar.grad.clone()
                    reference.step()
                    optimizer.step()
                    torch.testing.assert_close(parameter.full_tensor(), plain, rtol=2e-6, atol=1e-7)
                    torch.testing.assert_close(scalar, plain_scalar, rtol=0, atol=0)
                    state = optimizer.state[parameter]
                    if len(shape) >= 2:
                        expected_row = (
                            Replicate()
                            if shard_dim == len(shape) - 1 or shape[shard_dim] == 1
                            else Shard(shard_dim)
                        )
                        assert state["exp_avg_sq_row"].placements == (expected_row,)
                        col_reduced_dim = len(shape) - 2
                        expected_col = (
                            Replicate()
                            if shard_dim == col_reduced_dim
                            else Shard(shard_dim - int(col_reduced_dim < shard_dim))
                        )
                        assert state["exp_avg_sq_col"].placements == (expected_col,)
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            assert isinstance(value, DTensor)
                            torch.testing.assert_close(value.full_tensor(), reference.state[plain][key])
                    audit = sharded_optimizer_state_bytes(optimizer)
                    assert audit["local_state_bytes"] >= 0
                    if step == 0:
                        buffer = io.BytesIO()
                        torch.save(optimizer.state_dict(), buffer)
                        buffer.seek(0)
                        reloaded = Adafactor([parameter, scalar], **settings)
                        reloaded.load_state_dict(torch.load(buffer, weights_only=False))
                        optimizer = prepare_sharded_optimizer(reloaded)
                # A legacy full/plain state must not be silently treated as a shard.
                key = "exp_avg_sq_row" if len(shape) >= 2 else "exp_avg_sq"
                optimizer.state[parameter][key] = optimizer.state[parameter][key].to_local().clone()
                with pytest.raises(ValueError, match="incompatible FSDP2 layout"):
                    optimizer.step()
        empty_shard = torch.nn.Parameter(distribute_tensor(torch.ones(1, 12), mesh, [Shard(0)]))
        with pytest.raises(ValueError, match="nonempty shards"):
            prepare_sharded_optimizer(Adafactor([empty_shard], relative_step=False, lr=1e-3))
    finally:
        dist.destroy_process_group()


def test_adafactor_dtensor_updates_factor_layout_and_resume(tmp_path: Path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_sharded_optimizer_rejects_plain_parameters():
    optimizer = Adafactor([torch.nn.Parameter(torch.ones(3, 2))], relative_step=False, lr=1e-3)
    with pytest.raises(ValueError, match="after sharding"):
        prepare_sharded_optimizer(optimizer)


def test_sharded_optimizer_rejects_unverified_optimizer():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(3, 2))])
    with pytest.raises(ValueError, match="currently supports"):
        prepare_sharded_optimizer(optimizer)
