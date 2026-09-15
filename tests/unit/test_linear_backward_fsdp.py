"""Real two-rank CPU FSDP2, mixed precision, recomputation and exact resume."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed import fsdp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers.optimization import Adafactor

from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import resolve_training_compute_config
from ypuddin.optim.sharded import prepare_sharded_optimizer
from ypuddin.train.linear_backward import (
    install_linear_bf16_forward_fp32_backward,
    validate_linear_backward_installation,
)
from ypuddin.train.sharded_state import load_sharded_checkpoint, save_sharded_checkpoint
from ypuddin.train.state import Progress


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(17, 13), nn.Linear(13, 7)])

    def forward(self, x):
        for block in self.blocks:
            x = torch.tanh(checkpoint(block, x, use_reentrant=False))
        return x


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return (value.to_local() if isinstance(value, DTensor) else value).detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    return value


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def worker(rank, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(path / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        # Older supported Torch releases have FSDP2 but no CPU execution. Only
        # that capability failure may skip; operator/checkpoint failures cannot.
        probe = nn.Linear(4, 3)
        try:
            fsdp.fully_shard(probe, mesh=mesh)
            probe(torch.ones(2, 4)).sum().backward()
        except RuntimeError as error:
            reason = str(error)
            if any(
                text in reason.lower()
                for text in (
                    "requires cuda",
                    "does not support cpu",
                    "cpu device is not supported",
                    "device type cpu is not supported",
                )
            ):
                (path / f"rank-{rank}.json").write_text(json.dumps({"skip": reason}))
                return
            raise
        torch.manual_seed(442)
        model = Model()
        original_ids = [id(parameter) for parameter in model.parameters()]
        restore, counts = install_linear_bf16_forward_fp32_backward(model)
        assert original_ids == [id(parameter) for parameter in model.parameters()]
        for block in [*model.blocks, model]:
            fsdp.fully_shard(
                block,
                mesh=mesh,
                mp_policy=fsdp.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
                ),
                reshard_after_forward=True,
            )
        validate_linear_backward_installation(model, counts)
        optimizer = prepare_sharded_optimizer(
            Adafactor(model.parameters(), lr=1e-5, relative_step=False, scale_parameter=False, beta1=None)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)

        def state():
            return snapshot(
                {
                    "weights": dict(model.named_parameters()),
                    "optimizer": dict(optimizer.state),
                    "scheduler": scheduler.state_dict(),
                }
            )

        def step(index):
            generator = torch.Generator().manual_seed(442 + rank * 10 + index)
            x = torch.randn(2, 5, 17, generator=generator).requires_grad_()
            target = torch.randn(2, 5, 7, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                prediction = model(x)
                assert prediction.dtype == torch.bfloat16
                loss = (prediction.float() - target).square().mean()
            loss.backward()
            for parameter in model.parameters():
                assert isinstance(parameter, DTensor) and parameter.dtype == torch.float32
                assert parameter.grad.dtype == parameter.grad.to_local().dtype == torch.float32
                assert torch.isfinite(parameter.grad.to_local()).all()
            optimizer.step()
            scheduler.step()
            return loss.detach().clone()

        initial = state()["weights"]
        for index in range(2):
            step(index)
        cfg = TrainConfig.model_validate(
            {
                "model": {"family": "krea2"},
                "training": {"mode": "full"},
                "loop": {
                    "gpu_count": 2,
                    "distributed_strategy": "fsdp",
                    "deterministic": True,
                    "mixed_precision": "bf16",
                },
            }
        )
        _, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
        runtime = {"torch": str(torch.__version__), "device_type": "cpu", "scope": "CPU contract only"}
        saved_path = path / "state-2"
        save_sharded_checkpoint(
            saved_path,
            modules={"backbone": model},
            optimizer=optimizer,
            scheduler=scheduler,
            sampler_state={"rank": rank, "position": 2},
            progress=Progress(
                step=2, extra={"deterministic": True, "compute_policy": policy, "compute_runtime": runtime}
            ),
            rng={"torch": torch.get_rng_state(), "rank": rank},
            batch_size=1,
            grad_accum=1,
        )
        expected_loss = step(2)
        reference = state()
        assert not exact(initial, reference["weights"])
        checks = {
            "modules": {"backbone": model},
            "optimizer": optimizer,
            "expected_world_size": 2,
            "expected_deterministic": True,
            "expected_compute_policy": policy,
            "expected_compute_runtime": runtime,
        }
        for override in (
            {
                "expected_compute_policy": policy | {"linear_backward_implementation": "old-candidate"}
                if rank
                else policy
            },
            {"expected_compute_runtime": runtime | {"torch": "changed"} if rank else runtime},
        ):
            with pytest.raises(ValueError, match="计算"):
                load_sharded_checkpoint(saved_path, **(checks | override))
            assert exact(reference, state())
        optimizer.state.clear()
        ids = [id(parameter) for parameter in model.parameters()]
        loaded = load_sharded_checkpoint(saved_path, **checks)
        assert ids == [id(parameter) for parameter in model.parameters()]
        assert loaded["sampler"]["rank"] == rank
        scheduler.load_state_dict(loaded["scheduler"])
        assert torch.equal(expected_loss, step(2))
        assert exact(reference, state())
        validate_linear_backward_installation(model, counts)
        restore()
        (path / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "torch": str(torch.__version__),
                    "scope": "Actual two-rank CPU FSDP2, not DTK/GPU acceptance",
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_bf16_forward_fp32_backward_and_exact_sharded_resume(tmp_path):
    if not hasattr(fsdp, "fully_shard"):
        pytest.skip("Installed Torch lacks FSDP2")
    context = mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        assert context.join(timeout=60) or context.join(timeout=10), "FSDP workers timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    if any("skip" in result for result in results):
        pytest.skip("; ".join(result.get("skip", "") for result in results))
    assert all(result["passed"] for result in results)
