"""Actual FSDP2 evaluation cleanup, checkpoint IO and next-step regression on CPU."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed import fsdp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers.optimization import Adafactor

from ypuddin.config import OptimizerConfig
from ypuddin.optim.sharded import prepare_sharded_optimizer
from ypuddin.train.sharded import ShardedTrainer, parameter_shard, sharding_groups
from ypuddin.train.sharded_state import _optimizer_state, load_sharded_checkpoint, save_sharded_checkpoint
from ypuddin.train.state import Progress
from ypuddin.train.training_modes import FullTrainingSet

FSDPModule = getattr(fsdp, "FSDPModule", None)
MixedPrecisionPolicy = getattr(fsdp, "MixedPrecisionPolicy", None)
fully_shard = getattr(fsdp, "fully_shard", None)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 7), nn.Linear(7, 3)])
        self.projector = nn.Linear(4, 1, bias=False)
        self.scalar = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.blocks[1](torch.tanh(self.blocks[0](x))) + self.projector(x) + self.scalar


def cpu_fsdp_unavailable(mesh):
    """Only capability failures in this isolated PyTorch preflight may skip."""
    if not hasattr(FSDPModule, "set_reshard_after_forward"):
        return "Installed FSDP2 lacks explicit root reshard control required by this regression"
    probe = nn.Linear(4, 3)
    try:
        fully_shard(probe, mesh=mesh, reshard_after_forward=True)
        probe(torch.ones(2, 4)).sum().backward()
        probe.reshard()
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
            return f"Installed PyTorch has no CPU FSDP2 execution: {reason}"
        raise
    return None


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return (value.to_local() if isinstance(value, DTensor) else value).detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def worker(rank, directory):
    path = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(path / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    report = {
        "rank": rank,
        "torch": torch.__version__,
        "scope": "CPU FSDP2 with root-only reshard_after_forward=False to reproduce vendor 2.7 behavior",
        "passed": False,
    }
    try:
        capability_mesh = init_device_mesh("cpu", (2,))
        unsupported = cpu_fsdp_unavailable(capability_mesh)
        if unsupported:
            report["unsupported"] = unsupported
            return
        torch.manual_seed(42)
        model = Model()
        training = FullTrainingSet({"backbone": model})
        mesh = init_device_mesh("cpu", (2,))
        ignored = {model.scalar}
        for _, module in [*sharding_groups(model, model.blocks), ("", model)]:
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=True,
                ignored_params=ignored,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
                ),
                shard_placement_fn=lambda p: parameter_shard(p, 2),
            )
            training.rebind_parameters()
        model.set_reshard_after_forward(False, recurse=False)
        optimizer = prepare_sharded_optimizer(
            Adafactor(training.parameters(), lr=0.001, relative_step=False, scale_parameter=False)
        )
        trainer = object.__new__(ShardedTrainer)
        trainer.device = torch.device("cpu")
        trainer.is_primary = rank == 0
        trainer.distributed = SimpleNamespace(rank=rank, world_size=2)
        trainer.adapters = training
        trainer.cfg = SimpleNamespace(optimizer=OptimizerConfig(type="adafactor"))
        trainer.loaded = SimpleNamespace(backbone=model)
        trainer.optimizer = optimizer
        trainer.swapper = None
        trainer.gen = torch.Generator().manual_seed(4)
        trainer.loader_gen = torch.Generator().manual_seed(5)
        trainer._fsdp_modules = [module for _, module in [*sharding_groups(model, model.blocks), ("", model)]]
        report["evaluation_cleanup"] = "Actual ShardedTrainer._evaluation product context"

        def step(n):
            model.train()
            g = torch.Generator().manual_seed(99 + rank * 10 + n)
            x = torch.randn(2, 4, generator=g)
            y = torch.randn(2, 3, generator=g)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().mean()
            loss.backward()
            trainer._gradient_norm_and_clip()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return float(loss.detach())

        def save(target):
            return save_sharded_checkpoint(
                target,
                modules=training.modules,
                optimizer=optimizer,
                scheduler=None,
                sampler_state={"epoch": 0, "position": 1, "rank": rank},
                progress=Progress(step=1),
                rng={"rank": rank},
                batch_size=2,
                grad_accum=1,
            )

        step(1)
        save(path / "before-eval")
        reference_loss = step(2)
        reference = snapshot({"weights": dict(model.named_parameters()), "optimizer": optimizer.state_dict()})
        load_sharded_checkpoint(path / "before-eval", modules=training.modules, optimizer=optimizer)
        with trainer._evaluation(), torch.no_grad():
            model(torch.ones(2, 4))
            registered = {id(p) for p in model.parameters()}
            missing = [name for name, p in training._names.items() if id(p) not in registered]
            report["unregistered_inside_eval"] = missing
            report["child_parameters_remain_dtensor"] = all(
                isinstance(p, DTensor) for block in model.blocks for p in block.parameters()
            )
            assert missing and report["child_parameters_remain_dtensor"]
            try:
                _optimizer_state(training.modules, optimizer)
            except ValueError as exc:
                report["reproduced_error"] = str(exc)
            else:
                raise AssertionError("Expected stale parameter binding before reshard")
        registered = {id(p) for p in model.parameters()}
        assert all(id(p) in registered for p in training.parameters())
        _optimizer_state(training.modules, optimizer)
        rng_before = trainer.gen.get_state().clone()
        try:
            with trainer._evaluation(), torch.no_grad():
                model(torch.ones(2, 4))
                torch.rand(3, generator=trainer.gen)
                raise RuntimeError("intentional evaluation failure")
        except RuntimeError as exc:
            assert str(exc) == "intentional evaluation failure"
        assert model.training and torch.equal(rng_before, trainer.gen.get_state())
        registered = {id(p) for p in model.parameters()}
        assert all(id(p) in registered for p in training.parameters())
        report["exception_restores_parameter_bindings_mode_rng"] = True
        save(path / "after-eval")
        load_sharded_checkpoint(path / "after-eval", modules=training.modules, optimizer=optimizer)
        resumed_loss = step(2)
        resumed = snapshot({"weights": dict(model.named_parameters()), "optimizer": optimizer.state_dict()})
        report.update(
            reference_loss=reference_loss,
            resumed_loss=resumed_loss,
            next_weights_exact=exact(reference["weights"], resumed["weights"]),
            next_optimizer_exact=exact(reference["optimizer"], resumed["optimizer"]),
        )
        report["passed"] = (
            report["next_weights_exact"] and report["next_optimizer_exact"] and reference_loss == resumed_loss
        )
        assert report["passed"], report
    except Exception as exc:
        import traceback

        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        (path / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


@pytest.mark.skipif(
    any(api is None for api in (FSDPModule, MixedPrecisionPolicy, fully_shard)),
    reason="Installed PyTorch lacks the public FSDP2 API required by this regression",
)
def test_evaluation_reshards_before_save_and_restores_after_errors(tmp_path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    if all(report.get("unsupported") for report in reports):
        pytest.skip(reports[0]["unsupported"])
    assert all(report["passed"] for report in reports), reports
