"""Real two-rank DTensor state IO on CPU, without requiring FSDP CUDA kernels."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from ypuddin.config import TrainConfig
from ypuddin.train.sharded_state import (
    export_sharded_model_artifact,
    gather_full_training_state,
    load_sharded_checkpoint,
    read_sharded_checkpoint_metadata,
    save_sharded_checkpoint,
)
from ypuddin.train.state import Progress


def _model(mesh, dtype):
    model = nn.Module()
    for name, shape in (("matrix", (5, 3)), ("bias", (5,)), ("projector", (1, 7))):
        full = torch.linspace(-0.4, 0.6, torch.Size(shape).numel(), dtype=dtype).reshape(shape)
        model.register_parameter(
            name, nn.Parameter(distribute_tensor(full, mesh, [Shard(1 if name == "projector" else 0)]))
        )
    model.register_buffer("counter", torch.tensor(3))
    return model


def _step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.float().square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    return loss.full_tensor().item()


def _snapshot(model, optimizer):
    return {
        "weights": {name: value.to_local().clone() for name, value in model.named_parameters()},
        "states": {
            id(parameter): {
                key: value.to_local().clone()
                if isinstance(value, DTensor)
                else value.clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in state.items()
            }
            for parameter, state in optimizer.state.items()
        },
    }


def _assert_values(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_values(left[key], right[key])
    else:
        assert left == right


def _worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        results = []
        for dtype, optimizer_kind in (
            (torch.float32, "adamw"),
            (torch.bfloat16, "adamw"),
            (torch.float32, "sgd"),
            (torch.float32, "adafactor"),
        ):
            model = _model(mesh, dtype)
            modules = {"backbone": model}
            optimizer = (
                torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
                if optimizer_kind == "adamw"
                else torch.optim.SGD(model.parameters(), lr=0.01)
            )
            if optimizer_kind == "adafactor":
                from transformers.optimization import Adafactor

                from ypuddin.optim.sharded import prepare_sharded_optimizer

                optimizer = prepare_sharded_optimizer(
                    Adafactor(
                        model.parameters(),
                        lr=0.01,
                        relative_step=False,
                        scale_parameter=False,
                    )
                )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
            for _ in range(2):
                _step(model, optimizer)
                scheduler.step()
            step2 = gather_full_training_state(modules)
            checkpoint = directory / f"{dtype}-{optimizer_kind}"
            kwargs = {
                "modules": modules,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "sampler_state": {"epoch": 0, "position": 4, "rank": rank},
                "progress": Progress(step=2, batch_in_epoch=4, extra={"deterministic": False}),
                "rng": {"torch": torch.get_rng_state(), "rank_token": rank},
                "batch_size": 1,
                "grad_accum": 2,
                "config_hash": "config",
                "dataset_fingerprint": "dataset",
                "model_identity": "model",
            }
            # Any non-primary attempt to invoke the disk writer is an error.
            if rank == 1:
                import ypuddin.train.sharded_state as module

                original = module._atomic_directory
                module._atomic_directory = lambda *_: (_ for _ in ()).throw(AssertionError("rank1 wrote"))
            save_sharded_checkpoint(checkpoint, **kwargs)
            if rank == 1:
                module._atomic_directory = original
            if rank == 0:
                disk = load_file(checkpoint / "model.safetensors")
                _assert_values(step2, disk)
                assert (checkpoint / "complete.json").is_file()
            expected_loss = _step(model, optimizer)
            scheduler.step()
            reference = _snapshot(model, optimizer)
            checks = {
                "modules": modules,
                "optimizer": optimizer,
                "expected_world_size": 2,
                "expected_batch_size": 1,
                "expected_grad_accum": 2,
                "expected_dataset_fingerprint": "dataset",
                "expected_model_identity": "model",
                "expected_deterministic": False,
            }
            before = _snapshot(model, optimizer)
            for key, wrong in (
                ("expected_model_identity", "other"),
                ("expected_dataset_fingerprint", "other"),
                ("expected_world_size", 3),
                ("expected_batch_size", 2),
                ("expected_grad_accum", 1),
                ("expected_deterministic", True),
            ):
                with pytest.raises(ValueError):
                    load_sharded_checkpoint(checkpoint, **{**checks, key: wrong})
                _assert_values(before, _snapshot(model, optimizer))
            # Load into an optimizer whose state is empty, proving restoration
            # needs neither a fake optimizer step nor preallocated moment states.
            optimizer.state.clear()
            ids = [id(parameter) for parameter in model.parameters()]
            loaded = load_sharded_checkpoint(checkpoint, **checks)
            assert ids == [id(parameter) for parameter in model.parameters()]
            assert loaded["sampler"]["rank"] == rank
            assert loaded["rng"]["distributed"]["ranks"][rank]["rank_token"] == rank
            assert loaded["progress"].step == 2
            scheduler.load_state_dict(loaded["scheduler"])
            assert _step(model, optimizer) == expected_loss
            scheduler.step()
            _assert_values(reference, _snapshot(model, optimizer))
            if dtype == torch.float32 and optimizer_kind == "adamw":
                cfg = TrainConfig.model_validate(
                    {
                        "model": {"family": "toy"},
                        "training": {"mode": "full"},
                        "checkpoint": {"save_dtype": "fp32"},
                    }
                )
                artifact = export_sharded_model_artifact(
                    directory / "native.model",
                    SimpleNamespace(modules=modules),
                    cfg,
                    SimpleNamespace(),
                )
                values = gather_full_training_state(modules)
                if rank == 0:
                    native = load_file(artifact / "backbone/model.safetensors")
                    _assert_values(native, {k.removeprefix("backbone."): v for k, v in values.items()})
                    assert (
                        json.loads((artifact / "manifest.json").read_text())["format"]
                        == "ypuddin-full-model-v1"
                    )
                # A primary disk error propagates to both ranks and leaves the
                # previous complete directory and its model intact.
                import ypuddin.train.sharded_state as module

                original = module.save_file
                if rank == 0:
                    module.save_file = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full"))
                with pytest.raises(RuntimeError, match="disk full"):
                    save_sharded_checkpoint(checkpoint, **kwargs)
                module.save_file = original
                if rank == 0:
                    _assert_values(step2, load_file(checkpoint / "model.safetensors"))
                    assert not list(directory.glob("." + checkpoint.name + ".tmp-*"))
            results.append({"dtype": str(dtype), "optimizer": optimizer_kind, "exact_next_step": True})
        if rank == 0:
            (directory / "result.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_two_rank_native_export_optimizer_resume_and_preflight(tmp_path):
    context = mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        assert context.join(timeout=90) or context.join(timeout=10), "DTensor workers timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    assert len(json.loads((tmp_path / "result.json").read_text())) == 4


def test_legacy_or_incomplete_states_are_rejected_before_loading(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"format": 3, "training_kind": "full-model"}))
    with pytest.raises(ValueError, match="分布方式不同"):
        read_sharded_checkpoint_metadata(tmp_path)
    (tmp_path / "state.json").write_text(
        json.dumps({"format": 4, "strategy": "fsdp2", "training_kind": "full-model"})
    )
    with pytest.raises(ValueError, match="未完整写入"):
        read_sharded_checkpoint_metadata(tmp_path)


def test_staging_checkpoint_is_never_discovered_as_resumable(tmp_path):
    from ypuddin.train.sharded_state import _atomic_directory

    checkpoint = tmp_path / "state-4"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text("old complete state")

    def fail_after_metadata(temporary):
        (temporary / "state.json").write_text("unfinished replacement")
        discovered = [p for p in tmp_path.glob("state-*") if (p / "state.json").is_file()]
        assert discovered == [checkpoint]
        assert (checkpoint / "state.json").read_text() == "old complete state"
        raise OSError("interrupted write")

    with pytest.raises(OSError, match="interrupted write"):
        _atomic_directory(checkpoint, fail_after_metadata)
    assert (checkpoint / "state.json").read_text() == "old complete state"
    assert not list(tmp_path.glob(".state-4.tmp-*"))
