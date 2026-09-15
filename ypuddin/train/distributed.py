"""Explicit torchrun/DDP training through the existing family, loss and artifact contracts.

One process owns one device. Every optimizer step includes the same number of
bucket batches on every rank; incomplete rank groups are dropped, never repeated.
The service may launch this CLI, but merely selecting multiple visible devices is
not sufficient to activate it: torchrun's complete rank environment is required.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ypuddin.config.training_rules import distributed_training_errors
from ypuddin.data import BucketBatchSampler

from .events import Emitter, NullEmitter
from .trainer import StopRequested, Trainer
from .training_modes import FullTrainingSet


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str

    @classmethod
    def initialize(cls, device=None):
        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise ValueError(f"DDP requires torchrun environment; missing {', '.join(missing)}")
        rank, world, local = (int(os.environ[name]) for name in required[:3])
        if world < 2 or not 0 <= rank < world or local < 0:
            raise ValueError("invalid torchrun rank or world size")
        selected = torch.device(device) if device else Trainer._pick_device()
        if selected.type == "cuda":
            if selected.index is not None:
                raise ValueError("torchrun assigns LOCAL_RANK; use --device cuda without a device index")
            if not torch.cuda.is_available() or local >= torch.cuda.device_count():
                raise ValueError(f"LOCAL_RANK={local} has no visible CUDA/DTK device")
            selected = torch.device("cuda", local)
            torch.cuda.set_device(selected)
        elif selected.type != "cpu":
            raise ValueError(
                "DDP training supports CUDA/DTK or CPU; MPS multi-process training is unavailable"
            )
        backend = os.environ.get("YPUDDIN_DISTRIBUTED_BACKEND") or (
            "nccl" if selected.type == "cuda" else "gloo"
        )
        if backend not in {"nccl", "gloo"}:
            raise ValueError("YPUDDIN_DISTRIBUTED_BACKEND must be nccl or gloo")
        if selected.type == "cuda" and backend != "nccl":
            raise ValueError("CUDA/DTK DDP requires the vendor's NCCL-compatible process group")
        if backend == "nccl" and (selected.type != "cuda" or not dist.is_nccl_available()):
            raise ValueError("this PyTorch build has no usable NCCL-compatible CUDA/DTK process group")
        timeout = int(os.environ.get("YPUDDIN_DDP_TIMEOUT_SECONDS", "1800"))
        if timeout <= 0:
            raise ValueError("YPUDDIN_DDP_TIMEOUT_SECONDS must be positive")
        if dist.is_initialized():
            raise RuntimeError("training owns its process group; an existing process group cannot be reused")
        dist.init_process_group(backend, timeout=timedelta(seconds=timeout))
        return cls(rank, world, local, selected, backend)


class _TrainingGraph(nn.Module):
    """Register all trainable components while preserving native model/export names."""

    def __init__(self, trainer):
        super().__init__()
        self.backbone = trainer.loaded.backbone
        if isinstance(trainer.adapters, FullTrainingSet):
            self.components = nn.ModuleDict(trainer.adapters.modules)
        # Trainer is deliberately not a child Module: it owns caches, optimizer and I/O.
        object.__setattr__(self, "trainer", trainer)

    def forward(self, batch):
        return self.trainer.compute_loss(batch)[0]


class DistributedTrainer(Trainer):
    def __init__(self, cfg, *, context: DistributedContext, emitter=None):
        self.distributed = context
        self.is_primary = context.rank == 0
        self._checkpoint_rng = None
        super().__init__(cfg, device=context.device, emitter=emitter if self.is_primary else NullEmitter())

    def emit(self, type_: str, **data: Any) -> None:
        # Preparation is serialized. Control requests are coordinated only at
        # shared boundaries, not from rank-dependent cache progress callbacks.
        preparing, self._preparing = self._preparing, False
        try:
            super().emit(type_, **data)
        finally:
            self._preparing = preparing

    def _check_capabilities(self):
        super()._check_capabilities()
        unsupported = [f"{error['loc']}: {error['msg']}" for error in distributed_training_errors(self.cfg)]
        if unsupported:
            raise ValueError("DDP does not yet support: " + "; ".join(unsupported))
        if self.cfg.loop.gpu_count not in {1, self.distributed.world_size}:
            raise ValueError("loop.gpu_count differs from torchrun WORLD_SIZE")

    def prepare_data(self):
        # One writer fills shared caches first. Following ranks load and reuse
        # those entries in turn, avoiding manifest/fingerprint/cache temp races.
        for owner in range(self.distributed.world_size):
            error = [None]
            if self.distributed.rank == owner:
                try:
                    super().prepare_data()
                except Exception as exc:
                    error[0] = f"rank {owner} preparation failed: {type(exc).__name__}: {exc}"
                finally:
                    self._preparing = False
            dist.broadcast_object_list(error, src=owner)
            if error[0]:
                raise RuntimeError(error[0])
            request = self._control_request()
            if request in {"pause", "stop"}:
                raise StopRequested(request)

    def _sampler_options(self):
        return {"world_size": self.distributed.world_size, "rank": self.distributed.rank}

    def _prepare_training(self):
        super()._prepare_training()
        if self.progress.steps_per_epoch == 0:
            raise ValueError(
                "not enough bucket batches for all ranks; add images or reduce per-device batch size"
            )
        graph = _TrainingGraph(self)
        self.ddp = DistributedDataParallel(
            graph,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            output_device=self.device.index if self.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self._finish_distributed_preparation()

    def _finish_distributed_preparation(self):
        if not self.cfg.checkpoint.resume:
            seed = self.cfg.loop.seed + 100_003 * self.distributed.rank
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            self.gen.manual_seed(seed)
            self.loader_gen.manual_seed(seed + 1)
        unsharded = BucketBatchSampler(
            self.bundle.train.bucket_keys(), self.cfg.dataset.batch_size, seed=self.cfg.loop.seed
        ).plan()
        usable = len(unsharded) - len(unsharded) % self.distributed.world_size
        self.emit(
            "distributed.plan",
            strategy=self.cfg.loop.distributed_strategy,
            world_size=self.distributed.world_size,
            backend=self.distributed.backend,
            per_device_batch_size=self.cfg.dataset.batch_size,
            effective_batch_size=self.cfg.dataset.batch_size
            * self.cfg.loop.grad_accum
            * self.distributed.world_size,
            batches_per_rank=len(self.sampler.plan()),
            dropped_samples=sum(map(len, unsharded[usable:])),
            tail_policy="drop_incomplete_rank_group",
        )

    def _restore_checkpoint_rng(self, state):
        saved = state.get("distributed", {})
        if saved.get("world_size") != self.distributed.world_size:
            raise ValueError(
                "DDP exact resume requires the same saved world size; use weight-only warm start to change it"
            )
        if (
            saved.get("batch_size") != self.cfg.dataset.batch_size
            or saved.get("grad_accum") != self.cfg.loop.grad_accum
        ):
            raise ValueError(
                "DDP exact resume requires unchanged per-device batch size and gradient accumulation"
            )
        ranks = saved.get("ranks", [])
        if len(ranks) != self.distributed.world_size:
            raise ValueError("checkpoint is missing per-rank RNG state")
        self._restore_local_checkpoint_rng(ranks[self.distributed.rank])

    def _capture_checkpoint_rng(self):
        return self._checkpoint_rng

    def _primary_call(self, function, *args, **kwargs):
        payload = [None, None]
        if self.is_primary:
            try:
                payload[0] = function(*args, **kwargs)
            except Exception as exc:
                payload[1] = f"rank 0 operation failed: {type(exc).__name__}: {exc}"
        dist.broadcast_object_list(payload, src=0)
        if payload[1]:
            raise RuntimeError(payload[1])
        return payload[0]

    def save_state(self, tag=None):
        self.progress.extra["loss_ema"] = self._loss_ema
        local = self._capture_local_checkpoint_rng()
        states = [None] * self.distributed.world_size
        dist.all_gather_object(states, local)
        self._checkpoint_rng = {
            **states[0],
            "distributed": {
                "world_size": self.distributed.world_size,
                "batch_size": self.cfg.dataset.batch_size,
                "grad_accum": self.cfg.loop.grad_accum,
                "ranks": states,
            },
        }
        try:
            return self._primary_call(super().save_state, tag)
        finally:
            self._checkpoint_rng = None

    def save_weights(self, tag):
        return self._primary_call(super().save_weights, tag)

    def validate(self):
        return self._primary_call(super().validate)

    def sample_images(self, tag):
        return self._primary_call(super().sample_images, tag)

    def _control_request(self):
        request = super()._control_request() if self.is_primary else self._stop
        self._stop = None
        values = {None: 0, "save": 1, "pause": 2, "stop": 3}
        flag = torch.tensor(values[request], device=self.device, dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return (None, "save", "pause", "stop")[flag.item()]

    def _run_epoch(self):
        epoch = self.progress.epoch
        self.sampler.set_epoch(epoch)
        self.sampler.set_position(self.progress.batch_in_epoch)
        self.bundle.train.set_epoch(epoch)
        self.emit(
            "epoch.started",
            epoch=epoch,
            batches=len(self.sampler.plan()),
            position=self.progress.batch_in_epoch,
        )
        iterator = self._training_iterator()
        while group := list(islice(iterator, self.cfg.loop.grad_accum)):
            started = time.perf_counter()
            count, loss_sum, invalid = 0, 0.0, False
            for micro, batch in enumerate(group):
                self.progress.batch_in_epoch += 1
                local_count = len(batch["caption"])
                count += local_count
                with self._accumulation_context(micro < len(group) - 1):
                    loss = self._distributed_forward(batch)
                    finite = torch.tensor(int(torch.isfinite(loss).item()), device=self.device)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    invalid |= not bool(finite.item())
                    # Complete every reducer cycle even for rejected batches so
                    # the following forward cannot hang on unfinished reduction.
                    backward_loss = loss if finite.item() else torch.nan_to_num(loss) * 0
                    (backward_loss * local_count).backward()
                if finite.item():
                    loss_sum += loss.detach().item() * local_count
            totals = torch.tensor([count, loss_sum], dtype=torch.float64, device=self.device)
            dist.all_reduce(totals)
            total_count, total_loss = totals.tolist()
            self.progress.samples_seen += int(total_count)
            if invalid:
                self.optimizer.zero_grad(set_to_none=True)
                self.progress.nan_skips += 1
                self.emit(
                    "warning", code="loss.nonfinite", step=self.progress.step, skips=self.progress.nan_skips
                )
                if self.progress.nan_skips >= self.cfg.loop.nan_skip_limit:
                    raise RuntimeError("too many non-finite distributed losses")
                continue
            # DDP divides summed gradients by world size. Restore the actual
            # global image mean, including incomplete batches and accumulation.
            for parameter in self.adapters.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(self.distributed.world_size / total_count)
            self._optimizer_step(total_loss / total_count, time.perf_counter() - started)
            if self.progress.step >= self.progress.total_steps:
                break
        if self.progress.step >= self.progress.total_steps and self.progress.batch_in_epoch < len(
            self.sampler.plan()
        ):
            return
        self.progress.epoch += 1
        self.progress.batch_in_epoch = 0
        self.emit("epoch.finished", epoch=epoch, step=self.progress.step)
        self._epoch_hooks(epoch + 1)

    def _accumulation_context(self, accumulating):
        return self.ddp.no_sync() if accumulating else nullcontext()

    def _distributed_forward(self, batch):
        return self.ddp(batch)


def distributed_train(cfg, *, device=None, emitter=None, listeners=None):
    context = DistributedContext.initialize(device)
    try:
        if context.rank == 0:
            Path(cfg.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
            emitter = emitter or Emitter(
                path=cfg.logging.events_path or Path(cfg.checkpoint.output_dir) / "events.jsonl",
                fd=int(os.environ["YPUDDIN_EVENTS_FD"]) if os.environ.get("YPUDDIN_EVENTS_FD") else None,
            )
            for listener in listeners or []:
                emitter.add_listener(listener)
        trainer_type = DistributedTrainer
        if cfg.loop.distributed_strategy == "fsdp":
            from .sharded import ShardedTrainer

            trainer_type = ShardedTrainer
        return trainer_type(cfg, context=context, emitter=emitter).run()
    finally:
        # Never barrier on failure: the launcher terminates siblings and every
        # process must be able to release its group without waiting for a peer.
        if dist.is_initialized():
            dist.destroy_process_group()
