"""Explicit torchrun/DDP training through the existing family, loss and artifact contracts.

One process owns one device. Every optimizer step includes the same number of
bucket batches on every rank; incomplete rank groups are dropped, never repeated.
The service may launch this CLI, but merely selecting multiple visible devices is
not sufficient to activate it: torchrun's complete rank environment is required.
"""

from __future__ import annotations

import logging
import os
import random
import sys
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

from ypuddin.adapters.components import ComponentAdapterSet
from ypuddin.config.training_rules import distributed_training_errors
from ypuddin.data import BucketBatchSampler
from ypuddin.data.native import NativeBatchSampler

from .events import Emitter, NullEmitter
from .scheduler_contract import read_resume_scheduler_contract, validate_scheduler_recipe
from .trainer import StopRequested, Trainer
from .training_modes import FullTrainingSet

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str

    @classmethod
    def initialize(cls, device=None, *, strategy="ddp", required_dtypes=(), deterministic=False):
        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise ValueError(f"DDP requires torchrun environment; missing {', '.join(missing)}")
        rank, world, local = (int(os.environ[name]) for name in required[:3])
        if world < 2 or not 0 <= rank < world or local < 0:
            raise ValueError("invalid torchrun rank or world size")
        if (
            sys.platform == "win32"
            and deterministic
            and (device is None or torch.device(device).type == "cuda")
        ):
            # The communication probe runs a real GEMM before Trainer._seed_all.
            # cuBLAS must see this before the first CUDA use, including device selection.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
        windows_cuda = sys.platform == "win32" and selected.type == "cuda"
        if windows_cuda and strategy != "ddp":
            raise ValueError("Windows 原生多卡暂不支持显存分片，请使用 Linux CUDA／DTK")
        backend = os.environ.get("YPUDDIN_DISTRIBUTED_BACKEND") or (
            "nccl" if selected.type == "cuda" and not windows_cuda else "gloo"
        )
        if backend not in {"nccl", "gloo"}:
            raise ValueError("YPUDDIN_DISTRIBUTED_BACKEND must be nccl or gloo")
        if windows_cuda and backend != "gloo":
            raise ValueError("Windows 原生数据并行需要 Gloo 通信后端")
        if selected.type == "cuda" and backend != "nccl" and not (windows_cuda and backend == "gloo"):
            raise ValueError("CUDA/DTK DDP requires the vendor's NCCL-compatible process group")
        if backend == "nccl" and (selected.type != "cuda" or not dist.is_nccl_available()):
            raise ValueError("this PyTorch build has no usable NCCL-compatible CUDA/DTK process group")
        if backend == "gloo" and not dist.is_gloo_available():
            raise ValueError("当前 PyTorch 环境不包含 Gloo 多卡通信后端")
        timeout = int(os.environ.get("YPUDDIN_DDP_TIMEOUT_SECONDS", "1800"))
        if timeout <= 0:
            raise ValueError("YPUDDIN_DDP_TIMEOUT_SECONDS must be positive")
        if dist.is_initialized():
            raise RuntimeError("training owns its process group; an existing process group cannot be reused")
        if windows_cuda:
            # Bound TCPStore connection time independently of training collectives:
            # large model loads may legitimately exceed the short startup timeout.
            store, store_rank, store_world = next(dist.rendezvous("env://", timeout=timedelta(seconds=60)))
            if (store_rank, store_world) != (rank, world):
                raise RuntimeError("Windows Gloo rendezvous rank/world differs from torchrun")
            dist.init_process_group(
                backend, store=store, rank=rank, world_size=world, timeout=timedelta(seconds=timeout)
            )
        else:
            dist.init_process_group(backend, timeout=timedelta(seconds=timeout))
        if windows_cuda:
            from .gloo_probe import probe_windows_cuda_ddp

            try:
                probe_windows_cuda_ddp(selected, required_dtypes=required_dtypes)
            except BaseException:
                dist.destroy_process_group()
                raise
        return cls(rank, world, local, selected, backend)


class _TrainingGraph(nn.Module):
    """Register all trainable components while preserving native model/export names."""

    def __init__(self, trainer):
        super().__init__()
        self.backbone = trainer.loaded.backbone
        if isinstance(trainer.adapters, (FullTrainingSet, ComponentAdapterSet)):
            self.components = nn.ModuleDict(trainer.adapters.modules)
        # Trainer is deliberately not a child Module: it owns caches, optimizer and I/O.
        object.__setattr__(self, "trainer", trainer)

    def forward(self, batch):
        return self.trainer.compute_loss(batch)[0]


def _needs_unused_parameter_detection(trainer, graph: _TrainingGraph) -> bool:
    cfg = trainer.cfg
    if (
        cfg.model.family != "krea2"
        or cfg.training.mode != "adapter"
        or cfg.training.train_text_encoder
        or cfg.memory.compile
        or getattr(trainer, "compute_policy", None) is not None
    ):
        return True
    from ypuddin.adapters import AdaptedLinear, AdapterSet, DoRA, LoKr, LoRA
    from ypuddin.models.krea2.vendor.krea2_mmdit import SingleStreamDiT

    adapters, backbone = trainer.adapters, trainer.loaded.backbone
    if (
        type(backbone) is not SingleStreamDiT
        or type(adapters) is not AdapterSet
        or adapters.model is not backbone
        or not adapters.layers
    ):
        return True
    expected = set()
    modules = dict(backbone.named_modules())
    for name, layer in adapters.layers.items():
        if (
            type(layer) is not AdaptedLinear
            or type(layer.adapter) not in {LoRA, LoKr}
            or (layer.dora is not None and type(layer.dora) is not DoRA)
            or layer.module_dropout_p != 0
            or layer.multiplier == 0
            or modules.get(name) is not layer
        ):
            return True
        expected.update(id(p) for p in layer.adapter.parameters() if p.requires_grad)
        if layer.dora is not None:
            expected.update(id(p) for p in layer.dora.parameters() if p.requires_grad)
    # Krea's dense forward uses every adapted layer. Whole-layer dropout and
    # extra trainable components need discovery; ordinary/rank dropout keep the graph.
    actual = {id(p) for p in graph.parameters() if p.requires_grad}
    return not expected or actual != expected


class DistributedTrainer(Trainer):
    def __init__(self, cfg, *, context: DistributedContext, emitter=None):
        from .native_resolution import resolve_distributed_native_vram_config

        cfg = resolve_distributed_native_vram_config(
            cfg, device=context.device, rank=context.rank, world_size=context.world_size,
        )
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
        if self.cfg.checkpoint.resume and self.cfg.loop.distributed_strategy != "fsdp":
            # Capture on every rank before rank zero overwrites a same-directory
            # legacy config. Never place this collective inside the owner loop.
            error = None
            try:
                self._resume_unsharded_scheduler_contract = read_resume_scheduler_contract(
                    self.cfg.checkpoint.resume
                )
                validate_scheduler_recipe(self._resume_unsharded_scheduler_contract.contract, self.cfg)
            except Exception as exc:
                error = (
                    f"rank {self.distributed.rank} scheduler preflight failed: {type(exc).__name__}: {exc}"
                )
            errors = [None] * self.distributed.world_size
            dist.all_gather_object(errors, error)
            if any(errors):
                raise ValueError("; ".join(item for item in errors if item))
        # One writer fills shared caches first. Following ranks load and reuse
        # those entries in turn, avoiding manifest/fingerprint/cache temp races.
        for owner in range(self.distributed.world_size):
            error = [None]
            if self.distributed.rank == owner:
                try:
                    super().prepare_data()
                except Exception as exc:
                    log.exception("rank %s data preparation failed", owner)
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
            find_unused_parameters=_needs_unused_parameter_detection(self, graph),
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
        sampler_type = (
            NativeBatchSampler if self.cfg.dataset.resolution_mode == "native" else BucketBatchSampler
        )
        unsharded = sampler_type(
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
                parts = self._distributed_microbatches(batch)
                for part_index, (part, part_count) in enumerate(parts):
                    accumulating = micro < len(group) - 1 or part_index < len(parts) - 1
                    with self._accumulation_context(accumulating):
                        loss = self._distributed_forward(part)
                        finite = torch.tensor(int(torch.isfinite(loss).item()), device=self.device)
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        invalid |= not bool(finite.item())
                        # Complete every reducer cycle even for rejected batches so
                        # the following forward cannot hang on unfinished reduction.
                        backward_loss = loss if finite.item() else torch.nan_to_num(loss) * 0
                        (backward_loss * part_count).backward()
                    if finite.item():
                        loss_sum += loss.detach().item() * part_count
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

    def _distributed_microbatches(self, batch):
        """Keep collective order equal when ranks have different native shapes.

        FSDP gathers/reduces on every forward/backward, so a rank cannot simply
        stop after its last shape group. Missing groups run a zero-weight copy
        of the smallest local group. They contribute neither samples nor loss;
        the final normalization uses only real images across all ranks.
        """
        if self.cfg.dataset.resolution_mode != "native":
            return [(batch, len(batch["caption"]))]
        parts = batch["microbatches"]
        slots = torch.tensor(len(parts), device=self.device, dtype=torch.int64)
        dist.all_reduce(slots, op=dist.ReduceOp.MAX)
        result = [(part, len(part["caption"])) for part in parts]
        filler = min(parts, key=lambda part: len(part["caption"]) * part["bucket"][0] * part["bucket"][1])
        result.extend((filler, 0) for _ in range(slots.item() - len(parts)))
        return result

    def _accumulation_context(self, accumulating):
        return self.ddp.no_sync() if accumulating else nullcontext()

    def _distributed_forward(self, batch):
        return self.ddp(batch)


def distributed_train(cfg, *, device=None, emitter=None, listeners=None):
    from ypuddin.models.precision import model_load_precision

    context = DistributedContext.initialize(
        device,
        strategy=cfg.loop.distributed_strategy,
        deterministic=cfg.loop.deterministic,
        required_dtypes=tuple(
            dict.fromkeys(
                [
                    torch.float32,
                    {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
                        cfg.loop.mixed_precision
                    ],
                    {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
                        model_load_precision(cfg.model, "cuda")
                    ],
                    {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
                        cfg.adapter.param_dtype
                    ],
                ]
            )
        ),
    )
    try:
        if context.rank == 0:
            Path(cfg.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
            emitter = emitter or Emitter(
                path=cfg.logging.events_path
                or Path(cfg.logging.output_dir or cfg.checkpoint.output_dir) / "events.jsonl",
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
