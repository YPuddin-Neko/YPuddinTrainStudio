"""Full-model FSDP2 training with per-layer GPU residency and native exports.

One process computes on each GPU; model, gradients and optimizer states are
sharded. CPU-loaded weights are placed a group at a time, never as a full CUDA
model. Evaluation and checkpoint collectives must run on every rank.
"""

from __future__ import annotations

import inspect
import math
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from ypuddin.optim import optimizer_hyperparameter_snapshot, validate_optimizer_runtime
from ypuddin.optim.sharded import prepare_sharded_optimizer, sharded_optimizer_state_bytes

from .distributed import DistributedTrainer
from .sharded_state import export_sharded_model_artifact, load_sharded_checkpoint, save_sharded_checkpoint
from .trainer import Trainer


def parameter_shard(parameter, world_size):
    """Prefer rows; a single-row projection must shard columns instead.

    FSDP2 requires even splits for nonzero dimensions. Tiny scalar parameters
    are handled separately as explicitly synchronized replicas.
    """
    if parameter.ndim and parameter.shape[0] >= world_size:
        return Shard(0)
    for dimension, size in enumerate(parameter.shape[1:], start=1):
        if size >= world_size and size % world_size == 0:
            return Shard(dimension)
    raise ValueError(f"模型参数形状 {tuple(parameter.shape)} 无法分到 {world_size} 张显卡")


def sharding_groups(backbone, blocks):
    """Wrap native repeated blocks bottom-up, including text-fusion refiners."""
    selected = {id(block) for block in blocks}
    for module in backbone.modules():
        if isinstance(module, nn.ModuleList):
            selected.update(id(child) for child in module if next(child.parameters(), None) is not None)
    return [
        (name, module)
        for name, module in reversed(list(backbone.named_modules()))
        if name and id(module) in selected
    ]


class ShardedTrainer(DistributedTrainer):
    def _check_capabilities(self):
        super()._check_capabilities()
        if self.device.type != "cuda":
            raise ValueError("显存分片训练需要至少两张 CUDA 或 DTK 显卡")
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError as exc:
            raise ValueError("当前 PyTorch 缺少 FSDP2 显存分片支持，请使用支持 FSDP2 的版本") from exc
        if not {"shard_placement_fn", "ignored_params"} <= set(inspect.signature(fully_shard).parameters):
            raise ValueError("当前 PyTorch 的 FSDP2 接口不支持所需的参数分片方式")

    def _prepare_training(self):
        Trainer._prepare_training(self)
        if self.progress.steps_per_epoch == 0:
            raise ValueError("图片批次数不足以分配到所有显卡，请增加图片或减小每卡批大小")
        self._finish_distributed_preparation()

    def _place_training_model(self):
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        model = self.loaded.backbone
        world = self.distributed.world_size
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("data",))
        parameters = list(model.parameters())
        ignored = {parameter for parameter in parameters if parameter.numel() < world}
        # Validate every placement before the first FSDP mutation.
        for parameter in parameters:
            if parameter not in ignored:
                parameter_shard(parameter, world)
        for parameter in ignored:
            parameter.data = parameter.data.to(self.device)
            dist.broadcast(parameter.data, src=0)
        # FSDP replaces Parameters after moving each group. Retaining originals
        # here would accidentally retain every full GPU tensor during setup.
        del parameters
        if "parameter" in locals():
            del parameter
        groups = sharding_groups(model, self.family.memory_layout(self.loaded).blocks)
        self._fsdp_modules = [module for _, module in [*groups, ("", model)]]
        policy = MixedPrecisionPolicy(
            param_dtype=self.compute_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        for _, module in [*groups, ("", model)]:
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=True,
                shard_placement_fn=lambda parameter: parameter_shard(parameter, world),
                mp_policy=policy,
                ignored_params=ignored,
            )
            self.adapters.rebind_parameters()
        self.adapters.rebind_parameters()
        self._replicated_parameters = [p for p in self.adapters.parameters() if not isinstance(p, DTensor)]
        local_bytes = sum(
            (p.to_local() if isinstance(p, DTensor) else p).numel() * p.element_size()
            for p in self.adapters.parameters()
        )
        all_bytes = [None] * world
        dist.all_gather_object(all_bytes, local_bytes)
        self.emit(
            "distributed.sharded",
            strategy="fsdp2",
            world_size=world,
            groups=[name or "backbone" for name, _ in [*groups, ("", model)]],
            global_parameter_bytes=sum(p.numel() * p.element_size() for p in self.adapters.parameters()),
            local_parameter_bytes=all_bytes,
            replicated_parameter_count=sum(p.numel() for p in self._replicated_parameters),
            master_dtype="fp32",
            compute_dtype=str(self.compute_dtype),
        )

    def _build_training_optimizer(self, groups):
        return prepare_sharded_optimizer(super()._build_training_optimizer(groups))

    def _accumulation_context(self, accumulating):
        # Reduce-scatter every microbatch: no_sync would retain full gradients
        # and defeat the capacity goal of this mode.
        return nullcontext()

    def _distributed_forward(self, batch):
        return self.compute_loss(batch)[0]

    @contextmanager
    def _evaluation(self, *, unload_latent=False):
        with super()._evaluation(unload_latent=unload_latent):
            try:
                yield
            finally:
                # PyTorch 2.7 retains unsharded root parameters after no-grad
                # forwards even with reshard_after_forward=True. Restore the
                # optimizer's FP32 shards before saving or the next training step.
                for module in self._fsdp_modules:
                    module.reshard()

    def _gradient_norm_and_clip(self):
        squared = torch.zeros((), dtype=torch.float64, device=self.device)
        for parameter in self.adapters.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if isinstance(gradient, DTensor):
                local = gradient.to_local()
                squared.add_(torch.linalg.vector_norm(local).double().square())
            else:
                dist.all_reduce(gradient)
                gradient.div_(self.distributed.world_size)
                if self.is_primary:
                    squared.add_(torch.linalg.vector_norm(gradient).double().square())
        dist.all_reduce(squared)
        norm = math.sqrt(squared.item())
        limit = self.cfg.optimizer.grad_clip_norm
        if limit > 0 and math.isfinite(norm):
            scale = min(1.0, limit / (norm + 1e-6))
            for parameter in self.adapters.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        return norm

    def _optimizer_step(self, group_loss, elapsed):
        super()._optimizer_step(group_loss, elapsed)
        local = {
            "rank": self.distributed.rank,
            "optimizer_state_bytes": sharded_optimizer_state_bytes(self.optimizer)["local_state_bytes"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }
        if self.progress.step == 1 or self.progress.step == self.progress.total_steps:
            ranks = [None] * self.distributed.world_size
            dist.all_gather_object(ranks, local)
            self.emit("distributed.memory", step=self.progress.step, ranks=ranks)

    def _resume(self, path):
        expected = optimizer_hyperparameter_snapshot(self.cfg.optimizer, self.optimizer)
        saved = load_sharded_checkpoint(
            path,
            modules=self.adapters.modules,
            optimizer=self.optimizer,
            expected_world_size=self.distributed.world_size,
            expected_batch_size=self.cfg.dataset.batch_size,
            expected_grad_accum=self.cfg.loop.grad_accum,
            expected_dataset_fingerprint=self.bundle.plan.fingerprint,
            expected_model_identity=self.model_identity,
            expected_deterministic=self.cfg.loop.deterministic,
        )
        validate_optimizer_runtime(self.cfg.optimizer, self.optimizer, expected_groups=expected)
        if self.scheduler is not None and saved["scheduler"]:
            self.scheduler.load_state_dict(saved["scheduler"])
        self.progress = saved["progress"]
        self.sampler.load_state_dict(saved["sampler"])
        self._restore_checkpoint_rng(saved["rng"])
        self._loss_ema = self.progress.extra.get("loss_ema")
        self.emit(
            "run.resumed",
            step=self.progress.step,
            epoch=self.progress.epoch,
            batch_in_epoch=self.progress.batch_in_epoch,
        )

    def save_state(self, tag=None):
        self.progress.extra["loss_ema"] = self._loss_ema
        path = save_sharded_checkpoint(
            self.run_dir / f"state-{tag or self.progress.step}",
            modules=self.adapters.modules,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler_state={
                **self.sampler.state_dict(),
                "epoch": self.progress.epoch,
                "position": self.progress.batch_in_epoch,
            },
            progress=self.progress,
            rng=self._capture_local_checkpoint_rng(),
            batch_size=self.cfg.dataset.batch_size,
            grad_accum=self.cfg.loop.grad_accum,
            adapter_metadata=self._adapter_metadata(),
            config_hash=self.config_hash,
            dataset_fingerprint=self.bundle.plan.fingerprint,
            model_identity=self.model_identity,
        )
        self.emit("checkpoint.saved", kind="full", step=self.progress.step, path=str(path))
        return path

    def save_weights(self, tag):
        path = export_sharded_model_artifact(
            self.run_dir / f"{self.cfg.checkpoint.name}-{tag}.model", self.adapters, self.cfg, self.loaded
        )
        self.emit("checkpoint.saved", kind="model", step=self.progress.step, path=str(path), ema=False)
        self._primary_call(self._rotate_weights)
        return path

    def validate(self):
        return Trainer.validate(self)

    def sample_images(self, tag):
        return Trainer.sample_images(self, tag)
