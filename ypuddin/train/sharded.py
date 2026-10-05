"""Full-model FSDP2 training with per-layer GPU residency and native exports.

One process computes on each GPU; model, gradients and optimizer states are
sharded. CPU-loaded weights are placed a group at a time, never as a full CUDA
model. Evaluation and checkpoint collectives must run on every rank.
"""

from __future__ import annotations

import inspect
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from ypuddin.config.compute_policy import (
    DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID,
    DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS,
    DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
    DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID,
)
from ypuddin.optim import optimizer_hyperparameter_snapshot, validate_optimizer_runtime
from ypuddin.optim.sharded import prepare_sharded_optimizer, sharded_optimizer_state_bytes

from .distributed import DistributedTrainer
from .reproducibility import capture_compute_runtime
from .scheduler_contract import validate_scheduler_instance, validate_scheduler_recipe
from .sharded_state import (
    _collective_check,
    export_sharded_model_artifact,
    load_sharded_checkpoint,
    read_sharded_scheduler_contract,
    save_sharded_checkpoint,
)
from .trainer import Trainer
from .training_feature_contract import (
    STATE_KEY as TRAINING_FEATURE_STATE_KEY,
)
from .training_feature_contract import (
    training_feature_contract,
    validate_training_feature_resume,
)


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
    from ypuddin.adapters.base import AdapterModule
    from ypuddin.adapters.dora import DoRA
    from ypuddin.adapters.linear import AdaptedLayer

    selected = {id(block) for block in blocks}
    # Separate FP32 adapters from frozen BF16 groups, including on older FSDP2
    # versions that require a uniform storage dtype per communication group.
    selected.update(id(module) for module in backbone.modules() if isinstance(module, (AdapterModule, DoRA)))
    selected.update(
        id(module) for module in backbone.modules()
        if isinstance(module, AdaptedLayer) and module.dora is not None and module.dora.compute_mode == "comfyui"
    )
    for module in backbone.modules():
        if isinstance(module, nn.ModuleList):
            selected.update(id(child) for child in module if next(child.parameters(), None) is not None)
    return [
        (name, module)
        for name, module in reversed(list(backbone.named_modules()))
        if name and id(module) in selected
    ]


def exported_precision_modules(backbone):
    """Keep unquantized factors resident until their differentiable export casts."""
    from ypuddin.adapters.linear import AdaptedLayer

    return {
        id(module)
        for layer in backbone.modules()
        if isinstance(layer, AdaptedLayer) and layer.dora is not None and layer.dora.compute_mode == "comfyui"
        for module in (layer, layer.adapter, layer.dora)
    }


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

    def prepare_data(self):
        if self.cfg.checkpoint.resume:
            # DistributedTrainer prepares one owner at a time. Capture on all
            # ranks before entering that serialized loop; capability checks
            # inside the loop must never initiate a collective.
            self._resume_scheduler_contract = _collective_check(
                lambda: read_sharded_scheduler_contract(self.cfg.checkpoint.resume)
            )
        super().prepare_data()

    def _prepare_training(self):
        Trainer._prepare_training(self)
        if self.progress.steps_per_epoch == 0:
            raise ValueError("图片批次数不足以分配到所有显卡，请增加图片或减小每卡批大小")
        self._finish_distributed_preparation()

    def _place_training_model(self):
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        model = self.loaded.backbone
        if (getattr(self, "compute_policy", None) or {}).get("id") in DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS:
            if self.compute_policy["distributed_strategy"] != "fsdp":
                raise ValueError("FSDP 训练器需要显存分片的计算策略")
            self._install_backbone_adapter_compute_operators()
        elif (getattr(self, "compute_policy", None) or {}).get("id") == DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID:
            from .linear_backward import install_linear_bf16_forward_fp32_backward

            if getattr(self, "_linear_backward_counts", None) is not None:
                raise ValueError("BF16 Linear 反向计算策略不能重复安装")
            self._linear_backward_restore, self._linear_backward_counts = (
                install_linear_bf16_forward_fp32_backward(model)
            )
            self._validate_training_compute_policy()
        elif (getattr(self, "compute_policy", None) or {}).get(
            "id"
        ) == DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID:
            self._install_sdxl_compute_operators()
        elif (getattr(self, "compute_policy", None) or {}).get(
            "id"
        ) == DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID:
            self._install_anima_compute_operators()
        if self.cfg.training.mode == "adapter":
            from .sharded_adapters import shardable_frozen_weights

            if any(layer.adapter.kind not in {"lora", "lokr"} for layer in self.adapters.layers.values()):
                raise ValueError("显存分片的适配层规则只能选择 LoRA 或 LoKr")
            shardable_frozen_weights(model)
        world = self.distributed.world_size
        mesh = init_device_mesh(self.device.type, (world,), mesh_dim_names=("data",))
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
        export_modules = exported_precision_modules(model)
        export_policy = MixedPrecisionPolicy(
            param_dtype=None, reduce_dtype=torch.float32, cast_forward_inputs=False,
        )
        for _, module in [*groups, ("", model)]:
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=True,
                shard_placement_fn=lambda parameter: parameter_shard(parameter, world),
                mp_policy=export_policy if id(module) in export_modules else policy,
                ignored_params=ignored,
            )
            if hasattr(self.adapters, "rebind_parameters"):
                self.adapters.rebind_parameters()
        if hasattr(self.adapters, "rebind_parameters"):
            self.adapters.rebind_parameters()
        self._validate_training_compute_policy()
        self._replicated_parameters = [p for p in self.adapters.parameters() if not isinstance(p, DTensor)]
        trainable = self.adapters.parameters()
        local_bytes = sum(
            (p.to_local() if isinstance(p, DTensor) else p).numel() * p.element_size()
            for p in model.parameters()
        )
        all_bytes = [None] * world
        dist.all_gather_object(all_bytes, local_bytes)
        self.emit(
            "distributed.sharded",
            strategy="fsdp2",
            world_size=world,
            groups=[name or "backbone" for name, _ in [*groups, ("", model)]],
            global_parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
            trainable_parameter_bytes=sum(p.numel() * p.element_size() for p in trainable),
            local_parameter_bytes=all_bytes,
            replicated_parameter_count=sum(p.numel() for p in self._replicated_parameters),
            master_dtype=str(trainable[0].dtype),
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

    def _adapter_contract(self):
        if self.cfg.training.mode != "adapter":
            return None
        from ypuddin.adapters.dora_contract import canonical_adapter_contract

        return canonical_adapter_contract(self.cfg.adapter.model_dump(mode="json", exclude={"resume_weights"}))

    def _checkpoint_modules(self):
        if self.cfg.training.mode == "adapter":
            from .sharded_adapters import adapter_modules

            return adapter_modules(self.adapters)
        return self.adapters.modules

    def _resume(self, path):
        from ypuddin.adapters.dora_contract import compute_contract

        self._validate_training_compute_policy()
        expected = optimizer_hyperparameter_snapshot(self.cfg.optimizer, self.optimizer)
        saved = load_sharded_checkpoint(
            path,
            modules=self._checkpoint_modules(),
            optimizer=self.optimizer,
            expected_adapter_contract=self._adapter_contract(),
            expected_training_kind="adapter" if self.cfg.training.mode == "adapter" else "full-model",
            expected_world_size=self.distributed.world_size,
            expected_batch_size=self.cfg.dataset.batch_size,
            expected_grad_accum=self.cfg.loop.grad_accum,
            expected_dataset_fingerprint=self.bundle.plan.fingerprint,
            expected_model_identity=self.model_identity,
            expected_deterministic=self.cfg.loop.deterministic,
            expected_compute_policy=self.compute_policy,
            expected_compute_runtime=capture_compute_runtime(self.device) if self.compute_policy else None,
            expected_scheduler_config=self.cfg.scheduler.model_dump(mode="json"),
            expected_total_steps=self.progress.total_steps,
            legacy_scheduler_contract=getattr(self, "_resume_scheduler_contract", None),
            expected_dora_contract=compute_contract(self.adapters),
            expected_training_features=training_feature_contract(self.cfg),
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
        from ypuddin.adapters.dora_contract import STATE_KEY, compute_contract

        def validate_save_contract():
            self._validate_dora_contract()
            self._validate_training_compute_policy()
            validate_training_feature_resume(self.cfg, self._training_feature_contract)
            validate_scheduler_recipe(self._scheduler_contract, self.cfg, self.progress.total_steps)
            validate_scheduler_instance(self._scheduler_contract, self.scheduler)
            return self._scheduler_contract["config"]

        # One rank may have changed its configuration or scheduler instance.
        # All ranks must reject before any tensor-gather or checkpoint write.
        scheduler_config = _collective_check(validate_save_contract)
        self.progress.extra["loss_ema"] = self._loss_ema
        self.progress.extra[STATE_KEY] = compute_contract(self.adapters)
        self.progress.extra[TRAINING_FEATURE_STATE_KEY] = training_feature_contract(self.cfg)
        path = save_sharded_checkpoint(
            (Path(self.cfg.checkpoint.state_dir) if self.cfg.checkpoint.state_dir else self.run_dir)
            # Every rank writes to the name rank zero picks.
            / self._primary_call(self._state_name, tag),
            modules=self._checkpoint_modules(),
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
            adapter_contract=self._adapter_contract(),
            training_kind="adapter" if self.cfg.training.mode == "adapter" else "full-model",
            adapter_metadata=self._adapter_metadata(),
            config_hash=self.config_hash,
            dataset_fingerprint=self.bundle.plan.fingerprint,
            model_identity=self.model_identity,
            scheduler_config=scheduler_config,
        )
        self.emit("checkpoint.saved", kind="full", step=self.progress.step, path=str(path))
        return path

    def save_weights(self, tag):
        _collective_check(self._validate_dora_contract)
        if self.cfg.training.mode == "adapter":
            from ypuddin.adapters import save_adapter_file

            from .sharded_adapters import gathered_adapter_export

            tensors = gathered_adapter_export(self.adapters)
            path = self.run_dir / f"{self.cfg.checkpoint.name}-{tag}.safetensors"
            self._primary_call(
                lambda: save_adapter_file(
                    path, tensors, self._adapter_metadata(), dtype=self.cfg.checkpoint.save_dtype,
                    include_hash=self.cfg.checkpoint.save_training_metadata,
                )
            )
            self.emit("checkpoint.saved", kind="weights", step=self.progress.step, path=str(path), ema=False)
            self._primary_call(self._rotate_weights)
            return path
        metadata = _collective_check(lambda: self._adapter_metadata() if self.is_primary else None)
        path = export_sharded_model_artifact(
            self.run_dir / f"{self.cfg.checkpoint.name}-{tag}.model", self.adapters, self.cfg, self.loaded,
            metadata=metadata,
        )
        self.emit("checkpoint.saved", kind="model", step=self.progress.step, path=str(path), ema=False)
        self._primary_call(self._rotate_weights)
        return path

    def validate(self):
        return Trainer.validate(self)

    def sample_images(self, tag):
        return Trainer.sample_images(self, tag)

    def _preview_adapter_export(self):
        from .sharded_adapters import gathered_adapter_export

        return gathered_adapter_export(self.adapters)
