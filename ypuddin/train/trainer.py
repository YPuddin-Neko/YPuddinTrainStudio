"""The training loop: prepare -> run, with exact pause/resume, validation, previews and events."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import signal
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from functools import partial, wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ypuddin.adapters import AdapterSet, build_metadata, inject, save_adapter_file
from ypuddin.config import TrainConfig, config_hash, write_config
from ypuddin.data import (
    BucketBatchSampler,
    DataBundle,
    TextCache,
    build_data,
    build_text_cache,
    cache_latents,
    collate,
)
from ypuddin.data.native import NativeBatchSampler, collate_native
from ypuddin.memory import BlockSwapper
from ypuddin.models import LoadedModel, ModelFamily, TextCond, get_family
from ypuddin.optim import (
    build_optimizer,
    build_scheduler,
    is_schedule_free,
    manages_learning_rate,
    optimizer_hyperparameter_snapshot,
    optimizer_learning_rates,
    optimizer_rate_snapshot,
    validate_optimizer_runtime,
)
from ypuddin.runtime_profiles import current_profile

from .events import Emitter, NullEmitter
from .logging import TrainingLogs
from .reproducibility import configure_reproducibility, validate_resume_reproducibility
from .state import Progress, capture_rng, load_checkpoint, restore_rng, save_checkpoint
from .training_modes import FullTrainingSet, save_model_artifact

log = logging.getLogger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "no": torch.float32}


class StopRequested(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind  # "pause" | "stop"


def evaluation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._evaluation(unload_latent=method.__name__ == "sample_images"):
            return method(self, *args, **kwargs)

    return wrapped


class Trainer:
    is_primary = True

    def __init__(
        self, cfg: TrainConfig, *, device: str | torch.device | None = None, emitter: Emitter | None = None
    ):
        self.cfg = cfg
        self.device = torch.device(device) if device else self._pick_device()
        self.run_dir = Path(cfg.checkpoint.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        events_path = cfg.logging.events_path or (self.run_dir / "events.jsonl")
        self.emitter = emitter or Emitter(path=events_path)
        self.config_hash = config_hash(cfg)
        self.model_identity = ""
        self.progress = Progress()
        self.family: ModelFamily
        self.loaded: LoadedModel
        self.adapters: AdapterSet | FullTrainingSet
        self.bundle: DataBundle
        self.objective: Any  # family-owned noising, prediction target and loss contract
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any
        self.sampler: BucketBatchSampler
        self.loader: DataLoader
        self.text_cache: TextCache | None = None
        self.ema: dict[str, Tensor] | None = None
        self.swapper: BlockSwapper | None = None
        self.gen = torch.Generator().manual_seed(cfg.loop.seed)  # noise / timestep RNG (checkpointed)
        self.loader_gen = torch.Generator().manual_seed(cfg.loop.seed + 1)
        self._loader_epoch_state: dict[str, Any] | None = None
        self._stop: str | None = None
        self._loss_ema: float | None = None
        self._prepared = False
        self._preparing = False
        self._logs: TrainingLogs | None = None

    # ----------------------------------------------------------------- setup
    @staticmethod
    def _pick_device() -> torch.device:
        if current_profile().endswith("-cpu"):
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def compute_dtype(self) -> torch.dtype:
        if self.device.type == "mps":
            return torch.float32  # conservative MPS path: no autocast or unsupported low-precision kernels
        if self.device.type == "cpu" and self.cfg.loop.mixed_precision != "no":
            return torch.bfloat16 if self.cfg.loop.mixed_precision == "bf16" else torch.float32
        return DTYPES[self.cfg.loop.mixed_precision]

    def _seed_all(self) -> None:
        configure_reproducibility(self.cfg.loop.deterministic, self.device)
        s = self.cfg.loop.seed
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def emit(self, type_: str, **data: Any) -> None:
        if not self.is_primary:
            return
        self.emitter.emit(type_, **data)
        if self._logs is not None:
            try:
                self._logs.emit(type_, data)
            except Exception:
                log.exception("training log sink failed; continuing with JSONL events")
                self._logs.close(failed=True)
                self._logs = None
        if self._preparing and type_ in ("phase.changed", "cache.progress"):
            req = self._control_request()
            if req in ("pause", "stop"):
                raise StopRequested(req)
            if req == "save":
                self.emit(
                    "warning",
                    message="training state is not available during preparation; cached items are preserved",
                )

    def prepare(self) -> None:
        try:
            self.prepare_data()
            self._prepare_training()
        finally:
            self._preparing = False

    def prepare_data(self) -> None:
        """Load the model, index the dataset and fill the latent / text caches (what a cache job does)."""
        cfg = self.cfg
        # Reject incompatible/retired recipes before touching an existing run's files.
        self.family = get_family(cfg.model.family)
        self._check_capabilities()
        self._preparing = True
        self._install_signal_handlers()
        if self.is_primary:
            self._logs = TrainingLogs(cfg, self.run_dir)
        self._seed_all()
        if self.is_primary:
            write_config(cfg, self.run_dir / "config.toml")
        self.emit(
            "run.started", config_hash=self.config_hash, device=str(self.device), run_dir=str(self.run_dir)
        )
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = cfg.memory.allow_tf32
            torch.backends.cudnn.allow_tf32 = cfg.memory.allow_tf32
        model_dtype = (
            DTYPES[cfg.model.dtype]
            if self.device.type == "cuda"
            and not (
                cfg.training.mode == "full"
                and (cfg.training.train_backbone or cfg.memory.base_precision == "fp32")
            )
            else torch.float32
        )
        if self.device.type == "mps" and (cfg.model.dtype != "fp32" or cfg.loop.mixed_precision != "no"):
            self.emit(
                "warning", message="MPS training uses fp32 without autocast for numerical compatibility"
            )

        self.emit("phase.changed", phase="loading")
        from ypuddin.models.fingerprints import fingerprint_cache

        cache_root = Path(cfg.dataset.cache_dir) if cfg.dataset.cache_dir else self.run_dir / "cache"
        with fingerprint_cache(cache_root / "fingerprints"):
            self.loaded = self.family.load(
                cfg.model, cfg.memory, device=self.device, dtype=model_dtype, backbone_device="cpu"
            )
            self.model_identity = self._model_identity()
        self.objective = self.family.build_objective(self.loaded, cfg.objective)
        self.loaded.text.to(self.device)
        self.loaded.latent.to(self.device)

        self.emit("phase.changed", phase="indexing")
        self.bundle = build_data(
            cfg,
            replace(self.family.spec.latent, fingerprint=self.loaded.latent.fingerprint),
            cache_root=cache_root,
            progress=lambda k, d, t: self.emit("cache.progress", kind=k, done=d, total=t),
        )
        self.emit("data.plan", **self.bundle.plan.to_dict())

        if cfg.dataset.cache_latents:
            self.emit("phase.changed", phase="caching_latents")
            n = cache_latents(
                self.bundle,
                self.loaded.latent.encode,
                device=self.device,
                batch_size=1 if cfg.dataset.resolution_mode == "native" else max(1, cfg.dataset.batch_size),
                dtype=model_dtype,
                progress=lambda d, t: self.emit("cache.progress", kind="latents", done=d, total=t),
            )
            log.info("cached %d latents", n)
            self.loaded.latent.unload()

        self.text_mode = self._resolve_text_mode()
        if self.text_mode == "cached":
            self.emit("phase.changed", phase="caching_text")
            self._build_text_cache(cache_root)

    def _model_identity(self) -> str:
        """Bind full-state resume to actual model assets, independently of their absolute paths."""
        from ypuddin.adapters.io import sha256_of_tensors
        from ypuddin.models.fingerprints import content_fingerprint

        if self.cfg.model.dit_path:
            backbone = content_fingerprint([self.cfg.model.dit_path], namespace="training-backbone-v1")
        else:
            # The built-in toy has no file: its deterministic initial weights are the base model.
            backbone = sha256_of_tensors(self.loaded.backbone.state_dict())
        path_fields = {"dit_path", "text_encoder_path", "text_encoder_2_path", "vae_path", "tokenizer_path"}
        if self.family.spec.objective != "ddpm":
            # New SDXL-only defaults do not invalidate existing Anima/Krea/Toy checkpoints.
            path_fields.update({"prediction_type", "zero_terminal_snr"})
        if self.family.spec.name not in {"flux", "flux2"}:
            path_fields.add("training_guidance")
        if self.family.spec.name != "flux2":
            path_fields.add("flux2_variant")
        # Raw is the historical Krea2 training behavior; the new inference-only
        # variant field must not invalidate existing full-state checkpoints.
        path_fields.add("krea2_variant")
        payload = {
            "version": 1,
            "family": self.family.spec.name,
            "architecture": self.family.spec.architecture,
            "backbone": backbone,
            "latent": self.loaded.latent.fingerprint,
            "text": self.loaded.text.fingerprint,
            "dtype": str(self.loaded.dtype),
            "base_precision": self.cfg.memory.base_precision,
            "model": self.cfg.model.model_dump(mode="json", exclude=path_fields),
            "dit_config": {
                key: value
                for key, value in self.loaded.extra.get("dit_config", {}).items()
                if key not in {"_name_or_path", "_diffusers_version", "_use_default_values", "_commit_hash"}
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _prepare_training(self) -> None:
        cfg = self.cfg
        self.family.materialize_backbone(self.loaded)
        self.emit("phase.changed", phase="injecting")
        if cfg.training.mode == "full":
            from ypuddin.adapters.frozen import FrozenLinear

            if any(isinstance(layer, FrozenLinear) for layer in self.loaded.backbone.modules()):
                raise ValueError("Full fine-tuning requires unquantized base weights")
            self.loaded.backbone.requires_grad_(False)
            modules = {}
            if cfg.training.train_backbone:
                modules["backbone"] = self.loaded.backbone
                self.loaded.dtype = torch.float32
            if cfg.training.train_text_encoder:
                self.loaded.text.to(self.device)
                modules.update(self.loaded.text.enable_training())
            self.adapters = FullTrainingSet(modules)
            if cfg.training.resume_weights:
                self.adapters.load_weights(cfg.training.resume_weights, self.family.spec.name)
        else:
            presets = self.family.presets()
            if cfg.adapter.preset not in presets:
                raise ValueError(
                    f"unknown adapter preset {cfg.adapter.preset!r} for {self.family.spec.name}; available: {sorted(presets)}"
                )
            base_precision = cfg.memory.base_precision if cfg.memory.base_precision != "auto" else "keep"
            self.adapters = inject(
                self.loaded.backbone,
                cfg.adapter,
                presets[cfg.adapter.preset],
                prefix=self.family.spec.adapter_prefix,
                base_precision=base_precision,
            )
            if cfg.adapter.resume_weights:
                from ypuddin.adapters import load_adapter_file

                tensors, _ = load_adapter_file(cfg.adapter.resume_weights)
                missing = self.adapters.load_state(tensors, strict=False)
                matched = len(self.adapters.layers) - len(missing)
                if matched == 0:
                    raise ValueError(
                        "adapter.resume_weights matched no adapted layers; check the model family, "
                        "adapter targets and weight file format"
                    )
                if missing:
                    self.emit(
                        "warning",
                        code="adapter.partial_warm_start",
                        message=(
                            f"已加载 {matched}/{len(self.adapters.layers)} 个适配层；"
                            f"权重文件中缺少其余 {len(missing)} 层，这些层将从初始权重开始训练。"
                        ),
                        matched_layers=matched,
                        missing_layers=len(missing),
                    )
        self.emit("adapters.injected", **self.adapters.summary())
        self._place_training_model()

        groups = self.adapters.param_groups(
            cfg.optimizer.lr, cfg.optimizer.weight_decay, cfg.optimizer.group_lr
        )
        self.optimizer = self._build_training_optimizer(groups)
        if is_schedule_free(cfg.optimizer):
            self.optimizer.train()
        native = cfg.dataset.resolution_mode == "native"
        sampler_type = NativeBatchSampler if native else BucketBatchSampler
        self.sampler = sampler_type(
            self.bundle.train.bucket_keys(),
            cfg.dataset.batch_size,
            seed=cfg.loop.seed,
            **self._sampler_options(),
        )
        self.loader = DataLoader(
            self.bundle.train,
            batch_sampler=self.sampler,
            collate_fn=partial(collate_native, max_pixels=cfg.dataset.native_max_pixels)
            if native
            else collate,
            num_workers=cfg.dataset.num_workers,
            pin_memory=self.device.type == "cuda",
            generator=self.loader_gen,
        )
        batches = self.sampler.batches_per_epoch()
        self.progress.steps_per_epoch = math.ceil(batches / cfg.loop.grad_accum)
        by_epochs = (cfg.loop.epochs or 10**9) * self.progress.steps_per_epoch
        self.progress.total_steps = min(by_epochs, cfg.loop.max_steps or 10**9)
        self.scheduler = (
            None
            if manages_learning_rate(cfg.optimizer)
            else build_scheduler(cfg.scheduler, self.optimizer, self.progress.total_steps)
        )
        if cfg.loop.ema:
            self.ema = {
                k: v.detach().float().cpu().clone() for k, v in self.adapters.export_state()[0].items()
            }
        if cfg.checkpoint.resume:
            self._resume(cfg.checkpoint.resume)
        self.progress.extra["deterministic"] = cfg.loop.deterministic
        self._install_signal_handlers()
        self._prepared = True
        self.emit(
            "run.prepared",
            total_steps=self.progress.total_steps,
            steps_per_epoch=self.progress.steps_per_epoch,
            trainable_params=self.adapters.num_params(),
            text_mode=self.text_mode,
            deterministic=cfg.loop.deterministic,
        )

    def _place_training_model(self) -> None:
        """Place selected parameters before binding an optimizer to their final objects."""
        cfg = self.cfg
        if cfg.memory.blocks_to_swap > 0:
            blocks = self.family.memory_layout(self.loaded).blocks
            self.swapper = BlockSwapper(blocks, cfg.memory.blocks_to_swap, self.device)
            self.swapper.move_model_to_device(self.loaded.backbone)
            self.emit("memory.block_swap", **self.swapper.summary())
        else:
            self.loaded.backbone.to(self.device)
        if cfg.memory.compile:
            self.compile_blocks()

    def _build_training_optimizer(self, groups):
        return build_optimizer(self.cfg.optimizer, groups)

    def _check_capabilities(self) -> None:
        caps = self.family.spec.capabilities
        cfg = self.cfg
        problems = []
        if cfg.loop.gpu_count > 1 and not hasattr(self, "distributed"):
            problems.append("loop.gpu_count > 1 requires torchrun; refusing single-device execution")
        if cfg.memory.blocks_to_swap > 0 and "block_swap" not in caps:
            problems.append("memory.blocks_to_swap requires the block_swap capability")
        if cfg.memory.base_precision.startswith("fp8") and self.device.type != "cuda":
            problems.append("fp8 base precision requires CUDA")
        if (cfg.dataset.masked_loss or cfg.dataset.image_fit == "pad") and "masked_loss" not in caps:
            problems.append("masked loss (including image padding exclusion) is not supported by this family")
        if (
            cfg.dataset.text_encoding == "online"
            and "online_text" not in caps
            and not cfg.training.train_text_encoder
        ):
            problems.append(
                "dataset.text_encoding='online' is not supported by this family (its text encoder is too large to "
                "stay resident); use 'cached' or 'auto'"
            )
        if cfg.memory.activation_checkpointing != "none" and "activation_checkpointing" not in caps:
            problems.append("memory.activation_checkpointing is not supported by this family")
        if cfg.memory.compile and "compile" not in caps:
            problems.append("memory.compile is not supported by this family")
        if cfg.memory.compile and cfg.memory.blocks_to_swap > 0:
            problems.append(
                "memory.compile cannot be combined with memory.blocks_to_swap (swap hooks break the graph)"
            )
        problems += self.family.validate_config(cfg.model)
        problems += [f"{item['loc']}: {item['msg']}" for item in self.family.training_options_errors(cfg)]
        if problems:
            raise ValueError("; ".join(problems))

    def _sampler_options(self) -> dict[str, int]:
        return {}

    def compile_blocks(self, compile_fn: Callable[[torch.nn.Module], torch.nn.Module] | None = None) -> int:
        """Replace every transformer block by its ``torch.compile``d wrapper (after adapter injection).

        Adapters are referenced directly by the ``AdapterSet``, so export / checkpointing are unaffected by
        the ``_orig_mod`` wrapper. Only CUDA gets the real compiler: inductor has no MPS backend and CPU
        compilation costs more than it saves for the toy family.
        """
        if compile_fn is None:
            if self.device.type != "cuda":
                self.emit("warning", message="memory.compile is only applied on CUDA; running eagerly")
                return 0
            compile_fn = torch.compile
        blocks = self.family.memory_layout(self.loaded).blocks
        by_id = {id(m): n for n, m in self.loaded.backbone.named_modules()}
        n = 0
        for block in blocks:
            name = by_id.get(id(block))
            if name is None:
                continue
            parent_name, _, attr = name.rpartition(".")
            parent = self.loaded.backbone.get_submodule(parent_name) if parent_name else self.loaded.backbone
            setattr(parent, attr, compile_fn(block))
            n += 1
        return n

    def _resolve_text_mode(self) -> str:
        if self.cfg.training.train_text_encoder:
            return "online"
        mode = self.cfg.dataset.text_encoding
        if mode == "auto":
            return "online" if "online_text" in self.family.spec.capabilities else "cached"
        return mode

    def _build_text_cache(self, cache_root: Path) -> None:
        self.text_cache = TextCache(cache_root / "text")
        captions = {""}  # unconditional caption: caption dropout, CFG sampling
        for ds in (self.bundle.train, self.bundle.validation):
            if ds is not None:
                captions.update(ds.use_cached_captions())
        if self.cfg.sampling.enabled:
            prompts = list(self.cfg.sampling.prompts) + (
                _load_prompts_file(self.cfg.sampling.prompts_file) if self.cfg.sampling.prompts_file else []
            )
            for p in prompts:
                captions.update((p.prompt, p.negative))
        ordered = sorted(captions)
        n = build_text_cache(
            ordered,
            self.text_cache,
            self.loaded.text.encode_for_cache,
            self.loaded.text.fingerprint,
            progress=lambda d, t: self.emit("cache.progress", kind="text", done=d, total=t),
            total=len(ordered),
        )
        log.info("cached %d text encodings (%d distinct captions)", n, len(ordered))
        self.loaded.text.unload()

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):  # noqa: ANN001
            self._stop = "pause" if signum == signal.SIGINT else "stop"

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:  # not in main thread
                pass

    # ----------------------------------------------------------------- resume / checkpoint
    def _resume(self, path: str) -> None:
        ck = load_checkpoint(path)
        expected_kind = "full-model" if self.cfg.training.mode == "full" else "adapter"
        if ck.get("training_kind", "adapter") != expected_kind:
            raise ValueError(
                "checkpoint training mode differs: full-model weights and adapters are not interchangeable"
            )
        if ck["dataset_fingerprint"] and ck["dataset_fingerprint"] != self.bundle.plan.fingerprint:
            if ck["format"] == 1:
                raise ValueError(
                    "legacy checkpoint dataset fingerprint/order is incompatible with the current index; "
                    "start a new run using adapter.resume_weights with its adapter.safetensors"
                )
            raise ValueError("checkpoint was trained on a different dataset (fingerprint mismatch)")
        if ck["model_identity"]:
            if ck["model_identity"] != self.model_identity:
                raise ValueError(
                    "checkpoint model assets or model configuration differ from the current model; "
                    "exact resume requires the same backbone, latent encoder and text encoder"
                )
        else:
            self.emit(
                "warning",
                message="checkpoint has no model asset identity; the original backbone and encoders "
                "cannot be verified, so exact resume is not guaranteed",
            )
        validate_resume_reproducibility(
            self.cfg.loop.deterministic, ck["progress"].extra.get("deterministic")
        )
        if ck["config_hash"] and ck["config_hash"] != self.config_hash:
            log.warning("config changed since the checkpoint was written; resuming anyway")
        if "training" in ck:
            self.adapters.load_training_state(ck["training"])
        elif any(layer.adapter.scalar is not None for layer in self.adapters.layers.values()):
            raise ValueError(
                "legacy checkpoints did not save scalar training parameters; exact resume is unavailable. "
                "Use adapter.resume_weights with adapter.safetensors to start a new optimizer instead"
            )
        else:
            self.adapters.load_state(ck["adapter"])
            self.emit(
                "warning", message="resuming a legacy checkpoint; exact RNG compatibility is not guaranteed"
            )
        expected_optimizer_settings = optimizer_hyperparameter_snapshot(self.cfg.optimizer, self.optimizer)
        self.optimizer.load_state_dict(ck["optimizer"])
        validate_optimizer_runtime(
            self.cfg.optimizer, self.optimizer, expected_groups=expected_optimizer_settings
        )
        if self.scheduler is not None and ck["scheduler"]:
            self.scheduler.load_state_dict(ck["scheduler"])
        self.progress = ck["progress"]
        self.sampler.load_state_dict(ck["sampler"])
        self._restore_checkpoint_rng(ck["rng"])
        if "ema" in ck and self.ema is not None:
            self.ema = {k: v.float() for k, v in ck["ema"].items()}
        self._loss_ema = self.progress.extra.get("loss_ema")
        self.emit(
            "run.resumed",
            step=self.progress.step,
            epoch=self.progress.epoch,
            batch_in_epoch=self.progress.batch_in_epoch,
        )

    def _adapter_metadata(self) -> dict[str, str]:
        if self.cfg.training.mode == "full":
            return {
                "ypuddin.training_mode": "full",
                "ypuddin.family": self.family.spec.name,
                "ypuddin.components": json.dumps(sorted(self.adapters.modules)),
                "ypuddin.config_hash": self.config_hash,
            }
        _, targets = self.adapters.export_state()
        metadata = build_metadata(
            targets=targets,
            adapter_cfg=self.cfg.adapter.model_dump(mode="json"),
            family=self.family.spec.name,
            architecture=f"{self.family.spec.architecture}/{self.cfg.adapter.algo}",
            title=self.cfg.checkpoint.name,
            resolution=(
                None
                if self.cfg.dataset.resolution_mode == "native"
                else ",".join(str(r) for r in self.cfg.dataset.resolutions)
            ),
            config_hash=self.config_hash,
            dataset_fingerprint=self.bundle.plan.fingerprint,
            steps=self.progress.step,
            epoch=self.progress.epoch,
        )
        if self.cfg.dataset.resolution_mode == "native":
            metadata["ypuddin.resolution_mode"] = "native"
            metadata["ypuddin.native_max_pixels"] = str(self.cfg.dataset.native_max_pixels)
            metadata["ypuddin.native_max_side"] = str(self.cfg.dataset.native_max_side)
        return metadata

    @evaluation
    def save_weights(self, tag: str) -> Path:
        if isinstance(self.adapters, FullTrainingSet):
            path = save_model_artifact(
                self.run_dir / f"{self.cfg.checkpoint.name}-{tag}.model", self.adapters, self.cfg, self.loaded
            )
            self.emit("checkpoint.saved", kind="model", step=self.progress.step, path=str(path), ema=False)
            if self.ema is not None:
                ema_path = save_model_artifact(
                    self.run_dir / f"{self.cfg.checkpoint.name}-{tag}-ema.model",
                    self.adapters,
                    self.cfg,
                    self.loaded,
                    tensors=self.ema,
                )
                self.emit(
                    "checkpoint.saved", kind="model", step=self.progress.step, path=str(ema_path), ema=True
                )
            self._rotate_weights()
            return path
        tensors, _ = self.adapters.export_state()
        path = save_adapter_file(
            self.run_dir / f"{self.cfg.checkpoint.name}-{tag}.safetensors",
            tensors,
            self._adapter_metadata(),
            dtype=self.cfg.checkpoint.save_dtype,
        )
        self.emit("checkpoint.saved", kind="weights", step=self.progress.step, path=str(path), ema=False)
        if self.ema is not None:
            ema_path = save_adapter_file(
                self.run_dir / f"{self.cfg.checkpoint.name}-{tag}-ema.safetensors",
                self.ema,
                self._adapter_metadata(),
                dtype=self.cfg.checkpoint.save_dtype,
            )
            self.emit(
                "checkpoint.saved", kind="weights", step=self.progress.step, path=str(ema_path), ema=True
            )
        self._rotate_weights()
        return path

    def _rotate_weights(self) -> None:
        keep = self.cfg.checkpoint.keep_last_n
        if not keep:
            return
        prefix = f"{self.cfg.checkpoint.name}-step"
        groups: dict[int, list[Path]] = {}
        suffix = ".model" if self.cfg.training.mode == "full" else ".safetensors"
        for path in self.run_dir.glob(f"{prefix}*{suffix}"):
            step = path.stem.removeprefix(prefix).removesuffix("-ema")
            if step.isdecimal():
                groups.setdefault(int(step), []).append(path)
        for step in sorted(groups)[:-keep]:
            for path in groups[step]:
                if path.is_dir():
                    import shutil

                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

    def save_state(self, tag: str | None = None) -> Path:
        if self.cfg.training.mode == "full":
            # A full-state checkpoint needs the raw optimizer-point weights only.
            # Do not duplicate a complete model in host RAM or switch SF into eval.
            tensors = self.adapters.training_state_dict()
            training_tensors = tensors
        else:
            with self._evaluation():
                tensors, _ = self.adapters.export_state()
            training_tensors = self.adapters.training_state_dict()
        self.progress.extra["loss_ema"] = self._loss_ema
        # The sampler cursor is the iterator's starting point. DataLoader can
        # prefetch ahead, so only consumer progress identifies committed batches.
        sampler_state = {
            **self.sampler.state_dict(),
            "epoch": self.progress.epoch,
            "position": self.progress.batch_in_epoch,
        }
        path = save_checkpoint(
            self.run_dir / f"state-{tag or self.progress.step}",
            adapter_tensors=tensors,
            training_tensors=training_tensors,
            adapter_metadata=self._adapter_metadata(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler_state=sampler_state,
            progress=self.progress,
            rng=self._capture_checkpoint_rng(),
            ema_tensors=self.ema,
            config_hash=self.config_hash,
            dataset_fingerprint=self.bundle.plan.fingerprint,
            model_identity=self.model_identity,
            training_kind="full-model" if self.cfg.training.mode == "full" else "adapter",
        )
        self.emit("checkpoint.saved", kind="full", step=self.progress.step, path=str(path))
        return path

    def _capture_checkpoint_rng(self) -> dict[str, Any]:
        return self._capture_local_checkpoint_rng()

    def _capture_local_checkpoint_rng(self) -> dict[str, Any]:
        state = capture_rng({"main": self.gen, "loader": self.loader_gen}, device=self.device)
        if self._loader_epoch_state is not None:
            state["loader_iterator"] = self._loader_epoch_state
        return state

    def _restore_local_checkpoint_rng(self, state: dict[str, Any]) -> None:
        restore_rng(state, {"main": self.gen, "loader": self.loader_gen}, device=self.device)
        self._loader_epoch_state = state.get("loader_iterator")

    def _restore_checkpoint_rng(self, state: dict[str, Any]) -> None:
        if "distributed" in state:
            raise ValueError("distributed checkpoint requires the same torchrun world size for exact resume")
        self._restore_local_checkpoint_rng(state)

    def _training_iterator(self):
        """Recreate workers without advancing the saved epoch's seed stream twice.

        TrainDataset's augmentations depend on (seed, epoch, item), never on
        worker scheduling or prefetched samples. Reusing the iterator's original
        seed also keeps worker initialization identical after a mid-epoch resume.
        """
        epoch = self.progress.epoch
        current = self.loader_gen.get_state()
        saved = self._loader_epoch_state
        continuing = saved is not None and saved["epoch"] == epoch
        if continuing:
            self.loader_gen.set_state(saved["generator"])
        elif self.progress.batch_in_epoch == 0:
            self._loader_epoch_state = {"epoch": epoch, "generator": current}
        try:
            return iter(self.loader)
        finally:
            # Older checkpoints lack the pre-iterator state. Their per-item
            # augmentation is still reproducible; avoid an extra seed advance.
            # A fresh epoch consumes exactly one worker-base-seed draw.
            if continuing or self.progress.batch_in_epoch > 0:
                self.loader_gen.set_state(current)

    # ----------------------------------------------------------------- batch processing
    @contextmanager
    def _evaluation(self, *, unload_latent: bool = False):
        """Evaluate schedule-free parameters and restore both training weights and RNG on every exit."""
        rng = capture_rng({"main": self.gen, "loader": self.loader_gen}, device=self.device)
        mode = self.loaded.backbone.training
        swap_mode = self.swapper.forward_only if self.swapper is not None else False
        schedule_free = hasattr(self, "optimizer") and is_schedule_free(self.cfg.optimizer)
        raw = self.adapters.training_state_dict() if schedule_free else None
        try:
            if schedule_free:
                self.optimizer.eval()
            self.adapters.train(False)
            self.loaded.backbone.eval()
            if self.swapper is not None:
                self.swapper.set_forward_only(True)
            yield
        finally:
            if self.swapper is not None:
                self.swapper.release_all()
                self.swapper.set_forward_only(swap_mode)
            if unload_latent and self.cfg.dataset.cache_latents:
                self.loaded.latent.unload()
            if schedule_free:
                self.optimizer.train()
                # Mode switches can round low precision tensors; restore the exact pre-eval training values.
                self.adapters.load_training_state(raw)
            self.loaded.backbone.train(mode)
            self.adapters.train(mode)
            restore_rng(rng, {"main": self.gen, "loader": self.loader_gen}, device=self.device)

    def _text_cond(self, captions: list[str]) -> TextCond:
        if self.text_mode == "cached" and self.text_cache is not None:
            entries = []
            for c in captions:
                key = TextCache.key(c, self.loaded.text.fingerprint)
                if not self.text_cache.has(key):
                    raise RuntimeError(
                        f"text encoding missing from cache for caption {c[:80]!r}; the cache was built for a different caption transform"
                    )
                entries.append(self.text_cache.get(key))
            return self.loaded.text.cond_from_cache(entries, self.device)
        if not self.cfg.memory.offload_text_encoder or self.device.type == "cpu":
            return self.loaded.text.encode(captions, self.device)
        self.loaded.text.to(self.device)
        try:
            return self.loaded.text.encode(captions, self.device)
        finally:
            self.loaded.text.to("cpu")

    def _latents(self, batch: dict[str, Any]) -> Tensor:
        if "latents" in batch:
            return batch["latents"].to(self.device, torch.float32)
        with torch.no_grad():
            return self.loaded.latent.encode(batch["pixels"].to(self.device)).float()

    def _num_tokens(self, latents: Tensor) -> int:
        p = self.family.spec.latent.patch
        return (latents.shape[-2] // p) * (latents.shape[-1] // p)

    def _autocast(self):
        enabled = self.cfg.loop.mixed_precision != "no" and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=enabled)

    def compute_loss(
        self,
        batch: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
        t_override: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(loss, per_sample_unweighted, t)``."""
        gen = generator or self.gen
        x0 = self._latents(batch)
        cond = self._text_cond(batch["caption"])
        t = (
            t_override
            if t_override is not None
            else self.objective.sample_t(
                x0.shape[0], generator=gen, num_tokens=self._num_tokens(x0), device=self.device
            )
        )
        x_t, target, _ = self.objective.prepare(x0, t.to(self.device), generator=gen)
        mask = batch.get("mask")
        if mask is not None:
            # Includes mandatory whole-image validity even when user masks are disabled.
            # Area coverage excludes fully padded latent cells and weights boundary cells.
            mask = torch.nn.functional.interpolate(
                mask[:, None].to(self.device), size=x0.shape[-2:], mode="area"
            )[:, 0]
        x_in = x_t.to(self.loaded.dtype if self.device.type != "cpu" else torch.float32)
        if torch.is_grad_enabled() and (
            self.swapper is not None or self.cfg.memory.activation_checkpointing != "none"
        ):
            # block swap: every block needs an input grad so backward hooks fire in order;
            # offloaded checkpointing (custom autograd.Function) only records a graph when an *input* requires
            # grad -- adapter parameters alone are invisible to it
            x_in.requires_grad_(True)
        with self._autocast():
            pred = self.family.forward(
                self.loaded, x_in, t.to(self.device), cond, geometry=batch.get("geometry")
            )
        loss, per_sample = self.objective.loss(
            pred.float(), target, t, mask=mask, sample_weight=batch["weight"]
        )
        return loss, per_sample, t

    # ----------------------------------------------------------------- loop
    def run(self) -> str:
        outcome = "finished"
        try:
            try:
                if not self._prepared:
                    self.prepare()
                cfg = self.cfg
                self.adapters.train(True)
                self.loaded.backbone.train()
                if (
                    cfg.sampling.enabled
                    and cfg.sampling.at_start
                    and self.progress.step == 0
                    and not self.progress.extra.get("initial_sample_done")
                ):
                    self.sample_images("initial")
                    self.progress.extra["initial_sample_done"] = True
                self.emit("phase.changed", phase="training")
                while self.progress.step < self.progress.total_steps:
                    if cfg.loop.epochs is not None and self.progress.epoch >= cfg.loop.epochs:
                        break
                    self._run_epoch()
            except StopRequested as e:
                outcome = "paused" if e.kind == "pause" else "stopped"
            self._preparing = False
            self._finish(outcome)
            return outcome
        except Exception as e:
            outcome = "failed"
            self.emit("run.failed", error=f"{type(e).__name__}: {e}", step=self.progress.step)
            raise
        finally:
            self._preparing = False
            self._close_logs(failed=outcome == "failed")
            self.emitter.close()

    def _close_logs(self, *, failed: bool = False) -> None:
        if self._logs is not None:
            self._logs.close(failed=failed)
            self._logs = None

    def _control_request(self) -> str | None:
        ctl = self.run_dir / "control"
        for name in ("stop", "pause", "save"):
            f = ctl / name
            if f.exists():
                f.unlink()
                return name
        req, self._stop = self._stop, None
        return req

    def _run_epoch(self) -> None:
        if self.cfg.dataset.resolution_mode == "native":
            self._run_native_epoch()
            return
        cfg = self.cfg
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
        accum = cfg.loop.grad_accum
        micro = 0
        group_loss = 0.0
        t0 = time.perf_counter()
        for batch in self._training_iterator():
            self.progress.batch_in_epoch += 1
            self.progress.samples_seen += len(batch["caption"])
            loss, _, _ = self.compute_loss(batch)
            if not torch.isfinite(loss):
                self.progress.nan_skips += 1
                self.emit(
                    "warning", code="loss.nonfinite", step=self.progress.step, skips=self.progress.nan_skips
                )
                if self.progress.nan_skips >= cfg.loop.nan_skip_limit:
                    raise RuntimeError(f"{self.progress.nan_skips} consecutive non-finite losses")
                continue
            (loss / accum).backward()
            if self.swapper is not None:
                self.swapper.release_all()
            group_loss += loss.item() / accum
            micro += 1
            if micro < accum:
                continue
            self._optimizer_step(group_loss, time.perf_counter() - t0)
            micro, group_loss, t0 = 0, 0.0, time.perf_counter()
            if self.progress.step >= self.progress.total_steps:
                break
        else:
            if micro > 0:  # tail group: step on what we have, normalised to the real group size
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        if p.grad is not None:
                            p.grad.mul_(accum / micro)
                self._optimizer_step(group_loss * accum / micro, time.perf_counter() - t0)
        if self.progress.step >= self.progress.total_steps and self.progress.batch_in_epoch < len(
            self.sampler.plan()
        ):
            return
        self.progress.epoch += 1
        self.progress.batch_in_epoch = 0
        self.emit("epoch.finished", epoch=epoch, step=self.progress.step)
        self._epoch_hooks(epoch + 1)

    def _run_native_epoch(self) -> None:
        """A logical batch may contain many sizes; backward releases each group's graph.

        Normalise by the actual image count across accumulation, including tails.
        Thus a singleton shape does not outweigh a group containing several images.
        All model/loss/Mask/precision paths still go through compute_loss.
        """
        cfg = self.cfg
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
        target = cfg.loop.grad_accum * cfg.dataset.batch_size
        batches, images, loss_sum = 0, 0, 0.0
        t0 = time.perf_counter()

        def step() -> None:
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(target / images)
            self._optimizer_step(loss_sum / images, time.perf_counter() - t0)

        for batch in self._training_iterator():
            self.progress.batch_in_epoch += 1
            self.progress.samples_seen += len(batch["caption"])
            invalid = False
            for part in batch["microbatches"]:
                count = len(part["caption"])
                loss, _, _ = self.compute_loss(part)
                if not torch.isfinite(loss):
                    self.progress.nan_skips += 1
                    self.emit(
                        "warning",
                        code="loss.nonfinite",
                        step=self.progress.step,
                        skips=self.progress.nan_skips,
                    )
                    # A partially backpropagated logical batch must never be committed.
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.swapper is not None:
                        self.swapper.release_all()
                    if self.progress.nan_skips >= cfg.loop.nan_skip_limit:
                        raise RuntimeError(f"{self.progress.nan_skips} consecutive non-finite losses")
                    invalid = True
                    break
                (loss * (count / target)).backward()
                if self.swapper is not None:
                    self.swapper.release_all()
                loss_sum += loss.item() * count
                images += count
                del loss
            if invalid:
                batches, images, loss_sum, t0 = 0, 0, 0.0, time.perf_counter()
                continue
            batches += 1
            if batches < cfg.loop.grad_accum:
                continue
            step()
            batches, images, loss_sum, t0 = 0, 0, 0.0, time.perf_counter()
            if self.progress.step >= self.progress.total_steps:
                break
        else:
            if images:
                step()
        if self.progress.step >= self.progress.total_steps and self.progress.batch_in_epoch < len(
            self.sampler.plan()
        ):
            return
        self.progress.epoch += 1
        self.progress.batch_in_epoch = 0
        self.emit("epoch.finished", epoch=epoch, step=self.progress.step)
        self._epoch_hooks(epoch + 1)

    def _optimizer_step(self, group_loss: float, elapsed: float) -> None:
        cfg = self.cfg
        grad_norm = self._gradient_norm_and_clip()
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.progress.nan_skips += 1
            self.emit(
                "warning", code="grad.nonfinite", step=self.progress.step, skips=self.progress.nan_skips
            )
            if self.progress.nan_skips >= cfg.loop.nan_skip_limit:
                raise RuntimeError("too many non-finite gradients")
            return
        self.progress.nan_skips = 0
        next_step = self.progress.step + 1
        will_log = next_step % cfg.loop.log_every == 0 or next_step == self.progress.total_steps
        rate_snapshot = optimizer_rate_snapshot(self.optimizer) if will_log else None
        self.optimizer.step()
        lrs = (
            optimizer_learning_rates(cfg.optimizer, self.optimizer, before_step=rate_snapshot)
            if will_log
            else None
        )
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.progress.step += 1
        self._update_ema()
        self._loss_ema = group_loss if self._loss_ema is None else 0.98 * self._loss_ema + 0.02 * group_loss
        step = self.progress.step
        # Sampling may run between sparse log events. Keep the actual optimizer-step loss,
        # including its step identity, in checkpointed progress rather than reusing an EMA.
        self.progress.extra["train_loss"] = {"step": step, "loss": group_loss}
        if "loss_sum" not in self.progress.extra or "loss_count" not in self.progress.extra:
            self.progress.extra.update(
                loss_sum=0.0, loss_count=0, loss_mean_scope="since_resume" if step > 1 else "run"
            )
        self.progress.extra["loss_sum"] += group_loss
        self.progress.extra["loss_count"] += 1
        loss_count = self.progress.extra["loss_count"]
        loss_mean = self.progress.extra["loss_sum"] / loss_count
        if step % cfg.loop.log_every == 0 or step == self.progress.total_steps:
            remaining = self.progress.total_steps - step
            it_s = 1.0 / elapsed if elapsed > 0 else None
            self.emit(
                "step",
                step=step,
                epoch=self.progress.epoch,
                loss=group_loss,
                loss_ema=self._loss_ema,
                loss_mean=loss_mean,
                loss_count=loss_count,
                loss_mean_scope=self.progress.extra.get("loss_mean_scope", "run"),
                lr=lrs,
                grad_norm=grad_norm,
                it_s=it_s,
                eta_s=(remaining * elapsed) if it_s else None,
                vram_mb=(torch.cuda.max_memory_allocated(self.device) / 2**20)
                if self.device.type == "cuda"
                else (torch.mps.current_allocated_memory() / 2**20 if self.device.type == "mps" else None),
                vram_metric="peak_allocated"
                if self.device.type == "cuda"
                else ("current_allocated" if self.device.type == "mps" else None),
            )
        self._step_hooks(step)

    def _gradient_norm_and_clip(self) -> float:
        params = self.adapters.parameters()
        if self.cfg.optimizer.grad_clip_norm > 0:
            return torch.nn.utils.clip_grad_norm_(params, self.cfg.optimizer.grad_clip_norm).item()
        return math.sqrt(
            sum(float(p.grad.detach().float().pow(2).sum()) for p in params if p.grad is not None)
        )

    def _update_ema(self) -> None:
        if self.ema is None:
            return
        d = self.cfg.loop.ema_decay
        with self._evaluation():
            tensors, _ = self.adapters.export_state()
        for k, v in tensors.items():
            self.ema[k].mul_(d).add_(v.detach().float().cpu(), alpha=1 - d)

    def _step_hooks(self, step: int) -> None:
        cfg = self.cfg
        if cfg.validation.enabled and cfg.validation.every_steps and step % cfg.validation.every_steps == 0:
            self.validate()
        if cfg.sampling.enabled and cfg.sampling.every_steps and step % cfg.sampling.every_steps == 0:
            self.sample_images(tag=f"step{step}")
        if cfg.checkpoint.save_every_steps and step % cfg.checkpoint.save_every_steps == 0:
            self.save_weights(f"step{step:06d}")
        if cfg.checkpoint.save_state_every_steps and step % cfg.checkpoint.save_state_every_steps == 0:
            self.save_state()
        req = self._control_request()
        if req == "save":
            self.save_state()
        elif req in ("pause", "stop"):
            self.save_state("paused" if req == "pause" else "stopped")
            raise StopRequested(req)

    def _epoch_hooks(self, finished_epochs: int) -> None:
        cfg = self.cfg
        if (
            cfg.validation.enabled
            and cfg.validation.every_epochs
            and finished_epochs % cfg.validation.every_epochs == 0
        ):
            self.validate()
        if (
            cfg.sampling.enabled
            and cfg.sampling.every_epochs
            and finished_epochs % cfg.sampling.every_epochs == 0
        ):
            self.sample_images(tag=f"epoch{finished_epochs}")
        if cfg.checkpoint.save_every_epochs and finished_epochs % cfg.checkpoint.save_every_epochs == 0:
            self.save_weights(f"epoch{finished_epochs:04d}")

    def _finish(self, outcome: str) -> None:
        self.emit("phase.changed", phase="finalizing")
        if outcome == "finished" and self._prepared and self.cfg.checkpoint.save_on_finish:
            self.save_weights("final")
        self.emit(
            f"run.{outcome}",
            step=self.progress.step,
            epoch=self.progress.epoch,
            samples_seen=self.progress.samples_seen,
            preparing=not self._prepared,
        )

    # ----------------------------------------------------------------- validation & previews
    @torch.no_grad()
    @evaluation
    def validate(self) -> dict[str, float]:
        vcfg = self.cfg.validation
        ds = self.bundle.validation
        if ds is None:
            return {}
        ds.set_epoch(0)
        ts = self.objective.sampler.icdf(vcfg.timesteps)
        per_t: dict[float, list[float]] = {q: [] for q in vcfg.timesteps}
        by_bucket: dict[tuple[int, int], list[int]] = {}
        for i, it in enumerate(ds.items):
            by_bucket.setdefault(it.bucket.key, []).append(i)
        bs = max(1, self.cfg.dataset.batch_size)
        for indices in by_bucket.values():
            effective_bs = bs
            if self.cfg.dataset.resolution_mode == "native":
                effective_bs = min(
                    bs, max(1, self.cfg.dataset.native_max_pixels // ds.items[indices[0]].bucket.area)
                )
            for s in range(0, len(indices), effective_bs):
                batch = collate([ds[i] for i in indices[s : s + effective_bs]])
                for q, t_val in zip(vcfg.timesteps, ts.tolist(), strict=True):
                    gen = torch.Generator().manual_seed(
                        vcfg.seed * 100003 + int(q * 1000) + int(batch["index"][0])
                    )
                    t = torch.full((len(batch["caption"]),), float(t_val))
                    _, per_sample, _ = self.compute_loss(batch, generator=gen, t_override=t)
                    per_t[q].extend(per_sample.tolist())
        result = {str(q): float(np.mean(v)) for q, v in per_t.items() if v}
        mean = float(np.mean(list(result.values()))) if result else float("nan")
        self.emit("validation", step=self.progress.step, per_t=result, mean=mean)
        return result

    @torch.no_grad()
    @evaluation
    def sample_images(self, tag: str) -> list[Path]:
        from PIL import Image

        scfg = self.cfg.sampling
        prompts = list(scfg.prompts)
        if scfg.prompts_file:
            prompts += _load_prompts_file(scfg.prompts_file)
        if not prompts:
            return []
        defaults = self.family.sampling_defaults(self.loaded)
        out_dir = (
            Path(self.cfg.sampling.output_dir) if self.cfg.sampling.output_dir else self.run_dir / "samples"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        last_loss = self.progress.extra.get("train_loss")
        loss = last_loss.get("loss") if isinstance(last_loss, dict) else None
        if (
            self.progress.step <= 0
            or not isinstance(last_loss, dict)
            or last_loss.get("step") != self.progress.step
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
        ):
            loss = None
        if loss is not None:
            try:
                loss = float(loss)
            except OverflowError:
                loss = None
        if loss is not None and not math.isfinite(loss):
            loss = None
        stride = self.family.spec.latent.stride
        patch = self.family.spec.latent.patch
        for i, p in enumerate(prompts):
            w = (p.width or scfg.width) // self.family.spec.latent.align * self.family.spec.latent.align
            h = (p.height or scfg.height) // self.family.spec.latent.align * self.family.spec.latent.align
            steps = p.steps or scfg.steps or defaults.steps
            cfg_scale = p.cfg if p.cfg is not None else (scfg.cfg if scfg.cfg is not None else defaults.cfg)
            guidance = scfg.guidance if scfg.guidance is not None else defaults.guidance
            shift = scfg.shift or self.family.sampling_shift_for_model(
                self.loaded, (h // stride // patch) * (w // stride // patch), self.cfg.objective, steps=steps
            )
            seed = p.seed if p.seed is not None else scfg.seed + i
            cond = self._text_cond([p.prompt])
            uncond = self._text_cond([p.negative])
            shape = (1, self.family.spec.latent.channels, h // stride, w // stride)
            model_dtype = self.loaded.dtype if self.device.type != "cpu" else torch.float32

            def predict(
                x: Tensor, t: Tensor, c: TextCond = cond, dt: torch.dtype = model_dtype, g=guidance
            ) -> Tensor:
                with self._autocast():
                    return self.family.forward(
                        self.loaded, x.to(dt), t.to(self.device), c, inference=True, guidance=g
                    ).float()

            def predict_uncond(x: Tensor, t: Tensor, c: TextCond = uncond) -> Tensor:
                return predict(x, t, c)

            def on_step(done: int, n: int, idx: int = i, prompts_total: int = len(prompts)) -> None:
                self.emit(
                    "sample.progress",
                    step=self.progress.step,
                    prompt_index=idx,
                    prompts=prompts_total,
                    done=done,
                    total=n,
                )

            latents = self.family.sample_latents(
                self.loaded,
                predict,
                shape,
                sampler=scfg.sampler,
                scheduler=scfg.scheduler,
                er_sde_order=scfg.er_sde_order,
                er_sde_s_noise=scfg.er_sde_s_noise,
                steps=steps,
                shift=shift,
                cfg=cfg_scale,
                predict_uncond=predict_uncond,
                generator=torch.Generator().manual_seed(seed),
                device=self.device,
                dtype=torch.float32,
                on_step=on_step,
            )
            self.loaded.latent.to(self.device)
            pixels = self.loaded.latent.decode(latents).clamp(-1, 1)
            arr = ((pixels[0].permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5).round().astype("uint8")
            path = out_dir / f"{tag}_{i:02d}_{seed}.png"
            if self.is_primary:
                Image.fromarray(arr).save(path)
            paths.append(path)
            self.emit(
                "sample.saved",
                step=self.progress.step,
                prompt_index=i,
                prompt=p.prompt,
                seed=seed,
                path=str(path),
                width=w,
                height=h,
                loss=loss,
                sampler=scfg.sampler,
                scheduler=scfg.scheduler,
                steps=steps,
                cfg=cfg_scale,
                shift=shift,
                guidance=guidance,
                er_sde_order=scfg.er_sde_order,
                er_sde_s_noise=scfg.er_sde_s_noise,
            )
        return paths


def _load_prompts_file(path: str) -> list[Any]:
    from ypuddin.config import SamplePrompt

    p = Path(path)
    if p.suffix.lower() == ".toml":
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover
            import tomli as tomllib
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        return [SamplePrompt.model_validate(d) for d in data.get("prompt", [])]
    return [
        SamplePrompt(prompt=line.strip())
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def cache(cfg: TrainConfig, *, device: str | None = None, emitter: Emitter | None = None) -> str:
    """A cache-only job: same lifecycle events as ``train`` so the supervisor treats it uniformly."""
    Path(cfg.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
    em = emitter or Emitter(
        path=cfg.logging.events_path or (Path(cfg.checkpoint.output_dir) / "events.jsonl")
    )
    trainer = Trainer(cfg, device=device, emitter=em)
    failed = False
    try:
        trainer.prepare_data()
        trainer._preparing = False
        trainer.emit("phase.changed", phase="finalizing")
        trainer.emit(
            "run.finished", step=0, epoch=0, samples_seen=0, cache_only=True, **trainer.bundle.plan.to_dict()
        )
        return "finished"
    except StopRequested as e:
        trainer._preparing = False
        outcome = "paused" if e.kind == "pause" else "stopped"
        trainer.emit(f"run.{outcome}", step=0, epoch=0, samples_seen=0, preparing=True, cache_only=True)
        return outcome
    except Exception as e:  # noqa: BLE001
        failed = True
        trainer.emit("run.failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        trainer._preparing = False
        trainer._close_logs(failed=failed)
        em.close()


def train(
    cfg: TrainConfig,
    *,
    device: str | None = None,
    emitter: Emitter | None = None,
    listeners: list[Callable[[dict[str, Any]], None]] | None = None,
) -> str:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from .distributed import distributed_train

        return distributed_train(cfg, device=device, emitter=emitter, listeners=listeners)
    if cfg.loop.gpu_count > 1:
        raise ValueError("loop.gpu_count > 1 requires torchrun; refusing to silently train on one device")
    Path(cfg.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
    em = emitter or Emitter(
        path=cfg.logging.events_path or (Path(cfg.checkpoint.output_dir) / "events.jsonl"),
        fd=int(os.environ["YPUDDIN_EVENTS_FD"]) if os.environ.get("YPUDDIN_EVENTS_FD") else None,
    )
    for fn in listeners or []:
        em.add_listener(fn)
    trainer = Trainer(cfg, device=device, emitter=em)
    return trainer.run()


__all__ = ["Trainer", "train", "cache", "NullEmitter"]
