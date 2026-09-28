"""The training loop: prepare -> run, with exact pause/resume, validation, previews and events."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from functools import partial, wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ypuddin.adapters import AdapterSet, build_metadata, inject, save_adapter_file
from ypuddin.adapters.components import ComponentAdapterSet, inject_text_adapters
from ypuddin.config import TrainConfig, config_hash, write_config
from ypuddin.config.compute_policy import (
    BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
    DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID,
    DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID,
    DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID,
    DTK_ANIMA_LORA_SINGLE_FP16_POLICY_ID,
    DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS,
    DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
    DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
    DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID,
    DTK_SDXL_LONG_TEXT_POLICY_ID,
    DTK_TEXT_LORA_ALL_POLICY_IDS,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)
from ypuddin.data import (
    BucketBatchSampler,
    DataBundle,
    TextCache,
    build_data,
    build_text_cache,
    cache_latents,
    collate,
)
from ypuddin.data.dataset import training_layout_line
from ypuddin.data.native import NativeBatchSampler, collate_native
from ypuddin.memory import BlockSwapper
from ypuddin.models import LoadedModel, ModelFamily, TextCond, get_family
from ypuddin.optim import (
    build_optimizer,
    build_scheduler,
    is_schedule_free,
    load_optimizer_state,
    manages_learning_rate,
    optimizer_hyperparameter_snapshot,
    optimizer_learning_rates,
    optimizer_rate_snapshot,
    validate_optimizer_runtime,
)
from ypuddin.runtime_profiles import current_profile

from .events import Emitter, NullEmitter
from .logging import TrainingLogs, log_directory
from .reproducibility import (
    capture_compute_runtime,
    configure_reproducibility,
    validate_compute_runtime,
    validate_resume_reproducibility,
)
from .scheduler_contract import (
    read_resume_scheduler_contract,
    scheduler_recipe,
    validate_scheduler_instance,
    validate_scheduler_recipe,
)
from .state import Progress, capture_rng, load_checkpoint, restore_rng, save_checkpoint
from .training_modes import FullTrainingSet, save_model_artifact

log = logging.getLogger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "no": torch.float32}


class StopRequested(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind  # "pause" | "stop"


class _ControlWatch:
    """Logs a pause, stop or save request as soon as the service writes it.

    The loop still acts on the request between steps, where a resume point is exact; a step can take
    minutes, so the log says right away that the request arrived and when it takes effect.
    """

    def __init__(self, trainer: Trainer, directory: Path, interval: float = 0.5):
        self._trainer, self._directory, self._interval = trainer, directory, interval
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="control-watch", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)

    def _watch(self) -> None:
        seen: dict[str, tuple[int, int]] = {}
        while not self._closed.wait(self._interval):
            for name in ("stop", "pause", "save"):
                try:
                    stat = (self._directory / name).stat()
                except FileNotFoundError:
                    seen.pop(name, None)
                    continue
                except OSError:
                    continue
                key = (stat.st_mtime_ns, stat.st_ino)
                if seen.get(name) != key:
                    seen[name] = key
                    self._trainer._acknowledge_request(name)


def evaluation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._evaluation(unload_latent=method.__name__ == "sample_images"):
            return method(self, *args, **kwargs)

    return wrapped


class _ProgressLog:
    """Log a long loop when it starts, at most every 15 seconds while it runs, and when it ends."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.started = time.monotonic()
        self.last = 0.0

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if done == 0 or (done < total and now - self.last >= 15):
            self.last = now
            log.info("%s: %d/%d", self.label, done, total)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


class Trainer:
    is_primary = True
    # Log pacing and GPU identity; class defaults also serve trainers built without __init__ in tests.
    _gpu_identity: tuple[str | None, str] | None = None
    _gpu_sampler: Any = None
    _last_step_log = 0.0
    # The control request that is ending the run, and where the current epoch began.
    _stopping: str | None = None
    _stop_state: Path | None = None
    _pause_requested_at: float | None = None
    _epoch_mark: tuple[float, float, int, bool] | None = None
    # When the current step began and how long the last one took, for request acknowledgments; a
    # finished step whose previews and saves still run acts on requests before the next step.
    _hooks_pending = False
    _step_began: float | None = None
    _step_seconds: float | None = None

    def __init__(
        self, cfg: TrainConfig, *, device: str | torch.device | None = None, emitter: Emitter | None = None
    ):
        self.device = torch.device(device) if device else self._pick_device()
        self.cfg, self.compute_policy = resolve_training_compute_config(
            cfg, self.device.type, current_profile()
        )
        cfg = self.cfg
        self.compute_runtime: dict[str, Any] | None = None
        self.metal_attention_runtime: dict[str, str] | None = None
        self.run_dir = Path(cfg.checkpoint.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        events_path = cfg.logging.events_path or (
            (Path(cfg.logging.output_dir) if cfg.logging.output_dir else self.run_dir) / "events.jsonl"
        )
        self.emitter = emitter or Emitter(path=events_path)
        self.config_hash = config_hash(cfg)
        self.model_identity = ""
        self.progress = Progress()
        self.family: ModelFamily
        self.loaded: LoadedModel
        self.adapters: AdapterSet | ComponentAdapterSet | FullTrainingSet
        self.bundle: DataBundle
        self.objective: Any  # family-owned noising, prediction target and loss contract
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any
        self.grad_scaler: torch.amp.GradScaler | None = None
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
        self._debug_phase: tuple[str | None, float] = (None, 0.0)
        self._run_started = time.monotonic()

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
        if type_ == "warning" and data.get("message"):
            # Warnings belong in the job log as well as in the event stream.
            log.warning("%s", data["message"])
        if type_ == "checkpoint.saved":
            # Artifacts list the epoch and the loss of the step they were saved at.
            data.setdefault("epoch", self._epoch_now())
            data.setdefault("loss", self._step_loss())
            self._log_saved(data)
        elif type_ == "epoch.started":
            self._epoch_mark = (
                time.monotonic(),
                float(self.progress.extra.get("loss_sum", 0.0)),
                int(self.progress.extra.get("loss_count", 0)),
                bool(data.get("position")),
            )
        elif type_ == "epoch.finished":
            self._log_epoch(int(data.get("epoch", self.progress.epoch - 1)))
        self.emitter.emit(type_, **data)
        if log.isEnabledFor(logging.DEBUG):
            self._debug_event(type_, data)
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

    def _step_loss(self) -> float | None:
        """Training loss of the current step, when this step recorded a usable one (resume points may not)."""
        last = self.progress.extra.get("train_loss")
        if self.progress.step <= 0 or not isinstance(last, dict) or last.get("step") != self.progress.step:
            return None
        return _finite(last.get("loss"))

    def _point(self, step: int | None = None, epoch: float | None = None, loss: float | None = None) -> str:
        """Where training stands: step, epoch and loss, as logged for saves, pauses and resumes."""
        step = self.progress.step if step is None else step
        shown = _epoch_text(self._epoch_now() if epoch is None else epoch)
        value = _finite(loss)
        return f"step {step}/{self.progress.total_steps} | epoch {shown} | loss {'-' if value is None else f'{value:.4f}'}"

    def _log_saved(self, data: dict[str, Any]) -> None:
        path = Path(str(data.get("path")))
        point = self._point(data.get("step"), data.get("epoch"), data.get("loss"))
        if data.get("kind") == "full":
            saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if self._stopping:
                outcome = "paused" if self._stopping == "pause" else "stopped"
                duration = time.monotonic() - self._pause_requested_at if self._pause_requested_at else None
                suffix = f" | duration {duration:.3f}s" if outcome == "paused" and duration is not None else ""
                log.info("training %s; resume point saved %s | %s | %s%s", outcome, saved_at, point, path, suffix)
            else:
                log.info("saved resume point %s | %s | %s", saved_at, point, path)
            return
        kind = {"weights": "weights", "model": "model"}.get(data.get("kind"), "file")
        log.info("saved %s%s: %s | %s", kind, " (EMA)" if data.get("ema") else "", path.name, point)

    def _log_epoch(self, epoch: int) -> None:
        mark = self._epoch_mark
        if mark is None:
            log.info("epoch %d finished at step %d", epoch + 1, self.progress.step)
            return
        started, loss_sum, loss_count, resumed = mark
        count = int(self.progress.extra.get("loss_count", 0)) - loss_count
        mean = (float(self.progress.extra.get("loss_sum", 0.0)) - loss_sum) / count if count > 0 else None
        # The loss is the mean over this epoch's steps; after a mid-epoch resume, over those since resuming.
        log.info(
            "epoch %d finished at step %d | loss %s | %s%s",
            epoch + 1,
            self.progress.step,
            f"{mean:.4f}" if mean is not None and math.isfinite(mean) else "-",
            _duration(time.monotonic() - started, precise=True),
            " since resume" if resumed else "",
        )

    def _debug_event(self, type_: str, data: dict[str, Any]) -> None:
        """Trace lifecycle events; per-step metrics and per-item progress stay in events.jsonl."""
        if (
            type_ in _PER_STEP_EVENTS
            or (type_ == "cache.progress" and data.get("done") != data.get("total"))
            or (type_ == "warning" and data.get("message"))
        ):
            return
        if type_ == "phase.changed":
            now = time.perf_counter()
            previous, started = self._debug_phase
            self._debug_phase = (data.get("phase"), now)
            if previous is not None:
                log.debug("phase %s -> %s after %.1fs", previous, data.get("phase"), now - started)
                return
        log.debug("%s: %s", type_, _describe(data))

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
        resume_extra = self._read_resume_compute_metadata()
        self._preparing = True
        self._install_signal_handlers()
        self._seed_all()
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = cfg.memory.allow_tf32
            torch.backends.cudnn.allow_tf32 = cfg.memory.allow_tf32
        if self.compute_policy is not None:
            self.compute_runtime = capture_compute_runtime(self.device)
            if resume_extra is not None:
                validate_compute_runtime(self.compute_runtime, resume_extra.get("compute_runtime"))
        if self.is_primary:
            self._logs = TrainingLogs(cfg, self.run_dir)
            if not cfg.checkpoint.resume:
                write_config(cfg, self._record_dir() / "config.toml")
        self.emit(
            "run.started", config_hash=self.config_hash, device=str(self.device), run_dir=str(self.run_dir)
        )
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
        log.debug(
            "device %s: model dtype %s, compute dtype %s, mixed precision %s, attention %s, tf32 %s, "
            "deterministic %s, compute policy %s",
            self.device,
            str(model_dtype).removeprefix("torch."),
            str(self.compute_dtype).removeprefix("torch."),
            cfg.loop.mixed_precision,
            cfg.model.attention,
            cfg.memory.allow_tf32 if self.device.type == "cuda" else "n/a",
            cfg.loop.deterministic,
            (self.compute_policy or {}).get("id", "none"),
        )

        self.emit("phase.changed", phase="loading")
        from ypuddin.models.fingerprints import fingerprint_cache

        load_started = time.perf_counter()
        if cfg.checkpoint.resume:
            log.info("resume requested: restoring the run from %s", cfg.checkpoint.resume)
        log.info("loading %s model components", self.family.spec.name)

        cache_root = Path(cfg.dataset.cache_dir) if cfg.dataset.cache_dir else self.run_dir / "cache"
        with fingerprint_cache(cache_root / "fingerprints"):
            self.loaded = self.family.load(
                cfg.model, cfg.memory, device=self.device, dtype=model_dtype, backbone_device="cpu"
            )
            if (self.compute_policy or {}).get("frozen_text_implementation"):
                self.loaded.text.configure_compute(self.compute_policy["frozen_text_implementation"])
            self.model_identity = self._model_identity()
        log.info("model components loaded in %.1fs", time.perf_counter() - load_started)
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
        data_plan = self.bundle.plan
        log.info(
            "dataset: %d images (%d captioned), %d training items%s",
            data_plan.images,
            data_plan.captioned,
            data_plan.items,
            f", {data_plan.validation_images} validation images" if data_plan.validation_images else "",
        )
        log.info("%s", training_layout_line(cfg.dataset.resolution_mode, data_plan.buckets))

        if cfg.dataset.cache_latents:
            self.emit("phase.changed", phase="caching_latents")
            latent_log = _ProgressLog("VAE encoding")
            n = cache_latents(
                self.bundle,
                self.loaded.latent.encode,
                device=self.device,
                batch_size=1 if cfg.dataset.resolution_mode == "native" else max(1, cfg.dataset.batch_size),
                dtype=model_dtype,
                progress=lambda d, t: (
                    self.emit("cache.progress", kind="latents", done=d, total=t),
                    latent_log(d, t),
                ),
            )
            if n:
                log.info("cached %d latents in %.1fs", n, latent_log.elapsed)
            else:
                log.info("all latents were already cached")
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
            # DDPM-only fields must not change the identity of non-DDPM checkpoints.
            path_fields.update({"prediction_type", "zero_terminal_snr"})
        if self.family.spec.name not in {"flux", "flux2"}:
            path_fields.add("training_guidance")
        if self.family.spec.name != "flux2":
            path_fields.add("flux2_variant")
        if self.family.spec.name != "sdxl" or self.cfg.model.sdxl_max_token_length == 75:
            # The default 75-token context keeps the identity of checkpoints saved without this option.
            path_fields.add("sdxl_max_token_length")
        # Krea2 training always uses Raw; the inference-only variant field must not
        # change the identity of full-state checkpoints.
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
        if self.cfg.model.attention == "metal_flash":
            payload["metal_attention_runtime"] = self.metal_attention_runtime
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
            if cfg.training.train_backbone and cfg.adapter.preset not in presets:
                raise ValueError(
                    f"unknown adapter preset {cfg.adapter.preset!r} for {self.family.spec.name}; available: {sorted(presets)}"
                )
            base_precision = cfg.memory.base_precision if cfg.memory.base_precision != "auto" else "keep"
            self.loaded.backbone.requires_grad_(False)
            components = {}
            if cfg.training.train_backbone:
                components["backbone"] = inject(
                    self.loaded.backbone,
                    cfg.adapter,
                    presets[cfg.adapter.preset],
                    prefix=self.family.spec.adapter_prefix,
                    base_precision=base_precision,
                )
            if cfg.training.train_text_encoder:
                self.loaded.text.to(self.device)
                components.update(
                    inject_text_adapters(self.loaded.text.enable_adapter_training(), cfg.adapter)
                )
                self.adapters = ComponentAdapterSet(components)
            else:
                self.adapters = components["backbone"]
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
        self._validate_training_compute_policy()

        groups = self.adapters.param_groups(
            cfg.optimizer.lr, cfg.optimizer.weight_decay, cfg.optimizer.group_lr
        )
        self.optimizer = self._build_training_optimizer(groups)
        self._prepare_grad_scaler()
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
        captured_scheduler = getattr(self, "_resume_unsharded_scheduler_contract", None)
        if captured_scheduler is not None:
            validate_scheduler_recipe(captured_scheduler.contract, cfg, self.progress.total_steps)
        self.scheduler = (
            None
            if manages_learning_rate(cfg.optimizer)
            else build_scheduler(
                cfg.scheduler.model_copy(deep=True), self.optimizer, self.progress.total_steps
            )
        )
        self._scheduler_contract = scheduler_recipe(cfg, self.progress.total_steps)
        validate_scheduler_instance(self._scheduler_contract, self.scheduler)
        log.info(
            "optimizer %s: %d parameter groups, learning rates %s, weight decay %s",
            type(self.optimizer).__name__,
            len(self.optimizer.param_groups),
            ", ".join(f"{group.get('lr', 0):.3g}" for group in self.optimizer.param_groups),
            cfg.optimizer.weight_decay,
        )
        log.info(
            "scheduler %s: warmup %s, %d total steps",
            "managed by optimizer" if self.scheduler is None else cfg.scheduler.type,
            cfg.scheduler.warmup_steps,
            self.progress.total_steps,
        )
        log.debug(
            "data loader: %d batches per epoch, batch size %d, gradient accumulation %d, %d steps per epoch, "
            "%d workers, pin memory %s, %s resolution",
            batches,
            cfg.dataset.batch_size,
            cfg.loop.grad_accum,
            self.progress.steps_per_epoch,
            cfg.dataset.num_workers,
            self.device.type == "cuda",
            cfg.dataset.resolution_mode,
        )
        if cfg.loop.ema:
            self.ema = {
                k: v.detach().float().cpu().clone() for k, v in self.adapters.export_state()[0].items()
            }
        if cfg.checkpoint.resume:
            self._resume(cfg.checkpoint.resume)
            if self.is_primary:
                # A rejected same-directory legacy resume must preserve the
                # original config that authenticates its scheduler closure.
                write_config(cfg, self._record_dir() / "config.toml")
        if cfg.loop.distributed_strategy != "fsdp" or not hasattr(self, "distributed"):
            # DDP saves on rank zero but every rank must retain identical progress.
            self.progress.extra["scheduler_contract"] = deepcopy(self._scheduler_contract)
        self.progress.extra["deterministic"] = cfg.loop.deterministic
        if self.compute_policy is not None:
            self.progress.extra["compute_policy"] = dict(self.compute_policy)
            self.progress.extra["compute_runtime"] = self.compute_runtime
        if self.metal_attention_runtime is not None:
            self.progress.extra["metal_attention_runtime"] = dict(self.metal_attention_runtime)
        if getattr(self, "_text_adapter_operator_counts", None) is not None:
            self.progress.extra["text_adapter_operator_counts"] = self._text_adapter_operator_counts
        if getattr(self, "_fp16_adapter_operator_counts", None) is not None:
            self.progress.extra["fp16_adapter_operator_counts"] = self._fp16_adapter_operator_counts
        self._install_signal_handlers()
        self._prepared = True
        self.emit(
            "run.prepared",
            total_steps=self.progress.total_steps,
            steps_per_epoch=self.progress.steps_per_epoch,
            trainable_params=self.adapters.num_params(),
            text_mode=self.text_mode,
            deterministic=cfg.loop.deterministic,
            compute_policy=self.compute_policy,
        )

    def _place_training_model(self) -> None:
        """Place selected parameters before binding an optimizer to their final objects."""
        policy = getattr(self, "compute_policy", None) or {}
        if policy.get("id") == DTK_ANIMA_LORA_SINGLE_FP16_POLICY_ID:
            from .fp16_adapter_compute import install_fp16_adapter_compute

            if getattr(self, "_fp16_adapter_operator_counts", None) is not None:
                raise ValueError("FP16 算子计算策略不能重复安装")
            self._fp16_adapter_restore, self._fp16_adapter_operator_counts = install_fp16_adapter_compute(
                self.loaded.backbone
            )
            self._validate_training_compute_policy()
        if policy.get("id") in DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS:
            if policy["distributed_strategy"] == "fsdp":
                raise ValueError("BF16 分片计算策略必须由 FSDP 分片训练器安装")
            self._install_backbone_adapter_compute_operators()
        if (getattr(self, "compute_policy", None) or {}).get("id") in {
            DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID,
            DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
            DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID,
        }:
            raise ValueError("BF16 分片计算策略必须由 FSDP 分片训练器安装")
        if (getattr(self, "compute_policy", None) or {}).get("id") in DTK_TEXT_LORA_ALL_POLICY_IDS:
            self._install_text_adapter_compute_operators()
        if (getattr(self, "compute_policy", None) or {}).get("id") in {
            DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
            DTK_SDXL_LONG_TEXT_POLICY_ID,
        }:
            self._install_sdxl_compute_operators()
        if (getattr(self, "compute_policy", None) or {}).get("id") in {
            DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID,
            DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID,
        }:
            self._install_anima_compute_operators()
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

    def _text_compute_modules(self):
        if (getattr(self, "compute_policy", None) or {}).get("operator_components") == ["backbone"]:
            return {"backbone": self.loaded.backbone}
        return {"backbone": self.loaded.backbone, **self.loaded.text.trainable_modules()}

    def _install_backbone_adapter_compute_operators(self) -> None:
        if self.compute_policy["adapter_algorithm"] == "lora":
            self._install_text_adapter_compute_operators()
            return
        from .conv_forward import install_conv_fp32_forward
        from .linear_backward import (
            install_linear_bf16_forward_fp32_backward,
            install_linear_bf16_operands_fp32_compute,
        )

        if getattr(self, "_linear_backward_counts", None) is not None:
            raise ValueError("BF16 算子计算策略不能重复安装")
        install = (
            install_linear_bf16_operands_fp32_compute
            if self.compute_policy["linear_backward_implementation"]
            == BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
            else install_linear_bf16_forward_fp32_backward
        )
        restore, counts = install(self.loaded.backbone)
        try:
            if "conv_implementation" in self.compute_policy:
                self._conv_forward_restore, self._conv_forward_counts = install_conv_fp32_forward(
                    self.loaded.backbone
                )
        except BaseException:
            restore()
            raise
        self._linear_backward_restore, self._linear_backward_counts = restore, counts
        self._validate_training_compute_policy()

    def _install_text_adapter_compute_operators(self) -> None:
        from .text_adapter_compute import install_text_adapter_compute

        if getattr(self, "_text_adapter_operator_counts", None) is not None:
            raise ValueError("文本 LoRA 算子策略不能重复安装")
        self._text_adapter_compute_restore, self._text_adapter_operator_counts = install_text_adapter_compute(
            self._text_compute_modules(), self.compute_policy
        )
        self._validate_training_compute_policy()

    def _install_anima_compute_operators(self) -> None:
        """Use BF16-rounded operands and outputs around FP32 training Linear math."""
        from .linear_backward import install_linear_bf16_operands_fp32_compute

        if getattr(self, "_linear_backward_counts", None) is not None:
            raise ValueError("BF16 算子计算策略不能重复安装")
        self._linear_backward_restore, self._linear_backward_counts = (
            install_linear_bf16_operands_fp32_compute(self.loaded.backbone)
        )
        self._validate_training_compute_policy()

    def _install_sdxl_compute_operators(self) -> None:
        """Install the same operators before device movement or FSDP wrapping."""
        from .conv_forward import install_conv_fp32_forward
        from .linear_backward import (
            install_linear_bf16_forward_fp32_backward,
            install_linear_bf16_operands_fp32_compute,
        )

        if getattr(self, "_linear_backward_counts", None) is not None:
            raise ValueError("BF16 算子计算策略不能重复安装")
        installer = (
            install_linear_bf16_operands_fp32_compute
            if self.compute_policy["id"] == DTK_SDXL_LONG_TEXT_POLICY_ID
            else install_linear_bf16_forward_fp32_backward
        )
        restore, counts = installer(self.loaded.backbone)
        try:
            self._conv_forward_restore, self._conv_forward_counts = install_conv_fp32_forward(
                self.loaded.backbone
            )
        except Exception:
            restore()
            raise
        self._linear_backward_restore, self._linear_backward_counts = restore, counts
        self._validate_training_compute_policy()

    def _build_training_optimizer(self, groups):
        return build_optimizer(self.cfg.optimizer, groups)

    def _check_capabilities(self) -> None:
        caps = self.family.spec.capabilities
        cfg = self.cfg
        problems = []
        if cfg.model.attention == "metal_flash" and cfg.model.attention not in self.family.spec.attention_backends:
            problems.append("Metal FlashAttention is not supported by this model family")
        if cfg.model.attention == "metal_flash" and self.device.type != "mps":
            problems.append("Metal FlashAttention requires an Apple MPS device")
        if cfg.model.attention == "metal_flash" and cfg.memory.compile:
            problems.append("Metal FlashAttention does not support torch.compile")
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
        text_log = _ProgressLog("text encoding")
        n = build_text_cache(
            ordered,
            self.text_cache,
            self.loaded.text.encode_for_cache,
            self.loaded.text.fingerprint,
            progress=lambda d, t: (
                self.emit("cache.progress", kind="text", done=d, total=t),
                text_log(d, t),
            ),
            total=len(ordered),
        )
        if n:
            log.info(
                "cached %d text encodings (%d distinct captions) in %.1fs", n, len(ordered), text_log.elapsed
            )
        else:
            log.info("all text encodings were already cached")
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
    def _read_resume_compute_metadata(self) -> dict[str, Any] | None:
        """Reject numeric-policy changes before model loading or config overwrite.

        This is deliberately local/read-only. Distributed preparation serializes
        owners and broadcasts any error; adding a collective here would deadlock.
        """
        from .metal_compute import resolve_metal_attention_runtime, validate_metal_attention_resume

        self.metal_attention_runtime = resolve_metal_attention_runtime(
            self.cfg.model.attention, self.device.type
        )
        if not self.cfg.checkpoint.resume:
            return None
        path = Path(self.cfg.checkpoint.resume) / "state.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        extra = metadata.get("progress", {}).get("extra", {})
        validate_resume_reproducibility(self.cfg.loop.deterministic, extra.get("deterministic"))
        validate_resume_compute_policy(self.compute_policy, extra.get("compute_policy"))
        validate_metal_attention_resume(self.metal_attention_runtime, extra.get("metal_attention_runtime"))
        if metadata.get("strategy") != "fsdp2":
            self._resume_unsharded_scheduler_contract = read_resume_scheduler_contract(
                path.parent,
                metadata=metadata,
                captured=getattr(self, "_resume_unsharded_scheduler_contract", None),
            )
            validate_scheduler_recipe(self._resume_unsharded_scheduler_contract.contract, self.cfg)
        return extra

    def _prepare_grad_scaler(self) -> None:
        if self.device.type != "cuda" or self.compute_dtype != torch.float16:
            return
        if self.cfg.loop.gpu_count > 1:
            raise ValueError("FP16 动态梯度缩放暂只支持单卡；多卡请使用 BF16 或 FP32")
        if any(p.dtype == torch.float16 for g in self.optimizer.param_groups for p in g["params"]):
            raise ValueError("FP16 混合精度需要 FP32 或 BF16 可训练参数；请将适配器参数精度设为 FP32")
        self.grad_scaler = torch.amp.GradScaler("cuda")

    def _backward(self, loss: Tensor) -> None:
        scaler = getattr(self, "grad_scaler", None)
        (scaler.scale(loss) if scaler is not None else loss).backward()

    def _restore_grad_scaler(self, extra: dict) -> None:
        saved = extra.get("grad_scaler")
        scaler = getattr(self, "grad_scaler", None)
        if (scaler is not None) != bool(saved):
            raise ValueError(
                "检查点的 FP16 梯度缩放状态与当前训练不一致，不能精确恢复；旧检查点可仅加载权重开始新训练"
            )
        if scaler is not None:
            scaler.load_state_dict(saved)

    def _resume(self, path: str) -> None:
        log.debug("resuming training state from %s", path)
        self._validate_training_compute_policy()
        validate_scheduler_recipe(self._scheduler_contract, self.cfg, self.progress.total_steps)
        captured_scheduler = read_resume_scheduler_contract(
            path, captured=getattr(self, "_resume_unsharded_scheduler_contract", None)
        )
        validate_scheduler_recipe(captured_scheduler.contract, self.cfg, self.progress.total_steps)
        validate_scheduler_instance(captured_scheduler.contract, self.scheduler)
        ck = load_checkpoint(path)
        from .metal_compute import validate_metal_attention_resume

        validate_metal_attention_resume(
            self.metal_attention_runtime, ck["progress"].extra.get("metal_attention_runtime")
        )
        self._restore_grad_scaler(ck["progress"].extra)
        if bool(ck["scheduler"]) != (self.scheduler is not None):
            raise ValueError("保存的学习率调度器状态与原训练设置不一致，不能精确恢复")
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
        validate_resume_compute_policy(self.compute_policy, ck["progress"].extra.get("compute_policy"))
        if self.compute_policy is not None:
            validate_compute_runtime(
                capture_compute_runtime(self.device), ck["progress"].extra.get("compute_runtime")
            )
        if ck["config_hash"] and ck["config_hash"] not in self._resume_config_hashes():
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
        load_optimizer_state(self.cfg.optimizer, self.optimizer, ck["optimizer"])
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

    def _resume_config_hashes(self) -> set[str]:
        """The run's config hash with and without the resume path it was started with."""
        # A paused run is resumed from its own state: only checkpoint.resume differs.
        neutral = self.cfg.model_copy(deep=True)
        neutral.checkpoint.resume = None
        return {self.config_hash, config_hash(neutral)}

    def _adapter_metadata(self) -> dict[str, str]:
        if self.cfg.training.mode == "full":
            return {
                "ypuddin.training_mode": "full",
                "ypuddin.family": self.family.spec.name,
                "ypuddin.components": json.dumps(sorted(self.adapters.modules)),
                "ypuddin.config_hash": self.config_hash,
            }
        # Metadata must not gather or perform arithmetic on sharded parameters.
        targets = {
            name: layer.adapter.extra_metadata() | {"dora": layer.dora is not None, "mode": layer.mode}
            for name, layer in self.adapters.layers.items()
        }
        if isinstance(self.adapters, ComponentAdapterSet):
            targets = {
                item.export_key(name): layer.adapter.extra_metadata()
                | {"dora": layer.dora is not None, "mode": layer.mode}
                for item in self.adapters.components.values()
                for name, layer in item.layers.items()
            }
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
        if self.cfg.checkpoint.save_training_metadata:
            from ypuddin.adapters.recipe import training_recipe_metadata

            metadata.update(training_recipe_metadata(
                self.cfg, self.bundle, self.progress,
                world_size=getattr(getattr(self, "distributed", None), "world_size", 1),
            ))
        if isinstance(self.adapters, ComponentAdapterSet):
            metadata["ypuddin.components"] = json.dumps(sorted(self.adapters.components))
            metadata["ypuddin.component_prefixes"] = json.dumps(
                {name: item.prefix for name, item in self.adapters.components.items()}
            )
        if self.cfg.dataset.resolution_mode == "native":
            metadata["ypuddin.resolution_mode"] = "native"
            metadata["ypuddin.native_max_pixels"] = str(self.cfg.dataset.native_max_pixels)
            metadata["ypuddin.native_max_side"] = str(self.cfg.dataset.native_max_side)
        return metadata

    @evaluation
    def save_weights(self, tag: str) -> Path:
        started = time.perf_counter()
        path = self._save_weights(tag)
        log.debug("saved weights %s in %.1fs", tag, time.perf_counter() - started)
        return path

    def _save_weights(self, tag: str) -> Path:
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
        started = time.perf_counter()
        path = self._save_state(tag)
        log.debug("saved training state %s in %.1fs", path.name, time.perf_counter() - started)
        return path

    def _save_state(self, tag: str | None = None) -> Path:
        self._validate_training_compute_policy()
        validate_scheduler_recipe(self._scheduler_contract, self.cfg, self.progress.total_steps)
        validate_scheduler_instance(self._scheduler_contract, self.scheduler)
        self.progress.extra["scheduler_contract"] = deepcopy(self._scheduler_contract)
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
        if self.grad_scaler is not None:
            self.progress.extra["grad_scaler"] = self.grad_scaler.state_dict()
        # The sampler cursor is the iterator's starting point. DataLoader can
        # prefetch ahead, so only consumer progress identifies committed batches.
        sampler_state = {
            **self.sampler.state_dict(),
            "epoch": self.progress.epoch,
            "position": self.progress.batch_in_epoch,
        }
        path = save_checkpoint(
            (Path(self.cfg.checkpoint.state_dir) if self.cfg.checkpoint.state_dir else self.run_dir)
            / self._state_name(tag),
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

    def _state_name(self, tag: str | None = None) -> str:
        """Resume points are named by save time and step, so every save keeps its own folder."""
        name = f"state-{time.strftime('%Y%m%d-%H%M%S')}-step{self.progress.step:06d}"
        return f"{name}-{tag}" if tag else name

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
        if (getattr(self, "compute_policy", None) or {}).get("id") in DTK_TEXT_LORA_ALL_POLICY_IDS:
            # Online text runs before the backbone autocast context. Verify
            # actual encoder/adapter bindings before its first computation.
            self._validate_training_compute_policy()
            validate_compute_runtime(capture_compute_runtime(self.device), self.compute_runtime)
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
        self._validate_training_compute_policy()
        if self.compute_policy is not None:
            validate_compute_runtime(capture_compute_runtime(self.device), self.compute_runtime)
            if (
                self.cfg.loop.mixed_precision != self.compute_policy["mixed_precision"]
                or self.cfg.memory.allow_tf32 != self.compute_policy["allow_tf32"]
                or self.cfg.model.attention != self.compute_policy["attention"]
            ):
                raise ValueError("可复现训练的计算设置在运行中改变，已停止训练；请保持原设置后重试。")
        enabled = self.cfg.loop.mixed_precision != "no" and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=enabled)

    def _validate_training_compute_policy(self):
        metal_runtime = getattr(self, "metal_attention_runtime", None)
        if metal_runtime is not None:
            if self.cfg.model.attention != "metal_flash":
                raise ValueError("训练中的 Metal FlashAttention 设置发生变化，请保持原注意力后端")
            if self.cfg.model.family in {"sdxl", "flux2"}:
                from ypuddin.models.metal_attention import validate_metal_flash_processors

                validate_metal_flash_processors(self.loaded.backbone)
            elif getattr(self.loaded.backbone, "attn_mode", None) != "metal_flash":
                raise ValueError("Metal FlashAttention 主模型后端发生变化")
        policy = getattr(self, "compute_policy", None)
        if (policy or {}).get("attention_implementation"):
            from ypuddin.models.flux2.attention import validate_dtk_flash

            validate_dtk_flash(self.loaded.backbone)
        if (policy or {}).get("id") == DTK_ANIMA_LORA_SINGLE_FP16_POLICY_ID:
            from .fp16_adapter_compute import validate_fp16_adapter_compute

            _, expected = resolve_training_compute_config(self.cfg, self.device.type, current_profile())
            if (
                expected != policy
                or self.compute_dtype != torch.float16
                or self.cfg.memory.allow_tf32
                or self.cfg.model.attention != policy["attention"]
            ):
                raise ValueError("FP16 LoRA 计算策略与当前训练设置不一致")
            validate_fp16_adapter_compute(
                self.loaded.backbone, getattr(self, "_fp16_adapter_operator_counts", None)
            )
            return
        backbone_adapter_policy = (policy or {}).get("id") in DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS
        if (policy or {}).get("id") in DTK_TEXT_LORA_ALL_POLICY_IDS or (
            backbone_adapter_policy and policy["adapter_algorithm"] == "lora"
        ):
            from .text_adapter_compute import validate_text_adapter_compute

            _, expected = resolve_training_compute_config(self.cfg, self.device.type, current_profile())
            if (
                expected != policy
                or self.compute_dtype != torch.bfloat16
                or self.cfg.memory.allow_tf32
                or self.cfg.model.attention != policy["attention"]
            ):
                raise ValueError("文本 LoRA 计算策略与当前训练设置不一致")
            counts = getattr(self, "_text_adapter_operator_counts", None)
            if not counts:
                raise ValueError("文本 LoRA 计算策略未完整安装")
            validate_text_adapter_compute(self._text_compute_modules(), counts, policy)
            return
        if not backbone_adapter_policy and (policy or {}).get("id") not in {
            DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID,
            DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID,
            DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID,
            DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
            DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
            DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID,
            DTK_SDXL_LONG_TEXT_POLICY_ID,
        }:
            return
        from .linear_backward import validate_linear_backward_installation

        _, expected = resolve_training_compute_config(self.cfg, self.device.type, current_profile())
        if expected != policy or self.compute_dtype != torch.bfloat16:
            raise ValueError("BF16 Linear 反向计算策略与当前训练设置不一致")
        counts = getattr(self, "_linear_backward_counts", None)
        if not counts or not sum(counts.values()):
            raise ValueError("BF16 Linear 反向计算策略未完整安装到未量化的训练主干")
        if self.cfg.training.mode == "full" and (not counts.get("nn_linear") or counts.get("frozen_linear")):
            raise ValueError("BF16 Linear 全参策略需要未量化的训练主干")
        validate_linear_backward_installation(
            self.loaded.backbone,
            counts,
            expected_implementation=policy["linear_backward_implementation"],
        )
        if (
            backbone_adapter_policy
            or policy["id"]
            in {
                DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID,
                DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID,
            }
            and self.cfg.training.mode == "adapter"
        ):
            from .linear_backward import validate_lokr_bypass_backbone

            validate_lokr_bypass_backbone(self.loaded.backbone)
        if "conv_implementation" in policy:
            from .conv_forward import validate_conv_forward_installation

            validate_conv_forward_installation(
                self.loaded.backbone, getattr(self, "_conv_forward_counts", None)
            )
            if self.cfg.training.mode == "adapter":
                from .linear_backward import validate_lokr_bypass_backbone

                validate_lokr_bypass_backbone(self.loaded.backbone)

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
        # T-LoRA layers keep fewer ranks for noisier samples; t is each sample's noise level (0..1).
        set_noise_level = getattr(getattr(self, "adapters", None), "set_noise_level", None)
        if set_noise_level is not None:
            set_noise_level(t.to(self.device))
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
        with self._watching_control():
            return self._run()

    def _run(self) -> str:
        outcome = "finished"
        try:
            try:
                if not self._prepared:
                    self.prepare()
                cfg = self.cfg
                self.adapters.train(True)
                self.loaded.backbone.train()
                self._log_start()
                if (
                    cfg.sampling.enabled
                    and cfg.sampling.at_start
                    and self.progress.step == 0
                    and not self.progress.extra.get("initial_sample_done")
                ):
                    self.sample_images("initial")
                    self.progress.extra["initial_sample_done"] = True
                self.emit("phase.changed", phase="training")
                self._step_began = time.monotonic()
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
            if self._gpu_sampler is not None:
                self._gpu_sampler.close()
            self._close_logs(failed=outcome == "failed")
            self.emitter.close()

    def _primary_call(self, function, *args, **kwargs):
        """Run on the process that decides for all; one process decides for itself."""
        return function(*args, **kwargs)

    def _log_start(self) -> None:
        cfg = self.cfg
        distributed = getattr(self, "distributed", None)
        world = distributed.world_size if distributed is not None else 1
        effective = cfg.dataset.batch_size * cfg.loop.grad_accum * world
        if self.progress.step:
            log.info(
                "resumed training %s | %s | from %s",
                time.strftime("%Y-%m-%d %H:%M:%S"),
                self._point(loss=self._step_loss()),
                cfg.checkpoint.resume or "-",
            )
        else:
            log.info(
                "training %d steps: %d per epoch, effective batch %d (%d x %d accumulation x %d GPU)",
                self.progress.total_steps,
                self.progress.steps_per_epoch,
                effective,
                cfg.dataset.batch_size,
                cfg.loop.grad_accum,
                world,
            )
        if cfg.sampling.enabled:
            kept = self._stored_preview_seed() is not None
            if not cfg.sampling.seed and not kept:
                # Every rank keeps identical progress, so the first one picks the seed for all.
                self.progress.extra["preview_seed"] = self._primary_call(_random_preview_seed)
            seed = self._preview_seed()
            if cfg.sampling.seed:
                log.info("preview seed %d", seed)
            elif kept:
                log.info("preview seed %d (kept from the resume point)", seed)
            else:
                log.info("preview seed %d (random for this run)", seed)

    def _stored_preview_seed(self) -> int | None:
        stored = self.progress.extra.get("preview_seed")
        return stored if isinstance(stored, int) and not isinstance(stored, bool) and stored > 0 else None

    def _preview_seed(self) -> int:
        """sampling.seed 0 picks one random seed per run; resuming keeps it."""
        if self.cfg.sampling.seed:
            return self.cfg.sampling.seed
        seed = self._stored_preview_seed()
        if seed is None:
            seed = self.progress.extra["preview_seed"] = _random_preview_seed()
        return seed

    def _epoch_now(self) -> float | None:
        """Epochs completed at this step, fractional mid-epoch."""
        per_epoch = self.progress.steps_per_epoch
        return round(self.progress.step / per_epoch, 3) if per_epoch else None

    def _gpu_reading(self) -> dict[str, float]:
        """This run's GPU power, temperature and load from the driver; empty elsewhere."""
        if self.device.type == "mps":
            # Apple's sensors are slow to read; a background thread keeps the latest reading for each step.
            if self._gpu_sampler is None:
                from ypuddin.server.hardware import BackgroundReading, apple_gpu_reading

                self._gpu_sampler = BackgroundReading(apple_gpu_reading)
            return {f"gpu_{key}": value for key, value in self._gpu_sampler.latest().items()}
        if self.device.type != "cuda":
            return {}
        try:
            if self._gpu_identity is None:
                props = torch.cuda.get_device_properties(self.device)
                self._gpu_identity = (str(props.uuid) if getattr(props, "uuid", None) else None, props.name)
            from ypuddin.server.hardware import nvml_device_reading

            reading = nvml_device_reading(*self._gpu_identity)
        except Exception:  # noqa: BLE001
            return {}
        return {f"gpu_{key}": value for key, value in reading.items()}

    def _close_logs(self, *, failed: bool = False) -> None:
        if self._logs is not None:
            self._logs.close(failed=failed)
            self._logs = None

    def _record_dir(self) -> Path:
        """The run's log folder, which keeps its records; the output folder holds only products."""
        return log_directory(self.cfg, self.run_dir)

    def _control_dir(self) -> Path:
        # The service asks for pause / stop / save through files in the job's record folder.
        return Path(os.environ.get("YPUDDIN_CONTROL_DIR") or self.run_dir / "control")

    @contextmanager
    def _watching_control(self):
        """Acknowledge control requests while the run lasts; only the process that reads them watches."""
        watch = _ControlWatch(self, self._control_dir()) if self.is_primary else None
        try:
            yield
        finally:
            if watch is not None:
                watch.close()

    def _acknowledge_request(self, name: str) -> None:
        if self._stopping:
            return  # the loop is already acting on a request and logs it
        if self._preparing:
            if name in ("pause", "stop"):
                log.info("%s requested; stopping after the current preparation item", name)
            return
        total = self.progress.total_steps
        step = min(total, self.progress.step if self._hooks_pending else self.progress.step + 1)
        if name == "pause" and self._pause_requested_at is None:
            self._pause_requested_at = time.monotonic()
        about = ""
        if name != "pause" and not self._hooks_pending and self._step_seconds and self._step_began is not None:
            left = self._step_seconds - (time.monotonic() - self._step_began)
            about = f" (about {_duration(left)})" if left >= 1 else ""
        if name == "save":
            log.info("save requested; saving a resume point after step %d/%d%s", step, total, about)
        else:
            verb = "pausing" if name == "pause" else "stopping"
            log.info(
                "%s requested; %s after step %d/%d and saving a resume point%s",
                name,
                verb,
                step,
                total,
                about,
            )

    def _control_request(self) -> str | None:
        ctl = self._control_dir()
        for name in ("stop", "pause", "save"):
            f = ctl / name
            if f.exists():
                f.unlink()
                log.debug("control request %s at step %d", name, self.progress.step)
                return name
        req, self._stop = self._stop, None
        if req:
            log.debug("signal requested %s at step %d", req, self.progress.step)
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
            self._backward(loss / accum)
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
                self._backward(loss * (count / target))
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
        scaler = getattr(self, "grad_scaler", None)
        if scaler is not None:
            scaler.unscale_(self.optimizer)
        grad_norm = self._gradient_norm_and_clip()
        if not math.isfinite(grad_norm):
            if scaler is not None:
                old_scale = scaler.get_scale()
                # Explicitly back off even if the norm overflowed while each
                # individual gradient remained finite.
                scaler.update(new_scale=old_scale * scaler.get_backoff_factor())
                scale_state = scaler.state_dict()
                scale_state["_growth_tracker"] = 0
                scaler.load_state_dict(scale_state)
                self.emit(
                    "warning",
                    code="amp.overflow",
                    scale_before=old_scale,
                    scale_after=scaler.get_scale(),
                    step=self.progress.step,
                )
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
        if scaler is None:
            self.optimizer.step()
        else:
            scaler.step(self.optimizer)
            scaler.update()
        lrs = (
            optimizer_learning_rates(cfg.optimizer, self.optimizer, before_step=rate_snapshot)
            if will_log
            else None
        )
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.progress.step += 1
        self._hooks_pending, self._step_seconds = True, elapsed
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
                **self._gpu_reading(),
            )
            now = time.monotonic()
            if step == 1 or step == self.progress.total_steps or now - self._last_step_log >= 30:
                self._last_step_log = now
                rates = sorted({f"{value:.3g}" for value in (lrs or {}).values()})
                log.info(
                    "step %d/%d | epoch %s | loss %.4f | avg loss %.4f | lr %s | grad norm %.3g | %s it/s | eta %s",
                    step,
                    self.progress.total_steps,
                    f"{self._epoch_now():.2f}" if self._epoch_now() is not None else "-",
                    group_loss,
                    loss_mean,
                    "/".join(rates) or "-",
                    grad_norm,
                    f"{it_s:.3g}" if it_s else "-",
                    _duration(remaining * elapsed) if it_s else "-",
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
            self.save_state("manual")
        elif req in ("pause", "stop"):
            self._stopping = req
            if req == "pause" and self._pause_requested_at is None:
                self._pause_requested_at = time.monotonic()
            log.info(
                "%s at step %d/%d (epoch %s); saving a resume point",
                "pausing" if req == "pause" else "stopping",
                step,
                self.progress.total_steps,
                _epoch_text(self._epoch_now()),
            )
            self._stop_state = self.save_state("paused" if req == "pause" else "stopped")
            raise StopRequested(req)
        self._hooks_pending, self._step_began = False, time.monotonic()

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
        if (
            cfg.checkpoint.save_state_every_epochs
            and finished_epochs % cfg.checkpoint.save_state_every_epochs == 0
        ):
            self.save_state(f"epoch{finished_epochs:04d}")

    def _finish(self, outcome: str) -> None:
        self.emit("phase.changed", phase="finalizing")
        if outcome == "finished" and self._prepared and self.cfg.checkpoint.save_on_finish:
            self.save_weights("final")
        log.info(
            "training %s at step %d/%d after %s",
            outcome,
            self.progress.step,
            self.progress.total_steps,
            _duration(time.monotonic() - getattr(self, "_run_started", time.monotonic())),
        )
        self.emit(
            f"run.{outcome}",
            step=self.progress.step,
            epoch=self.progress.epoch,
            samples_seen=self.progress.samples_seen,
            preparing=not self._prepared,
            # The service resumes a paused job from the point it saved.
            **({"state_path": str(self._stop_state)} if self._stop_state and outcome != "finished" else {}),
        )

    # ----------------------------------------------------------------- validation & previews
    @torch.no_grad()
    @evaluation
    def validate(self) -> dict[str, float]:
        vcfg = self.cfg.validation
        ds = self.bundle.validation
        if ds is None:
            return {}
        started = time.perf_counter()
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
        log.debug("validation on %d images took %.1fs", len(ds.items), time.perf_counter() - started)
        return result

    @torch.no_grad()
    @evaluation
    def sample_images(self, tag: str) -> list[Path]:
        from PIL import Image

        from .preview_compute import preview_linear_compute

        scfg = self.cfg.sampling
        prompts = list(scfg.prompts)
        if scfg.prompts_file:
            prompts += _load_prompts_file(scfg.prompts_file)
        if not prompts:
            return []
        # Previews show what the exported file does: every T-LoRA rank.
        set_noise_level = getattr(getattr(self, "adapters", None), "set_noise_level", None)
        if set_noise_level is not None:
            set_noise_level(None)
        defaults = self.family.sampling_defaults(self.loaded)
        out_dir = (
            Path(self.cfg.sampling.output_dir) if self.cfg.sampling.output_dir else self.run_dir / "samples"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        loss = self._step_loss()
        stride = self.family.spec.latent.stride
        patch = self.family.spec.latent.patch
        started = time.perf_counter()
        base_seed = self._preview_seed()
        log.info("sampling %d previews at step %d (seed %d)", len(prompts), self.progress.step, base_seed)
        for i, p in enumerate(prompts):
            image_started = time.perf_counter()
            w = (p.width or scfg.width) // self.family.spec.latent.align * self.family.spec.latent.align
            h = (p.height or scfg.height) // self.family.spec.latent.align * self.family.spec.latent.align
            steps = p.steps or scfg.steps or defaults.steps
            cfg_scale = p.cfg if p.cfg is not None else (scfg.cfg if scfg.cfg is not None else defaults.cfg)
            guidance = scfg.guidance if scfg.guidance is not None else defaults.guidance
            shift = scfg.shift or self.family.sampling_shift_for_model(
                self.loaded, (h // stride // patch) * (w // stride // patch), self.cfg.objective, steps=steps
            )
            # A prompt without its own seed (or with 0) follows the run's preview seed.
            seed = p.seed if p.seed else base_seed + i
            cond = self._text_cond([p.prompt])
            uncond = self._text_cond([p.negative])
            shape = (1, self.family.spec.latent.channels, h // stride, w // stride)
            model_dtype = self.loaded.dtype if self.device.type != "cpu" else torch.float32

            def predict(
                x: Tensor, t: Tensor, c: TextCond = cond, dt: torch.dtype = model_dtype, g=guidance
            ) -> Tensor:
                with self._autocast(), preview_linear_compute(self.compute_policy, self.loaded.backbone):
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
                epoch=self._epoch_now(),
                negative=p.negative,
                sampler=scfg.sampler,
                scheduler=scfg.scheduler,
                steps=steps,
                cfg=cfg_scale,
                shift=shift,
                guidance=guidance,
                er_sde_order=scfg.er_sde_order,
                er_sde_s_noise=scfg.er_sde_s_noise,
            )
            log.info(
                "preview %d/%d saved: %s (%dx%d, seed %d, %.1fs)",
                i + 1,
                len(prompts),
                path.name,
                w,
                h,
                seed,
                time.perf_counter() - image_started,
            )
        log.info("previews finished in %.1fs", time.perf_counter() - started)
        return paths


def _epoch_text(epoch: float | None) -> str:
    if epoch is None:
        return "-"
    return str(round(epoch)) if abs(epoch - round(epoch)) < 0.005 else f"{epoch:.2f}"


def _random_preview_seed() -> int:
    return random.SystemRandom().randrange(1, 2**31 - 1)


def _finite(value: object) -> float | None:
    """A recorded number as a finite float; None when missing, boolean, non-finite or too large for a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _duration(seconds: float, precise: bool = False) -> str:
    """Compact duration for log lines: 70h28m, 5m31s, 12s; ``precise`` keeps tenths below ten seconds (3.4s)."""
    if precise and seconds < 10:
        return f"{max(seconds, 0.01):.2g}s"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _describe(data: dict[str, Any], width: int = 160) -> str:
    parts = []
    for key, value in data.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        parts.append(f"{key}={text if len(text) <= width else text[: width - 1] + '…'}")
    return " ".join(parts) or "-"


# Per-step metrics, per-image progress and epoch starts would bury the lifecycle trace.
_PER_STEP_EVENTS = {"step", "sample.progress", "xyz.progress", "epoch.started"}


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
        path=cfg.logging.events_path
        or (Path(cfg.logging.output_dir or cfg.checkpoint.output_dir) / "events.jsonl")
    )
    trainer = Trainer(cfg, device=device, emitter=em)
    failed = False
    try:
        with trainer._watching_control():
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
        path=cfg.logging.events_path
        or (Path(cfg.logging.output_dir or cfg.checkpoint.output_dir) / "events.jsonl"),
        fd=int(os.environ["YPUDDIN_EVENTS_FD"]) if os.environ.get("YPUDDIN_EVENTS_FD") else None,
    )
    for fn in listeners or []:
        em.add_listener(fn)
    trainer = Trainer(cfg, device=device, emitter=em)
    return trainer.run()


__all__ = ["Trainer", "train", "cache", "NullEmitter"]
