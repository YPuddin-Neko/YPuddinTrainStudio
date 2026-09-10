"""The training loop: prepare -> run, with exact pause/resume, validation, previews and events."""

from __future__ import annotations

import logging
import math
import os
import random
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ypuddin.adapters import AdapterSet, build_metadata, inject, save_adapter_file
from ypuddin.config import TrainConfig, config_hash, write_config
from ypuddin.data import BucketBatchSampler, DataBundle, TextCache, build_data, build_text_cache, cache_latents, collate
from ypuddin.memory import BlockSwapper
from ypuddin.models import LoadedModel, ModelFamily, TextCond, get_family
from ypuddin.objectives import Objective
from ypuddin.optim import build_optimizer, build_scheduler, is_schedule_free
from ypuddin.sampling import euler_sample

from .events import Emitter, NullEmitter
from .state import Progress, capture_rng, load_checkpoint, restore_rng, save_checkpoint

log = logging.getLogger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "no": torch.float32}


class StopRequested(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind  # "pause" | "stop"


class Trainer:
    def __init__(self, cfg: TrainConfig, *, device: str | torch.device | None = None, emitter: Emitter | None = None):
        self.cfg = cfg
        self.device = torch.device(device) if device else self._pick_device()
        self.run_dir = Path(cfg.checkpoint.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        events_path = cfg.logging.events_path or (self.run_dir / "events.jsonl")
        self.emitter = emitter or Emitter(path=events_path)
        self.config_hash = config_hash(cfg)
        self.progress = Progress()
        self.family: ModelFamily
        self.loaded: LoadedModel
        self.adapters: AdapterSet
        self.bundle: DataBundle
        self.objective: Objective
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any
        self.sampler: BucketBatchSampler
        self.loader: DataLoader
        self.text_cache: TextCache | None = None
        self.ema: dict[str, Tensor] | None = None
        self.swapper: BlockSwapper | None = None
        self.gen = torch.Generator().manual_seed(cfg.loop.seed)  # noise / timestep RNG (checkpointed)
        self._stop: str | None = None
        self._loss_ema: float | None = None
        self._prepared = False

    # ----------------------------------------------------------------- setup
    @staticmethod
    def _pick_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def compute_dtype(self) -> torch.dtype:
        if self.device.type == "cpu" and self.cfg.loop.mixed_precision != "no":
            return torch.bfloat16 if self.cfg.loop.mixed_precision == "bf16" else torch.float32
        return DTYPES[self.cfg.loop.mixed_precision]

    def _seed_all(self) -> None:
        s = self.cfg.loop.seed
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def emit(self, type_: str, **data: Any) -> None:
        self.emitter.emit(type_, **data)

    def prepare(self) -> None:
        cfg = self.cfg
        self._seed_all()
        write_config(cfg, self.run_dir / "config.toml")
        self.emit("run.started", config_hash=self.config_hash, device=str(self.device), run_dir=str(self.run_dir))
        self.family = get_family(cfg.model.family)
        self._check_capabilities()
        model_dtype = DTYPES[cfg.model.dtype] if self.device.type != "cpu" else torch.float32

        self.emit("phase.changed", phase="loading")
        self.loaded = self.family.load(cfg.model, cfg.memory, device=self.device, dtype=model_dtype)
        self.loaded.text.to(self.device)
        self.loaded.latent.to(self.device)

        self.emit("phase.changed", phase="indexing")
        cache_root = Path(cfg.dataset.cache_dir) if cfg.dataset.cache_dir else self.run_dir / "cache"
        self.bundle = build_data(cfg, self.family.spec.latent, cache_root=cache_root, progress=lambda k, d, t: self.emit("cache.progress", kind=k, done=d, total=t))
        self.emit("data.plan", **self.bundle.plan.to_dict())

        if cfg.dataset.cache_latents:
            self.emit("phase.changed", phase="caching_latents")
            n = cache_latents(
                self.bundle,
                self.loaded.latent.encode,
                device=self.device,
                batch_size=max(1, cfg.dataset.batch_size),
                dtype=torch.float32 if self.device.type == "cpu" else torch.bfloat16,
                progress=lambda d, t: self.emit("cache.progress", kind="latents", done=d, total=t),
            )
            log.info("cached %d latents", n)
            self.loaded.latent.unload()

        self.text_mode = self._resolve_text_mode()
        if self.text_mode == "cached":
            self.emit("phase.changed", phase="caching_text")
            self._build_text_cache(cache_root)

        self.emit("phase.changed", phase="injecting")
        presets = self.family.presets()
        if cfg.adapter.preset not in presets:
            raise ValueError(f"unknown adapter preset {cfg.adapter.preset!r} for {self.family.spec.name}; available: {sorted(presets)}")
        base_precision = cfg.memory.base_precision if cfg.memory.base_precision != "auto" else "keep"
        self.adapters = inject(self.loaded.backbone, cfg.adapter, presets[cfg.adapter.preset], prefix=self.family.spec.adapter_prefix, base_precision=base_precision)
        if cfg.adapter.resume_weights:
            from ypuddin.adapters import load_adapter_file

            tensors, _ = load_adapter_file(cfg.adapter.resume_weights)
            self.adapters.load_state(tensors, strict=False)
        self.emit("adapters.injected", **self.adapters.summary())
        if cfg.memory.blocks_to_swap > 0:
            blocks = self.family.memory_layout(self.loaded).blocks
            self.swapper = BlockSwapper(blocks, cfg.memory.blocks_to_swap, self.device)
            self.emit("memory.block_swap", **self.swapper.summary())

        self.objective = Objective(cfg.objective)
        groups = self.adapters.param_groups(cfg.optimizer.lr, cfg.optimizer.weight_decay, cfg.optimizer.group_lr)
        self.optimizer = build_optimizer(cfg.optimizer, groups)
        self.sampler = BucketBatchSampler(self.bundle.train.bucket_keys(), cfg.dataset.batch_size, seed=cfg.loop.seed)
        self.loader = DataLoader(self.bundle.train, batch_sampler=self.sampler, collate_fn=collate, num_workers=cfg.dataset.num_workers, pin_memory=self.device.type == "cuda")
        batches = self.sampler.batches_per_epoch()
        self.progress.steps_per_epoch = math.ceil(batches / cfg.loop.grad_accum)
        by_epochs = (cfg.loop.epochs or 10**9) * self.progress.steps_per_epoch
        self.progress.total_steps = min(by_epochs, cfg.loop.max_steps or 10**9)
        self.scheduler = None if is_schedule_free(cfg.optimizer) else build_scheduler(cfg.scheduler, self.optimizer, self.progress.total_steps)
        if cfg.loop.ema:
            self.ema = {k: v.detach().float().cpu().clone() for k, v in self.adapters.export_state()[0].items()}
        if cfg.checkpoint.resume:
            self._resume(cfg.checkpoint.resume)
        self._install_signal_handlers()
        self._prepared = True
        self.emit("run.prepared", total_steps=self.progress.total_steps, steps_per_epoch=self.progress.steps_per_epoch, trainable_params=self.adapters.num_params(), text_mode=self.text_mode)

    def _check_capabilities(self) -> None:
        caps = self.family.spec.capabilities
        cfg = self.cfg
        problems = []
        if cfg.memory.blocks_to_swap > 0 and "block_swap" not in caps:
            problems.append("memory.blocks_to_swap requires the block_swap capability")
        if cfg.memory.base_precision.startswith("fp8") and self.device.type != "cuda":
            problems.append("fp8 base precision requires CUDA")
        if cfg.dataset.masked_loss and "masked_loss" not in caps:
            problems.append("dataset.masked_loss is not supported by this family")
        if cfg.memory.activation_checkpointing != "none" and "activation_checkpointing" not in caps:
            problems.append("memory.activation_checkpointing is not supported by this family")
        problems += self.family.validate_config(cfg.model)
        if problems:
            raise ValueError("; ".join(problems))

    def _resolve_text_mode(self) -> str:
        mode = self.cfg.dataset.text_encoding
        if mode == "auto":
            return "online" if "online_text" in self.family.spec.capabilities else "cached"
        return mode

    def _build_text_cache(self, cache_root: Path) -> None:
        from ypuddin.data.captions import read_caption

        self.text_cache = TextCache(cache_root / "text")
        captions = [""]
        for ds in (self.bundle.train, self.bundle.validation):
            if ds is None:
                continue
            for it in ds.items:
                captions.append(read_caption(it.record.caption_path, it.source.class_prompt))
        n = build_text_cache(captions, self.text_cache, self.loaded.text.encode_for_cache, self.loaded.text.fingerprint, progress=lambda d, t: self.emit("cache.progress", kind="text", done=d, total=t), total=len(captions))
        log.info("cached %d text encodings", n)
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
        if ck["dataset_fingerprint"] and ck["dataset_fingerprint"] != self.bundle.plan.fingerprint:
            raise ValueError("checkpoint was trained on a different dataset (fingerprint mismatch)")
        if ck["config_hash"] and ck["config_hash"] != self.config_hash:
            log.warning("config changed since the checkpoint was written; resuming anyway")
        self.adapters.load_state(ck["adapter"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler is not None and ck["scheduler"]:
            self.scheduler.load_state_dict(ck["scheduler"])
        self.progress = ck["progress"]
        self.sampler.load_state_dict(ck["sampler"])
        restore_rng(ck["rng"], {"main": self.gen})
        if "ema" in ck and self.ema is not None:
            self.ema = {k: v.float() for k, v in ck["ema"].items()}
        self._loss_ema = self.progress.extra.get("loss_ema")
        self.emit("run.resumed", step=self.progress.step, epoch=self.progress.epoch, batch_in_epoch=self.progress.batch_in_epoch)

    def _adapter_metadata(self) -> dict[str, str]:
        _, targets = self.adapters.export_state()
        return build_metadata(
            targets=targets,
            adapter_cfg=self.cfg.adapter.model_dump(mode="json"),
            family=self.family.spec.name,
            architecture=f"{self.family.spec.architecture}/{self.cfg.adapter.algo}",
            title=self.cfg.checkpoint.name,
            resolution=",".join(str(r) for r in self.cfg.dataset.resolutions),
            config_hash=self.config_hash,
            dataset_fingerprint=self.bundle.plan.fingerprint,
            steps=self.progress.step,
            epoch=self.progress.epoch,
        )

    def save_weights(self, tag: str) -> Path:
        tensors, _ = self.adapters.export_state()
        path = save_adapter_file(self.run_dir / f"{self.cfg.checkpoint.name}-{tag}.safetensors", tensors, self._adapter_metadata(), dtype=self.cfg.checkpoint.save_dtype)
        if self.ema is not None:
            save_adapter_file(self.run_dir / f"{self.cfg.checkpoint.name}-{tag}-ema.safetensors", self.ema, self._adapter_metadata(), dtype=self.cfg.checkpoint.save_dtype)
        self.emit("checkpoint.saved", kind="weights", step=self.progress.step, path=str(path))
        self._rotate_weights()
        return path

    def _rotate_weights(self) -> None:
        keep = self.cfg.checkpoint.keep_last_n
        if not keep:
            return
        files = sorted(self.run_dir.glob(f"{self.cfg.checkpoint.name}-step*.safetensors"), key=lambda p: p.stat().st_mtime)
        for p in files[:-keep]:
            p.unlink(missing_ok=True)

    def save_state(self, tag: str | None = None) -> Path:
        tensors, _ = self.adapters.export_state()
        self.progress.extra["loss_ema"] = self._loss_ema
        path = save_checkpoint(
            self.run_dir / f"state-{tag or self.progress.step}",
            adapter_tensors=tensors,
            adapter_metadata=self._adapter_metadata(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler_state=self.sampler.state_dict(),
            progress=self.progress,
            rng=capture_rng({"main": self.gen}),
            ema_tensors=self.ema,
            config_hash=self.config_hash,
            dataset_fingerprint=self.bundle.plan.fingerprint,
        )
        self.emit("checkpoint.saved", kind="full", step=self.progress.step, path=str(path))
        return path

    # ----------------------------------------------------------------- batch processing
    def _text_cond(self, captions: list[str]) -> TextCond:
        if self.text_mode == "cached" and self.text_cache is not None:
            entries = [self.text_cache.get(TextCache.key(c, self.loaded.text.fingerprint)) for c in captions]
            return self.loaded.text.cond_from_cache(entries, self.device)
        return self.loaded.text.encode(captions, self.device)

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

    def compute_loss(self, batch: dict[str, Any], *, generator: torch.Generator | None = None, t_override: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(loss, per_sample_unweighted, t)``."""
        gen = generator or self.gen
        x0 = self._latents(batch)
        cond = self._text_cond(batch["caption"])
        t = t_override if t_override is not None else self.objective.sample_t(x0.shape[0], generator=gen, num_tokens=self._num_tokens(x0), device=self.device)
        x_t, target, _ = self.objective.prepare(x0, t.to(self.device), generator=gen)
        mask = batch.get("mask")
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask[:, None].to(self.device), size=x0.shape[-2:], mode="area")[:, 0]
        x_in = x_t.to(self.loaded.dtype if self.device.type != "cpu" else torch.float32)
        if self.swapper is not None and torch.is_grad_enabled():
            x_in.requires_grad_(True)  # gives every block an input grad so backward hooks fire in order
        with self._autocast():
            pred = self.family.forward(self.loaded, x_in, t.to(self.device), cond)
        loss, per_sample = self.objective.loss(pred.float(), target, t, mask=mask, sample_weight=batch["weight"])
        return loss, per_sample, t

    # ----------------------------------------------------------------- loop
    def run(self) -> str:
        if not self._prepared:
            self.prepare()
        cfg = self.cfg
        self.adapters.train(True)
        self.loaded.backbone.train()
        outcome = "finished"
        self.emit("phase.changed", phase="training")
        try:
            while self.progress.step < self.progress.total_steps:
                if cfg.loop.epochs is not None and self.progress.epoch >= cfg.loop.epochs:
                    break
                self._run_epoch()
        except StopRequested as e:
            outcome = "paused" if e.kind == "pause" else "stopped"
        self._finish(outcome)
        return outcome

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
        cfg = self.cfg
        epoch = self.progress.epoch
        self.sampler.set_epoch(epoch)
        self.sampler.set_position(self.progress.batch_in_epoch)
        self.bundle.train.set_epoch(epoch)
        self.emit("epoch.started", epoch=epoch, batches=len(self.sampler.plan()), position=self.progress.batch_in_epoch)
        accum = cfg.loop.grad_accum
        micro = 0
        group_loss = 0.0
        t0 = time.perf_counter()
        for batch in self.loader:
            self.progress.batch_in_epoch += 1
            self.progress.samples_seen += len(batch["caption"])
            loss, _, _ = self.compute_loss(batch)
            if not torch.isfinite(loss):
                self.progress.nan_skips += 1
                self.emit("warning", code="loss.nonfinite", step=self.progress.step, skips=self.progress.nan_skips)
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
        if self.progress.step >= self.progress.total_steps and self.progress.batch_in_epoch < len(self.sampler.plan()):
            return
        self.progress.epoch += 1
        self.progress.batch_in_epoch = 0
        self.emit("epoch.finished", epoch=epoch, step=self.progress.step)
        self._epoch_hooks(epoch + 1)

    def _optimizer_step(self, group_loss: float, elapsed: float) -> None:
        cfg = self.cfg
        params = self.adapters.parameters()
        grad_norm = None
        if cfg.optimizer.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optimizer.grad_clip_norm).item()
        else:
            grad_norm = math.sqrt(sum(float(p.grad.detach().float().pow(2).sum()) for p in params if p.grad is not None))
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.progress.nan_skips += 1
            self.emit("warning", code="grad.nonfinite", step=self.progress.step, skips=self.progress.nan_skips)
            if self.progress.nan_skips >= cfg.loop.nan_skip_limit:
                raise RuntimeError("too many non-finite gradients")
            return
        self.progress.nan_skips = 0
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.progress.step += 1
        self._update_ema()
        self._loss_ema = group_loss if self._loss_ema is None else 0.98 * self._loss_ema + 0.02 * group_loss
        step = self.progress.step
        if step % cfg.loop.log_every == 0 or step == self.progress.total_steps:
            lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(self.optimizer.param_groups)}
            remaining = self.progress.total_steps - step
            it_s = 1.0 / elapsed if elapsed > 0 else None
            self.emit(
                "step",
                step=step,
                epoch=self.progress.epoch,
                loss=group_loss,
                loss_ema=self._loss_ema,
                lr=lrs,
                grad_norm=grad_norm,
                it_s=it_s,
                eta_s=(remaining * elapsed) if it_s else None,
                vram_mb=(torch.cuda.max_memory_allocated() / 2**20) if self.device.type == "cuda" else None,
            )
        self._step_hooks(step)

    def _update_ema(self) -> None:
        if self.ema is None:
            return
        d = self.cfg.loop.ema_decay
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
        if cfg.validation.enabled and cfg.validation.every_epochs and finished_epochs % cfg.validation.every_epochs == 0:
            self.validate()
        if cfg.sampling.enabled and cfg.sampling.every_epochs and finished_epochs % cfg.sampling.every_epochs == 0:
            self.sample_images(tag=f"epoch{finished_epochs}")
        if cfg.checkpoint.save_every_epochs and finished_epochs % cfg.checkpoint.save_every_epochs == 0:
            self.save_weights(f"epoch{finished_epochs:04d}")

    def _finish(self, outcome: str) -> None:
        self.emit("phase.changed", phase="finalizing")
        if outcome == "finished" and self.cfg.checkpoint.save_on_finish:
            self.save_weights("final")
        self.emit(f"run.{outcome}", step=self.progress.step, epoch=self.progress.epoch, samples_seen=self.progress.samples_seen)
        self.emitter.close()

    # ----------------------------------------------------------------- validation & previews
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        vcfg = self.cfg.validation
        ds = self.bundle.validation
        if ds is None:
            return {}
        self.adapters.train(False)
        self.loaded.backbone.eval()
        if self.swapper is not None:
            self.swapper.set_forward_only(True)
        ds.set_epoch(0)
        ts = self.objective.sampler.icdf(vcfg.timesteps)
        per_t: dict[float, list[float]] = {q: [] for q in vcfg.timesteps}
        by_bucket: dict[tuple[int, int], list[int]] = {}
        for i, it in enumerate(ds.items):
            by_bucket.setdefault(it.bucket.key, []).append(i)
        bs = max(1, self.cfg.dataset.batch_size)
        for indices in by_bucket.values():
            for s in range(0, len(indices), bs):
                batch = collate([ds[i] for i in indices[s : s + bs]])
                for q, t_val in zip(vcfg.timesteps, ts.tolist()):
                    gen = torch.Generator().manual_seed(vcfg.seed * 100003 + int(q * 1000) + int(batch["index"][0]))
                    t = torch.full((len(batch["caption"]),), float(t_val))
                    _, per_sample, _ = self.compute_loss(batch, generator=gen, t_override=t)
                    per_t[q].extend(per_sample.tolist())
        result = {str(q): float(np.mean(v)) for q, v in per_t.items() if v}
        mean = float(np.mean(list(result.values()))) if result else float("nan")
        self.emit("validation", step=self.progress.step, per_t=result, mean=mean)
        if self.swapper is not None:
            self.swapper.set_forward_only(False)
        self.adapters.train(True)
        self.loaded.backbone.train()
        return result

    @torch.no_grad()
    def sample_images(self, tag: str) -> list[Path]:
        from PIL import Image

        scfg = self.cfg.sampling
        prompts = list(scfg.prompts)
        if scfg.prompts_file:
            prompts += _load_prompts_file(scfg.prompts_file)
        if not prompts:
            return []
        defaults = self.family.spec.sampling
        self.adapters.train(False)
        self.loaded.backbone.eval()
        if self.swapper is not None:
            self.swapper.set_forward_only(True)
        out_dir = self.run_dir / "samples"
        out_dir.mkdir(exist_ok=True)
        paths: list[Path] = []
        stride = self.family.spec.latent.stride
        for i, p in enumerate(prompts):
            w = (p.width or scfg.width) // self.family.spec.latent.align * self.family.spec.latent.align
            h = (p.height or scfg.height) // self.family.spec.latent.align * self.family.spec.latent.align
            steps = p.steps or scfg.steps or defaults.steps
            cfg_scale = p.cfg if p.cfg is not None else (scfg.cfg if scfg.cfg is not None else defaults.cfg)
            shift = scfg.shift or defaults.shift
            seed = p.seed if p.seed is not None else scfg.seed + i
            cond = self._text_cond([p.prompt])
            uncond = self._text_cond([p.negative])
            shape = (1, self.family.spec.latent.channels, h // stride, w // stride)
            model_dtype = self.loaded.dtype if self.device.type != "cpu" else torch.float32

            def predict(x: Tensor, t: Tensor, c: TextCond = cond) -> Tensor:
                with self._autocast():
                    return self.family.forward(self.loaded, x.to(model_dtype), t.to(self.device), c).float()

            latents = euler_sample(predict, shape, steps=steps, shift=shift, cfg=cfg_scale, predict_uncond=lambda x, t: predict(x, t, uncond), generator=torch.Generator().manual_seed(seed), device=self.device, dtype=torch.float32)
            self.loaded.latent.to(self.device)
            pixels = self.loaded.latent.decode(latents).clamp(-1, 1)
            arr = ((pixels[0].permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5).round().astype("uint8")
            path = out_dir / f"{tag}_{i:02d}_{seed}.png"
            Image.fromarray(arr).save(path)
            paths.append(path)
            self.emit("sample.saved", step=self.progress.step, prompt_index=i, prompt=p.prompt, seed=seed, path=str(path), width=w, height=h)
        if self.swapper is not None:
            self.swapper.set_forward_only(False)
        self.adapters.train(True)
        self.loaded.backbone.train()
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
    return [SamplePrompt(prompt=line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def train(cfg: TrainConfig, *, device: str | None = None, emitter: Emitter | None = None, listeners: list[Callable[[dict[str, Any]], None]] | None = None) -> str:
    Path(cfg.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
    em = emitter or Emitter(path=cfg.logging.events_path or (Path(cfg.checkpoint.output_dir) / "events.jsonl"), fd=int(os.environ["YPUDDIN_EVENTS_FD"]) if os.environ.get("YPUDDIN_EVENTS_FD") else None)
    for fn in listeners or []:
        em.add_listener(fn)
    trainer = Trainer(cfg, device=device, emitter=em)
    trainer.prepare()
    return trainer.run()


__all__ = ["Trainer", "train", "NullEmitter"]
