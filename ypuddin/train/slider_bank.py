"""Task-local frozen-base latent samples for text sliders."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from ypuddin.data.buckets import BucketManager
from ypuddin.data.cache import TextCache
from ypuddin.data.dataset import DataBundle, DataPlan
from ypuddin.models.fingerprints import content_fingerprint

from .preview_compute import preview_linear_compute
from .state import capture_rng, restore_rng


class SliderBankPlan(DataPlan):
    def to_dict(self):
        return {**super().to_dict(), "concept_samples": self.items}


class SliderBankDataset(Dataset):
    """Sampled x0 latents; these are concepts, not indexed user images."""

    def __init__(self, directory: Path, latents: list[torch.Tensor], recipe: dict):
        self.directory, self.latents, self.recipe = directory, latents, recipe
        self.epoch = 0

    def __len__(self):
        return len(self.latents)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def bucket_keys(self):
        side = self.recipe["resolution"]
        return [(side, side)] * len(self)

    def use_cached_captions(self):
        return [(TextCache.PROMPTS, self.recipe["neutral_prompt"])]

    def __getitem__(self, index: int) -> dict[str, Any]:
        side = self.recipe["resolution"]
        return {
            "index": index, "caption": self.recipe["neutral_prompt"], "uncond": False,
            "is_reg": False, "weight": 1.0, "bucket": (side, side),
            "geometry": {"original_size": (side, side), "crop_top_left": (0, 0),
                         "target_size": (side, side)},
            "path": str(self.directory / _filename(index)), "cache_name": TextCache.PROMPTS,
            "latents": self.latents[index],
        }


def _filename(index: int) -> str:
    return f"sample-{index:05d}.safetensors"


def _recipe(trainer) -> dict:
    cfg, family, loaded = trainer.cfg, trainer.family, trainer.loaded
    settings, sampling, spec = cfg.slider, cfg.sampling, family.spec.latent
    side = settings.resolution // spec.align * spec.align
    if side < spec.align:
        raise ValueError("文字滑块的样本尺寸小于模型对齐尺寸")
    defaults = family.sampling_defaults(loaded)
    tokens = (side // spec.stride // spec.patch) ** 2
    return {
        "version": 1, "model_identity": trainer.model_identity,
        "family": family.spec.name, "neutral_prompt": settings.neutral_prompt,
        "count": settings.samples_per_epoch, "resolution": side,
        "requested_resolution": settings.resolution,
        "shape": [spec.channels, side // spec.stride, side // spec.stride],
        "seeds": [cfg.loop.seed + 1000 + i for i in range(settings.samples_per_epoch)],
        "sampling": {
            "steps": settings.sampling_steps, "sampler": sampling.sampler,
            "scheduler": sampling.scheduler, "noise": sampling.noise,
            "cfg": sampling.cfg if sampling.cfg is not None else defaults.cfg,
            "guidance": sampling.guidance if sampling.guidance is not None else defaults.guidance,
            "shift": sampling.shift or family.sampling_shift_for_model(
                loaded, tokens, cfg.objective, steps=settings.sampling_steps,
            ),
            "er_sde_order": sampling.er_sde_order, "er_sde_s_noise": sampling.er_sde_s_noise,
        },
    }


@torch.no_grad()
def _generate(trainer, directory: Path, recipe: dict) -> None:
    loaded, family, cfg = trainer.loaded, trainer.family, trainer.cfg
    rng = capture_rng(device=trainer.device)
    modes = [(module, module.training) for module in loaded.backbone.modules()]
    try:
        loaded.latent.unload()
        loaded.text.to(trainer.device)
        encode_prompt = getattr(loaded.text, "encode_prompt", None)

        def encode(text):
            if callable(encode_prompt):
                return encode_prompt([text], trainer.device, mode=cfg.sampling.noise)
            return loaded.text.encode([text], trainer.device)

        cond, uncond = encode(recipe["neutral_prompt"]), encode("")
        loaded.text.unload()
        family.materialize_backbone(loaded)
        trainer._prepare_attention()
        loaded.backbone.to(trainer.device).eval()
        dtype = loaded.dtype if trainer.device.type != "cpu" else torch.float32
        settings = recipe["sampling"]

        def predict(x, t, conditioning=cond):
            compute = (
                nullcontext() if cfg.sampling.noise == "comfyui"
                else preview_linear_compute(getattr(trainer, "compute_policy", None), loaded.backbone)
            )
            with trainer._autocast(), compute:
                return family.forward(
                    loaded, x.to(dtype), t.to(trainer.device), conditioning,
                    inference=True, guidance=settings["guidance"],
                ).float()

        def predict_uncond(x, t):
            return predict(x, t, uncond)

        with trainer._autocast():
            pair = family.sampling_prediction_pair(
                loaded, predict, cond, uncond, cfg=settings["cfg"], noise=settings["noise"],
            )
        records = []
        for index, seed in enumerate(recipe["seeds"]):
            def on_step(done, total, sample_index=index):
                trainer.emit(
                    "cache.progress", kind="slider_bank", done=sample_index * total + done,
                    total=recipe["count"] * total,
                )

            latent = family.sample_latents(
                loaded, predict, (1, *recipe["shape"]),
                **{key: value for key, value in settings.items() if key != "guidance"},
                predict_uncond=predict_uncond,
                generator=torch.Generator().manual_seed(seed),
                device=trainer.device, dtype=torch.float32, on_step=on_step,
                **({"predict_pair": pair} if pair is not None else {}),
            ).detach().float().cpu()
            if list(latent.shape) != [1, *recipe["shape"]] or not torch.isfinite(latent).all():
                raise ValueError("文字滑块的底模采样未生成有效 latent")
            path = directory / _filename(index)
            save_file({"latent": latent[0].contiguous()}, str(path))
            records.append({
                "file": path.name,
                "identity": content_fingerprint([path], namespace="slider-bank-latent-v1"),
            })
        manifest = {"format": "ypuddin-slider-bank", "version": 1, "recipe": recipe, "samples": records}
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(directory / "manifest.json")
    finally:
        try:
            loaded.backbone.cpu()
            loaded.text.unload()
            for module, training in modes:
                module.training = training
        finally:
            restore_rng(rng, device=trainer.device)


def _read_bank(directory: Path, recipe: dict, expected_identity: str | None):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "ypuddin-slider-bank" or manifest.get("version") != 1
            or manifest.get("recipe") != recipe):
        raise ValueError("文字滑块概念库与当前模型或生成配置不一致")
    records = manifest.get("samples")
    names = [_filename(i) for i in range(recipe["count"])]
    if not isinstance(records, list) or [record.get("file") for record in records] != names:
        raise ValueError("文字滑块概念库的样本清单不完整")
    paths = [directory / name for name in names]
    identity = content_fingerprint([manifest_path, *paths], namespace="slider-bank-v1")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("文字滑块概念库已改变，不能恢复训练")
    latents = []
    for path, record in zip(paths, records, strict=True):
        if content_fingerprint([path], namespace="slider-bank-latent-v1") != record.get("identity"):
            raise ValueError("文字滑块概念样本已改变，不能恢复训练")
        tensors = load_file(str(path), device="cpu")
        latent = tensors.get("latent")
        if (set(tensors) != {"latent"} or latent is None or list(latent.shape) != recipe["shape"]
                or latent.dtype != torch.float32 or not torch.isfinite(latent).all()):
            raise ValueError("文字滑块概念样本无效")
        latents.append(latent)
    if content_fingerprint([manifest_path, *paths], namespace="slider-bank-v1") != identity:
        raise ValueError("文字滑块概念库在读取期间改变")
    return latents, identity


def build_slider_bank(trainer, cache_root: Path) -> DataBundle:
    """Generate once, or verify a checkpoint's original bank without sampling again."""
    cfg = trainer.cfg
    if cfg.training.task != "slider" or cfg.slider.mode != "text":
        raise ValueError("概念库仅用于文字滑块训练")
    if cfg.dataset.resolution_mode != "bucket":
        raise ValueError("文字滑块使用滑块分辨率，请选择分桶模式")
    recipe = _recipe(trainer)
    expected = None
    if cfg.checkpoint.resume:
        saved = (getattr(trainer, "_resume_extra", None) or {}).get("slider_bank")
        if not isinstance(saved, dict) or not saved.get("directory") or not saved.get("identity"):
            raise ValueError("恢复状态缺少文字滑块概念库")
        directory, expected = Path(saved["directory"]), saved["identity"]
        if not directory.is_absolute():
            raise ValueError("恢复状态中的文字滑块概念库路径无效")
    else:
        # A shared dataset cache must never own a task's frozen concept samples.
        directory = (Path(trainer.run_dir) / "slider-bank").resolve()
        if not directory.exists():
            directory.mkdir(parents=True)
            _generate(trainer, directory, recipe)
    try:
        latents, identity = _read_bank(directory, recipe, expected)
    except OSError as error:
        raise ValueError("文字滑块概念库缺失或不完整，不能恢复训练") from error
    trainer.progress.extra["slider_bank"] = {
        "directory": str(directory), "identity": identity, "count": len(latents),
    }
    side = recipe["resolution"]
    plan = SliderBankPlan(
        images=0, items=len(latents), fingerprint=identity,
        buckets=[{"base": side, "w": side, "h": side, "items": len(latents)}],
    )
    buckets = BucketManager([side], align=trainer.family.spec.latent.align, step=trainer.family.spec.latent.align)
    return DataBundle(SliderBankDataset(directory, latents, recipe), None, plan, [], buckets, None)
