"""SDXL base UNet, dual CLIP and AutoencoderKL with DDPM epsilon/v-prediction training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import MemoryConfig, ModelConfig, ObjectiveConfig, TrainConfig
from ypuddin.models.base import (
    LatentSpec,
    LoadedModel,
    MemoryLayout,
    ModelFamily,
    ModelSpec,
    SamplingDefaults,
    TextCond,
    TextSpec,
)
from ypuddin.models.registry import register

from .latent import SDXLLatent
from .loading import (
    check_component_storage,
    component_config,
    component_keys,
    component_path,
    load_diffusers_component,
)
from .text import SDXLText, tokenizer_paths


class SDXLFamily(ModelFamily):
    spec = ModelSpec(
        name="sdxl",
        attention_backends=("auto", "sdpa", "xformers"),
        label="SDXL",
        latent=LatentSpec(4, 8, 1, "sdxl-vae-f8c4-v1"),
        text=TextSpec(77, "sdxl-dual-clip-penultimate-pooled-v1", encoder_params=817_000_000),
        sampling=SamplingDefaults(steps=28, cfg=7.0, shift=1.0, sampler="euler"),
        capabilities=frozenset({"activation_checkpointing", "masked_loss"}),
        objective="ddpm",
        t_convention="ddpm_1000",
        architecture="stable-diffusion-xl-v1-base",
        adapter_prefix="lora_unet",
        weights=(
            ("dit_path", "SDXL 模型", "完整 SDXL checkpoint（可内含双 CLIP 与 VAE）或 Diffusers 目录"),
            ("text_encoder_path", "CLIP-L", "可选：覆盖模型内含的第一个文本编码器"),
            ("text_encoder_2_path", "CLIP-G", "可选：覆盖模型内含的第二个文本编码器"),
            ("vae_path", "SDXL VAE", "可选：覆盖模型内含的 VAE"),
        ),
        optional_weights=("text_encoder_path", "text_encoder_2_path", "vae_path"),
        sampling_samplers=("euler", "heun"),
        sampling_schedulers=("uniform",),
        objective_timestep_sampling=("uniform", "logit_normal"),
        objective_weighting=("none",),
    )

    def validate_config(self, cfg: ModelConfig) -> list[str]:
        problems = []
        if not cfg.dit_path:
            problems.append("model.dit_path is required for SDXL")
        for field in ("dit_path", "text_encoder_path", "text_encoder_2_path", "vae_path", "tokenizer_path"):
            value = getattr(cfg, field, None)
            if value and not Path(value).expanduser().exists():
                problems.append(f"model.{field} does not exist: {value}")
        if cfg.attention not in {"auto", "sdpa", "xformers"}:
            problems.append("SDXL supports auto/SDPA or xFormers attention")
        if cfg.zero_terminal_snr and cfg.prediction_type != "v_prediction":
            problems.append("SDXL zero_terminal_snr requires v_prediction")
        if cfg.dit_path and Path(cfg.dit_path).expanduser().exists():
            for component, override in (
                ("unet", None),
                ("text_encoder", cfg.text_encoder_path),
                ("text_encoder_2", cfg.text_encoder_2_path),
                ("vae", cfg.vae_path),
            ):
                path = component_path(cfg.dit_path, component, override)
                if path.is_dir():
                    base = (
                        "diffusion_pytorch_model.safetensors"
                        if component in {"unet", "vae"}
                        else "model.safetensors"
                    )
                    index = path / (base + ".index.json")
                    if not (path / "config.json").is_file() or not (
                        (path / base).is_file() or index.is_file()
                    ):
                        problems.append(
                            f"SDXL {component} directory needs config.json and safetensors weights: {path}"
                        )
                    if index.is_file():
                        import json

                        try:
                            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                            if not shards or any(not (path / shard).is_file() for shard in shards):
                                problems.append(
                                    f"SDXL {component} directory has missing safetensors shards: {path}"
                                )
                        except (OSError, ValueError, KeyError, AttributeError, TypeError):
                            problems.append(f"SDXL {component} safetensors index is invalid: {index}")
                elif path.is_file() and path.suffix.lower() == ".safetensors":
                    from safetensors import SafetensorError, safe_open

                    try:
                        with safe_open(str(path), framework="pt", device="cpu") as file:
                            component_keys(list(file.keys()), component)
                    except (OSError, ValueError, SafetensorError) as error:
                        problems.append(str(error))
                if path.exists():
                    from safetensors import SafetensorError

                    try:
                        check_component_storage(path, component)
                    except (
                        OSError,
                        ValueError,
                        SafetensorError,
                        KeyError,
                        TypeError,
                        AttributeError,
                    ) as error:
                        if str(error) not in problems:
                            problems.append(str(error))
            try:
                tokenizers = tokenizer_paths(Path(cfg.dit_path).expanduser(), cfg.tokenizer_path)
                for tokenizer in tokenizers:
                    if not (tokenizer / "tokenizer.json").is_file() and not all(
                        (tokenizer / name).is_file() for name in ("vocab.json", "merges.txt")
                    ):
                        problems.append(f"SDXL tokenizer is incomplete: {tokenizer}")
            except ValueError as error:
                problems.append(str(error))
        return problems

    def training_options_errors(self, cfg: TrainConfig) -> list[dict[str, str]]:
        problems = super().training_options_errors(cfg)
        if cfg.memory.activation_checkpointing == "unsloth":
            problems.append(
                {"loc": "memory.activation_checkpointing", "msg": "SDXL supports none or block checkpointing"}
            )
        return problems

    def load(
        self, cfg: ModelConfig, memory: MemoryConfig, *, device, dtype, backbone_device=None
    ) -> LoadedModel:
        problems = self.validate_config(cfg)
        if problems:
            raise ValueError("; ".join(problems))
        if memory.blocks_to_swap or memory.activation_checkpointing == "unsloth" or memory.compile:
            raise ValueError(
                "SDXL currently supports block checkpointing, without block swap/unsloth/compile"
            )
        if memory.base_precision.startswith("fp8"):
            raise ValueError("SDXL FP8 base conversion is not supported")
        path = Path(cfg.dit_path).expanduser()
        unet_path = component_path(path, "unet")
        unet = load_diffusers_component(unet_path, "unet", device=backbone_device or device, dtype=dtype)
        if memory.activation_checkpointing == "block":
            unet.enable_gradient_checkpointing()
        if cfg.attention == "xformers":
            if torch.device(device).type != "cuda":
                raise ValueError("SDXL xFormers attention requires CUDA")
            unet.enable_xformers_memory_efficient_attention()
        text = SDXLText(
            component_path(path, "text_encoder", cfg.text_encoder_path),
            component_path(path, "text_encoder_2", cfg.text_encoder_2_path),
            tokenizer_paths(path, cfg.tokenizer_path),
            device=device,
            dtype=dtype,
        )
        config = dict(unet.config)
        if config.get("in_channels") != 4 or config.get("out_channels") != 4:
            raise ValueError(
                "SDXL base training requires a 4-channel UNet (inpainting/refiner is unsupported)"
            )
        if (
            config.get("addition_embed_type") != "text_time"
            or config.get("cross_attention_dim") != text.hidden_size
        ):
            raise ValueError("SDXL UNet does not match the dual CLIP conditioning dimensions")
        projection_input = 6 * int(config["addition_time_embed_dim"]) + text.pooled_size
        if config.get("projection_class_embeddings_input_dim") != projection_input:
            raise ValueError("SDXL UNet pooled/time conditioning dimensions do not match CLIP-G")
        latent = SDXLLatent(component_path(path, "vae", cfg.vae_path), device=device)
        return LoadedModel(
            unet,
            text,
            latent,
            torch.device(device),
            dtype,
            extra={
                "dit_config": config,
                "prediction_type": cfg.prediction_type,
                "zero_terminal_snr": cfg.zero_terminal_snr,
                "num_train_timesteps": 1000,
            },
        )

    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        if extra.get("inference", False):
            from ypuddin.sampling.ddpm import sampling_timestep

            timestep = sampling_timestep(t, loaded.extra.get("num_train_timesteps", 1000))
        else:
            from ypuddin.objectives.ddpm import unit_to_timesteps

            timestep = unit_to_timesteps(t, loaded.extra.get("num_train_timesteps", 1000))
        geometry = extra.get("geometry")
        if geometry is None:
            size = x_t.new_tensor([x_t.shape[-2] * 8, x_t.shape[-1] * 8]).expand(x_t.shape[0], -1)
            geometry = {"original_size": size, "crop_top_left": torch.zeros_like(size), "target_size": size}
        values = []
        for key in ("original_size", "crop_top_left", "target_size"):
            value = geometry[key]
            if value.shape != (x_t.shape[0], 2):
                raise ValueError(f"SDXL geometry.{key} must have shape [batch, 2]")
            values.append(value.to(device=x_t.device, dtype=loaded.dtype))
        return loaded.backbone(
            x_t,
            timestep.to(x_t.device),
            encoder_hidden_states=cond["embeds"].to(x_t.device, loaded.dtype),
            added_cond_kwargs={
                "text_embeds": cond["pooled"].to(x_t.device, loaded.dtype),
                "time_ids": torch.cat(values, dim=-1),
            },
            return_dict=False,
        )[0]

    def build_objective(self, loaded: LoadedModel, cfg: ObjectiveConfig):
        from ypuddin.objectives.ddpm import DDPMObjective

        return DDPMObjective(cfg, **self._schedule_options(loaded))

    @staticmethod
    def _schedule_options(loaded: LoadedModel) -> dict[str, Any]:
        return {
            key: loaded.extra[key] for key in ("prediction_type", "zero_terminal_snr", "num_train_timesteps")
        }

    def sample_latents(self, loaded: LoadedModel, predict, shape, **options):
        from ypuddin.sampling.ddpm import sample_ddpm

        return sample_ddpm(predict, shape, **self._schedule_options(loaded), **options)

    def sampling_shift(self, num_tokens: int, objective: Any | None = None) -> float:
        return 1.0

    def presets(self) -> dict[str, TargetPreset]:
        attn = ("*.attn1.{to_q,to_k,to_v,to_out.0}", "*.attn2.{to_q,to_k,to_v,to_out.0}")
        mlp = ("*.ff.net.0.proj", "*.ff.net.2")
        return {
            "attn-only": TargetPreset("attn-only", attn, description="仅训练图像模型的注意力投影，训练参数更少。"),
            "attn-mlp": TargetPreset("attn-mlp", attn + mlp, description="训练图像模型的注意力和前馈层。"),
        }

    def memory_layout(self, loaded: LoadedModel) -> MemoryLayout:
        return MemoryLayout()

    def latent_fingerprint(self, cfg: ModelConfig, *, dtype: torch.dtype) -> str:
        if not cfg.dit_path and not cfg.vae_path:
            return self.spec.latent.fingerprint
        path = component_path(cfg.dit_path, "vae", cfg.vae_path)
        if not path.exists():
            return self.spec.latent.fingerprint
        return SDXLLatent(path).fingerprint

    def meta_backbone(self, cfg: ModelConfig) -> nn.Module:
        from diffusers import UNet2DConditionModel

        path = component_path(cfg.dit_path or ".", "unet")
        return UNet2DConditionModel.from_config(component_config(path, "unet"))

    def linear_module_names(self) -> list[str]:
        with torch.device("meta"):
            model = self.meta_backbone(ModelConfig(family="sdxl"))
        return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


register("sdxl", SDXLFamily)
