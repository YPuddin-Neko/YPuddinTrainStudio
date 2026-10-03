"""SDXL base UNet, dual CLIP and AutoencoderKL with DDPM epsilon/v-prediction training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset, adaptable_modules
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
    checkpoint_objective,
    component_config,
    component_keys,
    component_path,
    load_diffusers_component,
    objective_problems,
)
from .text import SDXLText, tokenizer_paths


class SDXLFamily(ModelFamily):
    spec = ModelSpec(
        name="sdxl",
        attention_backends=("auto", "sdpa", "xformers", "metal_flash"),
        label="SDXL 2.6B",
        latent=LatentSpec(4, 8, 1, "sdxl-vae-f8c4-v1"),
        text=TextSpec(77, "sdxl-dual-clip-penultimate-pooled-v1", encoder_params=817_000_000),
        sampling=SamplingDefaults(steps=28, cfg=7.0, shift=1.0, sampler="euler"),
        capabilities=frozenset({"activation_checkpointing", "masked_loss"}),
        # Peak activations of LoRA on the attn-mlp linears, measured on Apple's GPU with flash-style attention
        # (1-10% above a meta-device count of the saved tensors). Diffusers' checkpointing keeps each ResNet
        # and transformer block's inputs and recomputes one block at a time.
        backbone_activation_units=(("none", 222_500.0), ("block", 20_500.0)),
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
        sampling_samplers=("euler", "euler_ancestral", "heun"),
        sampling_schedulers=("uniform",),
        objective_timestep_sampling=("uniform", "logit_normal"),
        objective_weighting=("none", "min_snr"),
    )

    def validate_config(self, cfg: ModelConfig) -> list[str]:
        problems = []
        if not cfg.dit_path:
            problems.append("model.dit_path is required for SDXL")
        for field in ("dit_path", "text_encoder_path", "text_encoder_2_path", "vae_path", "tokenizer_path"):
            value = getattr(cfg, field, None)
            if value and not Path(value).expanduser().exists():
                problems.append(f"model.{field} does not exist: {value}")
        if cfg.attention not in {"auto", "sdpa", "xformers", "metal_flash"}:
            problems.append("SDXL supports auto/SDPA, xFormers or Metal FlashAttention")
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
                            keys = list(file.keys())
                        component_keys(keys, component)
                        if component == "unet":
                            problems += objective_problems(checkpoint_objective(keys), cfg)
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
        if cfg.attention == "metal_flash":
            from ypuddin.models.metal_attention import require_metal_flash

            require_metal_flash(device)
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
        # xFormers / Metal processors are installed by prepare_attention once a real-dtype check passes.
        text = SDXLText(
            component_path(path, "text_encoder", cfg.text_encoder_path),
            component_path(path, "text_encoder_2", cfg.text_encoder_2_path),
            tokenizer_paths(path, cfg.tokenizer_path),
            device=device,
            dtype=dtype,
            max_token_length=cfg.sdxl_max_token_length,
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

    def prepare_attention(self, loaded, configured, *, device, dtype, training, pinned=None):
        """Check the UNet's own processor on tiny self/cross layers, then switch every layer at once.

        Diffusers' ``enable_xformers_memory_efficient_attention`` probes with FP32 inputs,
        which FlashAttention-only xFormers builds reject (SM120); this check uses the
        UNet's real attention dtype and head dims instead. A failed check leaves the
        loaded SDPA processors untouched.
        """
        from diffusers.models.attention_processor import Attention, AttnProcessor2_0, XFormersAttnProcessor

        from ypuddin.models.attention_check import check_attention, grad_mode

        unet = loaded.backbone
        shapes = sorted(
            {
                (module.inner_dim // module.heads, module.cross_attention_dim if module.is_cross_attention else None)
                for module in unet.modules()
                if isinstance(module, Attention)
            },
            key=lambda shape: (shape[0], shape[1] or 0),
        )
        selected = "sdpa" if configured == "auto" else configured

        def processor(backend):
            if backend == "xformers":
                import importlib

                importlib.import_module("xformers.ops")
                if any(type(p) is not AttnProcessor2_0 for p in unet.attn_processors.values()):
                    raise ValueError("SDXL UNet 含有非默认的注意力处理器，不能整体切换到 xFormers")
                return XFormersAttnProcessor()
            if backend == "metal_flash":
                from ypuddin.models.metal_attention import _SDXLMetalProcessor

                return _SDXLMetalProcessor(AttnProcessor2_0())
            return AttnProcessor2_0()

        def run(backend):
            with torch.autocast(torch.device(device).type, enabled=False), grad_mode(training):
                for head_dim, cross_dim in shapes:
                    layer = Attention(
                        query_dim=2 * head_dim,
                        heads=2,
                        dim_head=head_dim,
                        cross_attention_dim=cross_dim,
                        processor=processor(backend),
                    ).to(device=device, dtype=dtype)
                    hidden = torch.randn(1, 16, 2 * head_dim, device=device, dtype=dtype, requires_grad=training)
                    context = (
                        None
                        if cross_dim is None
                        else torch.randn(1, 8, cross_dim, device=device, dtype=dtype, requires_grad=training)
                    )
                    out = layer(hidden, encoder_hidden_states=context)
                    if training:
                        out.float().square().mean().backward()

        actual = check_attention(
            configured=configured, selected=selected, run=run, device=device, dtype=dtype,
            training=training, pinned=pinned,
        )
        if actual == "xformers":
            unet.set_attn_processor(XFormersAttnProcessor())
        elif actual == "metal_flash":
            from ypuddin.models.metal_attention import install_metal_flash_processors

            install_metal_flash_processors(unet, "sdxl")

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
        # kohya's LoCon scope: every layer of the ResNet and resampling blocks, time_emb_proj included.
        resnet = (
            "*.resnets.*.{conv1,conv2,conv_shortcut,time_emb_proj}",
            "*.{downsamplers,upsamplers}.*.conv",
        )
        return {
            "attn-only": TargetPreset("attn-only", attn, description="只训练注意力层，训练的参数最少。"),
            "attn-mlp": TargetPreset(
                "attn-mlp",
                attn + mlp,
                description="训练注意力层和前馈层，适合大多数 LoRA / LoKr；同时训练卷积层时，还包括 ResNet 模块和上下采样层。",
                conv=resnet,
            ),
            "all-layers": TargetPreset(
                "all-layers",
                ("*",),
                description="训练 UNet 里的全部线性层；同时训练卷积层时，卷积层也全部训练。",
                conv=("*",),
            ),
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
        # Import-time library probes require real CPU scalars, not meta tensors.
        with torch.device("cpu"):
            from diffusers import UNet2DConditionModel

        path = component_path(cfg.dit_path or ".", "unet")
        return UNet2DConditionModel.from_config(component_config(path, "unet"))

    def linear_module_names(self) -> list[str]:
        return [name for name, kernel in self.adaptable_modules().items() if not kernel]

    def adaptable_modules(self) -> dict[str, tuple[int, ...]]:
        with torch.device("meta"):
            return adaptable_modules(self.meta_backbone(ModelConfig(family="sdxl")))


register("sdxl", SDXLFamily)
