"""FLUX.1 dev/schnell text-to-image: lazy weights, native packed latent flow prediction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import ModelConfig, TrainConfig
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

from .latent import FluxLatent
from .loading import DEFAULTS, component_config, component_path, create_meta, load_component, weight_files
from .text import FluxText, tokenizer_paths


def pack_latents(latents: Tensor) -> Tensor:
    b, c, h, w = latents.shape
    if c != 16 or h % 2 or w % 2:
        raise ValueError("FLUX.1 latents require 16 channels and even spatial dimensions")
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w // 4, c * 4)


def unpack_latents(tokens: Tensor, height: int, width: int) -> Tensor:
    b = tokens.shape[0]
    return (
        tokens.reshape(b, height // 2, width // 2, 16, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, 16, height, width)
    )


def image_ids(height: int, width: int, *, device, dtype) -> Tensor:
    ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(height // 2, device=device)[:, None]
    ids[..., 2] = torch.arange(width // 2, device=device)[None, :]
    return ids.reshape(-1, 3)


class FluxFamily(ModelFamily):
    spec = ModelSpec(
        name="flux",
        label="FLUX.1",
        latent=LatentSpec(16, 8, 2, "flux1-ae-f8c16-v1"),
        text=TextSpec(512, "flux1-clip-l-t5xxl-v1", encoder_params=4_885_000_000),
        sampling=SamplingDefaults(steps=28, cfg=1, shift=None, sampler="euler", guidance=3.5),
        capabilities=frozenset({"activation_checkpointing", "fp8_base", "masked_loss"}),
        architecture="flux1",
        adapter_prefix="lora_transformer",
        weights=(
            ("dit_path", "FLUX.1 模型", "dev / schnell 主模型或完整 Diffusers 目录"),
            ("text_encoder_path", "CLIP-L", "完整模型目录可留空；独立 DiT 必须选择 CLIP-L"),
            ("text_encoder_2_path", "T5-XXL", "完整模型目录可留空；独立 DiT 必须选择 T5-XXL"),
            ("vae_path", "FLUX AE", "完整模型目录可留空；独立 DiT 必须选择 FLUX AE"),
            ("tokenizer_path", "分词器", "可选：包含 tokenizer/ 和 tokenizer_2/ 的目录"),
        ),
        optional_weights=("text_encoder_path", "text_encoder_2_path", "vae_path", "tokenizer_path"),
        attention_backends=("auto", "sdpa"),
    )

    @staticmethod
    def _paths(cfg):
        root = Path(cfg.dit_path).expanduser()
        return tuple(
            component_path(root, component, getattr(cfg, field, None) if field else None)
            for component, field in (
                ("transformer", None),
                ("text_encoder", "text_encoder_path"),
                ("text_encoder_2", "text_encoder_2_path"),
                ("vae", "vae_path"),
            )
        )

    def validate_config(self, cfg: ModelConfig) -> list[str]:
        issues = []
        if not cfg.dit_path:
            return ["model.dit_path is required for FLUX.1"]
        if cfg.attention not in {"auto", "sdpa"}:
            issues.append("FLUX.1 currently supports auto/SDPA attention")
        try:
            paths = self._paths(cfg)
            for path, component in zip(
                paths, ("transformer", "text_encoder", "text_encoder_2", "vae"), strict=True
            ):
                weight_files(path, component)
                component_config(path, component)
            root = Path(cfg.dit_path).expanduser()
            index = root / "model_index.json" if root.is_dir() else root.parent / "model_index.json"
            if index.is_file() and json.loads(index.read_text(encoding="utf-8")).get(
                "_class_name", "FluxPipeline"
            ) not in {"FluxPipeline", "FluxImg2ImgPipeline"}:
                issues.append(
                    "Only FLUX.1 dev/schnell text-to-image weights are supported; Kontext/Fill/Control are unsupported"
                )
            tokenizer_paths(root, cfg.tokenizer_path, paths[1:3])
        except (ValueError, OSError, KeyError, TypeError, SafetensorError) as error:
            issues.append(str(error))
        return issues

    def training_options_errors(self, cfg: TrainConfig):
        issues = super().training_options_errors(cfg)
        if cfg.memory.activation_checkpointing == "unsloth":
            issues.append(
                {
                    "loc": "memory.activation_checkpointing",
                    "msg": "FLUX.1 supports none or block checkpointing",
                }
            )
        return issues

    def load(self, cfg, memory, *, device, dtype, backbone_device=None):
        issues = self.validate_config(cfg)
        if issues:
            raise ValueError("; ".join(issues))
        if memory.blocks_to_swap or memory.compile or memory.activation_checkpointing == "unsloth":
            raise ValueError(
                "FLUX.1 supports block checkpointing; block swap/compile/unsloth are not yet supported"
            )
        paths = self._paths(cfg)
        config = component_config(paths[0], "transformer")
        guided = bool(config["guidance_embeds"])
        text = FluxText(
            paths[1:3],
            tokenizer_paths(Path(cfg.dit_path).expanduser(), cfg.tokenizer_path, paths[1:3]),
            max_len=512 if guided else 256,
            device=device,
            dtype=dtype,
        )
        if (
            text.hidden_size != config["joint_attention_dim"]
            or text.pooled_size != config["pooled_projection_dim"]
        ):
            raise ValueError("FLUX.1 transformer conditioning does not match the selected CLIP-L/T5 encoders")
        loaded = LoadedModel(
            create_meta("transformer", config),
            text,
            FluxLatent(paths[3], device=device),
            torch.device(device),
            dtype,
            extra={
                "dit_config": config,
                "variant": "dev" if guided else "schnell",
                "training_guidance": cfg.training_guidance,
                "pending_backbone": paths[0],
                "backbone_device": torch.device(backbone_device or device),
                "checkpoint_blocks": memory.activation_checkpointing == "block",
            },
        )
        return loaded

    def materialize_backbone(self, loaded):
        path = loaded.extra.get("pending_backbone")
        if path is None:
            return
        # The frozen encoders are no longer needed once conditioning/caches exist.
        # Also enforce this for direct family callers, not only Trainer.prepare().
        loaded.text.unload()
        loaded.latent.unload()
        loaded.backbone = load_component(
            path,
            "transformer",
            device=loaded.extra["backbone_device"],
            dtype=loaded.dtype,
            config=loaded.extra["dit_config"],
        )
        if loaded.extra["checkpoint_blocks"]:
            loaded.backbone.enable_gradient_checkpointing()
        loaded.extra["pending_backbone"] = None

    def forward(self, loaded, x_t, t, cond: TextCond, **extra: Any):
        self.materialize_backbone(loaded)
        cond = cond.to(x_t.device, x_t.dtype)
        guidance = None
        if loaded.extra["dit_config"]["guidance_embeds"]:
            value = extra.get("guidance")
            if value is None:
                value = (
                    self.spec.sampling.guidance
                    if extra.get("inference")
                    else loaded.extra["training_guidance"]
                )
            guidance = torch.as_tensor(value, device=x_t.device, dtype=x_t.dtype).expand(x_t.shape[0])
            if not torch.isfinite(guidance).all() or (guidance < 0).any():
                raise ValueError("FLUX.1 guidance must be finite and nonnegative")
        output = loaded.backbone(
            hidden_states=pack_latents(x_t),
            timestep=t.to(x_t.device, x_t.dtype),
            encoder_hidden_states=cond["embeds"],
            pooled_projections=cond["pooled"],
            img_ids=image_ids(*x_t.shape[-2:], device=x_t.device, dtype=x_t.dtype),
            txt_ids=torch.zeros(cond["embeds"].shape[1], 3, device=x_t.device, dtype=x_t.dtype),
            guidance=guidance,
            return_dict=False,
        )[0]
        return unpack_latents(output, *x_t.shape[-2:])

    def sampling_shift(self, num_tokens, objective=None):
        return math.exp(0.5 + (num_tokens - 256) * (1.15 - 0.5) / (4096 - 256))

    def sampling_defaults(self, loaded):
        if loaded.extra["variant"] == "schnell":
            return SamplingDefaults(steps=4, cfg=1, shift=1, sampler="euler", guidance=None)
        return self.spec.sampling

    def presets(self):
        attention = (
            "transformer_blocks.*.attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}",
            "single_transformer_blocks.*.attn.{to_q,to_k,to_v}",
        )
        mlp = (
            "transformer_blocks.*.{ff,ff_context}.net.{0.proj,2}",
            "single_transformer_blocks.*.{proj_mlp,proj_out}",
        )
        return {
            "attn-only": TargetPreset("attn-only", attention, description="双流与单流 block 的注意力投影"),
            "attn-mlp": TargetPreset(
                "attn-mlp", attention + mlp, description="注意力与 MLP，含单流 block 的联合输出投影"
            ),
            "all-linear": TargetPreset("all-linear", ("*",), description="所有 Linear 层"),
        }

    def memory_layout(self, loaded):
        return MemoryLayout(
            blocks=[*loaded.backbone.transformer_blocks, *loaded.backbone.single_transformer_blocks],
            keep_high_precision=(
                "time_text_embed*",
                "context_embedder*",
                "x_embedder*",
                "*norm*",
                "proj_out",
            ),
        )

    def meta_backbone(self, cfg):
        config = DEFAULTS["transformer"]
        if cfg.dit_path:
            path = component_path(Path(cfg.dit_path).expanduser(), "transformer")
            config = component_config(path, "transformer")
        return create_meta("transformer", config)

    def linear_module_names(self):
        model = self.meta_backbone(ModelConfig(family="flux"))
        return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    def latent_fingerprint(self, cfg, *, dtype):
        if not cfg.dit_path:
            return self.spec.latent.fingerprint
        try:
            path = self._paths(cfg)[3]
        except ValueError:
            return self.spec.latent.fingerprint
        if not path.exists():
            return self.spec.latent.fingerprint
        return FluxLatent(path).fingerprint


register("flux", FluxFamily)
