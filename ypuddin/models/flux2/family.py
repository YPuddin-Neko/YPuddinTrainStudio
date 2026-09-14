"""FLUX.2 Klein base text-to-image adapter training (local weights only).

The VAE produces normalized, already patchified BCHW latents (128 channels,
stride 16). The transformer packs only the spatial axes and receives unit flow
time. Editing/reference images and distilled Klein/KV models are not supported.
"""

from pathlib import Path

import torch
from safetensors import SafetensorError
from torch import nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import ModelConfig
from ypuddin.models.base import (
    LatentSpec,
    LoadedModel,
    MemoryLayout,
    ModelFamily,
    ModelSpec,
    SamplingDefaults,
    TextSpec,
)
from ypuddin.models.registry import register

from .latent import Flux2Latent
from .loading import (
    DEV_UNSUPPORTED,
    VARIANTS,
    component,
    load_transformer,
    reject_dev_config,
    reject_dev_weights,
    resolve_variant,
    shapes,
    transformer_config,
)
from .text import Flux2Text


def image_ids(x):
    b, _, h, w = x.shape
    ids = torch.zeros(h, w, 4, device=x.device)
    ids[:, :, 1] = torch.arange(h, device=x.device)[:, None]
    ids[:, :, 2] = torch.arange(w, device=x.device)[None, :]
    return ids.reshape(1, h * w, 4).expand(b, -1, -1)


def text_ids(embeds):
    b, length, _ = embeds.shape
    ids = torch.zeros(b, length, 4, device=embeds.device)
    ids[:, :, 3] = torch.arange(length, device=embeds.device)
    return ids


class Flux2Family(ModelFamily):
    spec = ModelSpec(
        name="flux2",
        label="FLUX.2 Klein",
        latent=LatentSpec(128, 16, 1, "flux2-vae-mode-fp32-patch2-bn-v1"),
        text=TextSpec(512, "flux2-variant-hidden-layers-v1", encoder_params=4_000_000_000),
        sampling=SamplingDefaults(steps=50, cfg=4.0, shift=None, sampler="euler"),
        capabilities=frozenset({"activation_checkpointing", "masked_loss", "block_swap"}),
        architecture="flux2",
        adapter_prefix="lora_transformer",
        weights=(
            ("dit_path", "Klein base DiT", "Klein base 4B/9B 完整本地 HF 目录或原始 BFL safetensors"),
            (
                "text_encoder_path",
                "文本编码器",
                "完整模型目录可留空；单独 DiT 须选择 Qwen3 本地 HF 目录：Klein 4B 使用 Qwen3-4B，Klein 9B 使用 Qwen3-8B",
            ),
            (
                "vae_path",
                "FLUX.2 VAE",
                "完整模型目录可留空；单独 DiT 须选择含 batchnorm 统计的 FLUX.2 专用 VAE",
            ),
            (
                "tokenizer_path",
                "Qwen3 分词器",
                "模型或文本编码器目录内含时可留空；否则选择本地 Qwen3 tokenizer 与原始 chat template",
            ),
        ),
        optional_weights=("text_encoder_path", "vae_path", "tokenizer_path"),
        directory_only_weights=("text_encoder_path",),
    )

    @staticmethod
    def _paths(cfg):
        root = Path(cfg.dit_path or ".").expanduser()
        text = component(root, "text_encoder", cfg.text_encoder_path)
        tokenizer = (
            Path(cfg.tokenizer_path).expanduser()
            if cfg.tokenizer_path
            else (root / "tokenizer" if (root / "tokenizer").is_dir() else text)
        )
        return root, component(root, "transformer"), text, component(root, "vae", cfg.vae_path), tokenizer

    def validate_config(self, cfg):
        if getattr(cfg, "flux2_variant", "auto") == "dev":
            return [DEV_UNSUPPORTED]
        if not cfg.dit_path:
            return ["model.dit_path is required for FLUX.2"]
        problems = []
        if cfg.attention not in {"auto", "sdpa", "flash_attn", "xformers"}:
            problems.append("FLUX.2 supports SDPA, FlashAttention or xFormers; Sage training is unsupported")
        try:
            root = Path(cfg.dit_path).expanduser()
            dit = component(root, "transformer")
            config = transformer_config(dit)
            resolve_variant(root, config, getattr(cfg, "flux2_variant", "auto"))
            reject_dev_weights(dit)
            _, _, text, vae, tokenizer = self._paths(cfg)
            for path in (text, vae):
                shapes(path)
            if not (text / "config.json").is_file() or not (tokenizer / "tokenizer_config.json").is_file():
                problems.append("FLUX.2 text encoder/tokenizer local directory is incomplete")
        except (OSError, ValueError, KeyError, TypeError, SafetensorError) as error:
            problems.append(str(error))
        return problems

    def training_options_errors(self, cfg):
        if getattr(cfg.model, "flux2_variant", "auto") == "dev":
            return [{"loc": "model.flux2_variant", "msg": DEV_UNSUPPORTED}]
        errors = super().training_options_errors(cfg)
        if cfg.memory.activation_checkpointing == "unsloth":
            errors.append(
                {
                    "loc": "memory.activation_checkpointing",
                    "msg": "FLUX.2 supports none or block checkpointing",
                }
            )
        if cfg.memory.blocks_to_swap and cfg.memory.activation_checkpointing != "block":
            errors.append(
                {
                    "loc": "memory.activation_checkpointing",
                    "msg": "FLUX.2 block swap requires block checkpointing",
                }
            )
        return errors

    def load(self, cfg, memory, *, device, dtype, backbone_device=None):
        problems = self.validate_config(cfg)
        if problems:
            raise ValueError("; ".join(problems))
        if (
            memory.base_precision.startswith("fp8")
            or memory.compile
            or memory.activation_checkpointing == "unsloth"
        ):
            raise ValueError("FLUX.2 FP8 conversion, compile and unsloth checkpointing are unsupported")
        if memory.blocks_to_swap and memory.activation_checkpointing != "block":
            raise ValueError("FLUX.2 block swap requires block checkpointing")
        root, dit, text_path, vae, tokenizer = self._paths(cfg)
        config = transformer_config(dit)
        variant = resolve_variant(root, config, getattr(cfg, "flux2_variant", "auto"))
        text = Flux2Text(text_path, tokenizer, variant=variant, dtype=dtype, device=device)
        if text.hidden_size != config.get("joint_attention_dim", 7680):
            raise ValueError("FLUX.2 text encoder hidden width does not match the selected transformer")
        from diffusers import Flux2Transformer2DModel

        with torch.device("meta"):
            backbone = Flux2Transformer2DModel.from_config(config)
        return LoadedModel(
            backbone,
            text,
            Flux2Latent(vae, device=device),
            torch.device(device),
            dtype,
            extra={
                "dit_config": config,
                "variant": variant,
                "dit_path": dit,
                "backbone_device": backbone_device or device,
                "checkpointing": memory.activation_checkpointing == "block",
                "attention": cfg.attention,
                "text_encoder_weight_elements": text.weight_elements,
                "materialized": False,
            },
        )

    def sampling_defaults(self, loaded):
        return self.spec.sampling

    def sampling_shift(self, num_tokens, objective=None):
        return self._sampling_shift(num_tokens, 50)

    def sampling_shift_for_model(self, loaded, num_tokens, objective=None, *, steps=None):
        return self._sampling_shift(num_tokens, steps or 50)

    @staticmethod
    def _sampling_shift(num_tokens, steps):
        import math

        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

        return math.exp(compute_empirical_mu(num_tokens, steps))

    def materialize_backbone(self, loaded):
        reject_dev_config(loaded.extra["dit_config"], loaded.extra.get("variant", "auto"))
        if loaded.extra["materialized"]:
            return
        # The native trainer has cached and unloaded these by this phase. Direct
        # callers get the same bounded residency contract.
        loaded.text.unload()
        loaded.latent.unload()
        model = load_transformer(
            loaded.extra["dit_path"],
            loaded.extra["dit_config"],
            device=loaded.extra["backbone_device"],
            dtype=loaded.dtype,
        )
        if loaded.extra["checkpointing"]:
            model.enable_gradient_checkpointing()
        attention = loaded.extra["attention"]
        backend = {"auto": "native", "sdpa": "native", "flash_attn": "flash", "xformers": "xformers"}[
            attention
        ]
        if attention in {"flash_attn", "xformers"} and loaded.device.type != "cuda":
            raise ValueError(f"FLUX.2 {attention} requires CUDA")
        model.set_attention_backend(backend)
        loaded.backbone = model
        loaded.extra["materialized"] = True

    def forward(self, loaded, x_t, t, cond, **extra):
        reject_dev_config(loaded.extra["dit_config"], loaded.extra.get("variant", "auto"))
        if not loaded.extra.get("materialized", False):
            raise RuntimeError("Call materialize_backbone after caching and before FLUX.2 forward")
        b, c, h, w = x_t.shape
        if c != 128:
            raise ValueError("FLUX.2 expects normalized patchified 128-channel latents")
        embeds = cond["embeds"].to(x_t.device, loaded.dtype)
        prediction = loaded.backbone(
            hidden_states=x_t.flatten(2).transpose(1, 2),
            encoder_hidden_states=embeds,
            timestep=t.to(x_t.device, torch.float32),
            img_ids=image_ids(x_t),
            txt_ids=text_ids(embeds),
            guidance=None,
            return_dict=False,
        )[0]
        return prediction.transpose(1, 2).reshape(b, c, h, w)

    def presets(self):
        attn = (
            "transformer_blocks.*.attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}",
            "single_transformer_blocks.*.attn.{to_qkv_mlp_proj,to_out}",
        )
        mlp = ("transformer_blocks.*.{ff,ff_context}.{linear_in,linear_out}",)
        return {
            "attn-only": TargetPreset(
                "attn-only", attn, description="训练注意力投影；单流模块的投影同时包含融合的前馈部分。"
            ),
            "attn-mlp": TargetPreset("attn-mlp", attn + mlp, description="训练双流和单流模块的注意力及前馈层。"),
        }

    def memory_layout(self, loaded):
        return MemoryLayout(
            blocks=[*loaded.backbone.transformer_blocks, *loaded.backbone.single_transformer_blocks],
            keep_high_precision=(
                "x_embedder",
                "context_embedder",
                "time_guidance_embed.*",
                "*stream_modulation*",
                "norm_out.*",
                "proj_out",
            ),
        )

    def latent_fingerprint(self, cfg, *, dtype):
        try:
            path = component(cfg.dit_path or ".", "vae", cfg.vae_path)
        except ValueError:
            return self.spec.latent.fingerprint
        return Flux2Latent(path).fingerprint if path.exists() else self.spec.latent.fingerprint

    def meta_backbone(self, cfg):
        from diffusers import Flux2Transformer2DModel

        requested = getattr(cfg, "flux2_variant", "auto")
        if requested == "dev":
            raise ValueError(DEV_UNSUPPORTED)
        if cfg.dit_path:
            root = Path(cfg.dit_path).expanduser()
            dit = component(root, "transformer")
            config = transformer_config(dit)
            resolve_variant(root, config, requested)
            reject_dev_weights(dit)
        else:
            config = dict(VARIANTS["klein-base-4b" if requested == "auto" else requested])
        return Flux2Transformer2DModel.from_config(config)

    def linear_module_names(self):
        with torch.device("meta"):
            model = self.meta_backbone(ModelConfig(family="flux2"))
        return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


register("flux2", Flux2Family)
