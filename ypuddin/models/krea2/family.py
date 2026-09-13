"""Krea 2 model family: 12.9B single-stream MMDiT (``SingleStreamDiT``), Qwen3-VL-4B-Instruct conditioner,
Qwen-Image VAE (shared with Anima), rectified flow with a resolution-aware timestep shift.

Conventions verified against musubi-tuner / ComfyUI / diffusers:
* latent ``(B, 16, H, W)`` is patchified in 2x2 cells -> image tokens ``(B, h*w, 64)``; image tokens come
  first in the joint sequence, text after (text gets zero RoPE position); ``t ∈ (0,1)`` raw
* text conditioning = ``(B, L, 12, 2560)`` stacked Qwen3-VL hidden states + boolean mask
* checkpoints: bare keys (official / ComfyUI ``diffusion_models``), ``model.diffusion_model.`` prefixed
  files, and Comfy-Org ``fp8_scaled`` files (fp8 ``weight`` + ``scale_weight``) -> frozen fp8 layers
* Raw inference shift ``exp(mu)`` is resolution dependent; Turbo fixes ``mu=1.15``
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset
from ypuddin.adapters.frozen import FP8_DTYPES, FrozenLinear
from ypuddin.config import MemoryConfig, ModelConfig
from ypuddin.models.anima.family import AnimaLatent
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
from ypuddin.objectives.flow import resolution_shift_value

from .text import Krea2Text

log = logging.getLogger(__name__)

# mu interpolation endpoints of the RAW checkpoint (musubi ``timesteps``: minres 256 / maxres 1280 at 16 px per token)
MU_TOKENS = (256, 6400)
MU_RANGE = (0.5, 1.15)
SCALE_SUFFIXES = (".scale_weight", ".weight_scale")


# --------------------------------------------------------------------------- loading
def _strip_prefix(key: str) -> str:
    from .vendor.krea2_mmdit import KREA2_KEY_PREFIXES

    for p in KREA2_KEY_PREFIXES:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _read_state_dict(
    path: str | Path, dtype: torch.dtype | None
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Returns ``(weights, fp8_scales)``; fp8 weights keep their dtype, other floats are cast to ``dtype``."""
    from safetensors.torch import load_file

    weights: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    for k, v in load_file(str(path)).items():
        k = _strip_prefix(k)
        if k == "scaled_fp8":
            continue
        if k.endswith(SCALE_SUFFIXES):
            base = k.rsplit(".", 1)[0]
            scales[base] = v.reshape(()).to(torch.float32)
            continue
        if v.is_floating_point() and v.dtype not in FP8_DTYPES.values() and dtype is not None:
            v = v.to(dtype)
        weights[k] = v
    return weights, scales


def _freeze_fp8_linears(model: nn.Module, scales: dict[str, Tensor]) -> int:
    """Replace every ``nn.Linear`` whose weight arrived in fp8 by a ``FrozenLinear`` carrying the checkpoint's
    per-tensor scale (dequantized on the fly, exactly like ComfyUI consumes ``fp8_scaled`` files)."""
    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module.weight.dtype not in FP8_DTYPES.values():
            continue
        kind = next(k for k, dt in FP8_DTYPES.items() if dt == module.weight.dtype)
        scale = scales.get(name)
        if scale is None:
            raise RuntimeError(
                f"fp8 weight {name} has no scale in the checkpoint (expected {name}.scale_weight)"
            )
        frozen = FrozenLinear(
            module.weight.data, None if module.bias is None else module.bias.data, precision=kind, scale=scale
        )
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, frozen)
        n += 1
    return n


def load_dit(
    path: str | Path, *, device: torch.device | str, dtype: torch.dtype
) -> tuple[nn.Module, dict[str, Any]]:
    """Build on the meta device, assign checkpoint tensors, materialise the fp8 layers."""
    from .vendor.krea2_mmdit import SingleStreamDiT, infer_config

    weights, scales = _read_state_dict(path, dtype)
    config = infer_config(weights)
    with torch.device("meta"):
        model = SingleStreamDiT(config)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"Krea 2 checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    if unexpected:
        log.warning(
            "ignoring %d unexpected tensors in Krea 2 checkpoint, e.g. %s", len(unexpected), unexpected[:5]
        )
    if scales:
        n = _freeze_fp8_linears(model, scales)
        log.info("Krea 2: %d fp8_scaled linear layers kept in fp8 with their checkpoint scales", n)
    leftover = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if leftover:
        raise RuntimeError(f"parameters still on meta after load: {leftover[:5]}")
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model, config.__dict__.copy()


# --------------------------------------------------------------------------- family
class Krea2Family(ModelFamily):
    spec = ModelSpec(
        name="krea2",
        latent=LatentSpec(channels=16, stride=8, patch=2, fingerprint=AnimaLatent.fingerprint),
        text=TextSpec(
            max_len=512, fingerprint=Krea2Text.fingerprint, pad_floor=False, encoder_params=4_022_000_000
        ),
        sampling=SamplingDefaults(steps=28, cfg=5.5, shift=None, sampler="euler"),
        # no ``online_text``: the 4B conditioner has no business staying resident next to a 12.9B DiT
        capabilities=frozenset(
            {"block_swap", "fp8_base", "activation_checkpointing", "masked_loss", "compile"}
        ),
        architecture="krea2",
        adapter_prefix="lora_unet",
        label="Krea 2 Raw 12.9B",
        weights=(
            (
                "dit_path",
                "DiT",
                "krea2_raw_bf16.safetensors（约 26 GB）或 Comfy-Org krea2_fp8_scaled.safetensors（约 13 GB，按 fp8 加载）",
            ),
            (
                "text_encoder_path",
                "Qwen3-VL-4B-Instruct",
                "HF 目录（推荐）或 ComfyUI 单文件 qwen_3vl_4b*.safetensors（bf16 / fp8_scaled）",
            ),
            ("vae_path", "Qwen-Image VAE", "qwen_image_vae.safetensors（与 Anima 共用）"),
            ("tokenizer_path", "分词器目录", "可选的 Qwen3-VL 分词器覆盖目录"),
        ),
        optional_weights=("tokenizer_path",),
    )

    # ----------------------------------------------------------------- loading
    def validate_config(self, cfg: ModelConfig) -> list[str]:
        problems = []
        from .variants import resolve_variant

        try:
            resolve_variant(cfg.dit_path, cfg.krea2_variant)
        except ValueError as error:
            problems.append(str(error))
        for field in ("dit_path", "text_encoder_path", "vae_path"):
            value = getattr(cfg, field)
            if not value:
                problems.append(f"model.{field} is required for krea2")
            elif not Path(value).expanduser().exists():
                problems.append(f"model.{field} does not exist: {value}")
        return problems

    def training_options_errors(self, cfg):
        from .variants import TURBO_TRAINING_ERROR, resolve_variant

        problems = super().training_options_errors(cfg)
        try:
            if resolve_variant(cfg.model.dit_path, cfg.model.krea2_variant) == "turbo":
                problems.append({"loc": "model.krea2_variant", "msg": TURBO_TRAINING_ERROR})
        except ValueError as error:
            problems.append({"loc": "model.krea2_variant", "msg": str(error)})
        return problems

    def sampling_defaults(self, loaded):
        if loaded.extra.get("variant") == "turbo":
            # Official Krea sampling.py: distilled weights were trained at fixed mu=1.15.
            return SamplingDefaults(steps=8, cfg=0.0, shift=math.exp(1.15), sampler="euler")
        return self.spec.sampling

    def sampling_needs_uncond(self, loaded, cfg):
        return cfg > 0 if loaded.extra.get("variant") == "turbo" else cfg != 1.0

    def sample_latents(self, loaded, predict, shape, **options):
        if loaded.extra.get("variant") == "turbo":
            # Krea's g=0 means cond; shared Flow uses uncond+s*(cond-uncond).
            # Thus s=g+1, preserving explicit positive Turbo guidance as well.
            guidance = options.get("cfg", 0.0)
            if not math.isfinite(guidance) or guidance < 0:
                raise ValueError("Turbo guidance must be finite and nonnegative")
            options["cfg"] = guidance + 1.0
            options.setdefault("shift", self.sampling_defaults(loaded).shift)
        return super().sample_latents(loaded, predict, shape, **options)

    def latent_fingerprint(self, cfg: ModelConfig, *, dtype: torch.dtype) -> str:
        return AnimaLatent(cfg.vae_path, dtype=dtype).fingerprint

    def load(
        self,
        cfg: ModelConfig,
        memory: MemoryConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        backbone_device: torch.device | str | None = None,
    ) -> LoadedModel:
        from .variants import resolve_variant

        problems = self.validate_config(cfg)
        if problems:
            raise FileNotFoundError("; ".join(problems))
        dit, config = load_dit(cfg.dit_path, device=backbone_device or device, dtype=dtype)
        if memory.activation_checkpointing != "none":
            if memory.activation_checkpointing == "unsloth":
                log.warning(
                    "krea2: unsloth offloaded checkpointing is not available for this family, using block checkpointing"
                )
            dit.enable_gradient_checkpointing()
        dit.attn_mode = self.resolve_attention(cfg.attention, device)
        text = Krea2Text(
            cfg.text_encoder_path,
            tokenizer_path=cfg.tokenizer_path,
            dtype=dtype,
            device=device,
            txtlayers=config["txtlayers"],
        )
        if text.hidden_size != config["txtdim"]:
            raise ValueError(
                f"text encoder hidden size {text.hidden_size} does not match the DiT text width {config['txtdim']} "
                "(Krea 2 needs Qwen3-VL-4B-Instruct)"
            )
        latent = AnimaLatent(
            cfg.vae_path, device=device, dtype=torch.float32 if torch.device(device).type == "cpu" else dtype
        )
        log.info(
            "loaded Krea 2 DiT: features=%s layers=%s heads=%s/%s",
            config["features"],
            config["layers"],
            config["heads"],
            config["kvheads"],
        )
        return LoadedModel(
            backbone=dit,
            text=text,
            latent=latent,
            device=torch.device(device),
            dtype=dtype,
            extra={"dit_config": config, "variant": resolve_variant(cfg.dit_path, cfg.krea2_variant)},
        )

    @staticmethod
    def resolve_attention(requested: str, device: torch.device | str) -> str:
        from ypuddin.models.anima.family import AnimaFamily

        return AnimaFamily.resolve_attention(requested, device)

    # ----------------------------------------------------------------- forward
    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        from ypuddin.models.anima.vendor.attention import sampling_attention

        dit = loaded.backbone
        patch = int(dit.config.patch)
        b, c, h, w = x_t.shape
        if h % patch or w % patch:
            raise ValueError(f"latent {h}x{w} is not a multiple of the patch size {patch}")
        hp, wp = h // patch, w // patch
        img = (
            x_t.reshape(b, c, hp, patch, wp, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, hp * wp, c * patch * patch)
        )
        device = x_t.device
        img_pos = torch.zeros(hp, wp, 3, device=device, dtype=torch.float32)
        img_pos[..., 1] = torch.arange(hp, device=device, dtype=torch.float32)[:, None]
        img_pos[..., 2] = torch.arange(wp, device=device, dtype=torch.float32)[None, :]
        img_pos = img_pos.reshape(1, hp * wp, 3).expand(b, -1, -1)
        context = cond["embeds"].to(device=device, dtype=img.dtype)
        txt_mask = cond["attn_mask"].to(device=device, dtype=torch.bool)
        pos = torch.cat(
            [img_pos, torch.zeros(b, context.shape[1], 3, device=device, dtype=torch.float32)], dim=1
        )
        mask = torch.cat([torch.ones(b, hp * wp, device=device, dtype=torch.bool), txt_mask], dim=1)
        with sampling_attention(not dit.training):
            out = dit(
                img=img, context=context, t=t.to(device=device, dtype=torch.float32), pos=pos, mask=mask
            )
        return out.reshape(b, hp, wp, c, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

    # ----------------------------------------------------------------- sampling
    def sampling_shift(self, num_tokens: int, objective: Any | None = None) -> float:
        tokens, mu = MU_TOKENS, MU_RANGE
        if objective is not None and getattr(objective, "timestep_sampling", None) == "resolution_shift":
            tokens, mu = tuple(objective.res_shift_tokens), tuple(objective.res_shift_mu)
        return resolution_shift_value(num_tokens, tokens[0], tokens[1], mu[0], mu[1])

    # ----------------------------------------------------------------- adapters / memory
    def presets(self) -> dict[str, TargetPreset]:
        attn = ("blocks.*.attn.{wq,wk,wv,gate,wo}",)
        mlp = ("blocks.*.mlp.{gate,up,down}",)
        text = (
            "txtfusion.*_blocks.*.attn.{wq,wk,wv,gate,wo}",
            "txtfusion.*_blocks.*.mlp.{gate,up,down}",
            "txtfusion.projector",
            "txtmlp.*",
        )
        return {
            "all-linear": TargetPreset(
                "all-linear",
                include=("*",),
                description="全部 264 个 Linear（Krea 官方默认：rank 32 / alpha 32）",
            ),
            "attn-mlp": TargetPreset(
                "attn-mlp",
                include=attn + mlp,
                description="28 个主 block 的注意力 + SwiGLU（不含文本融合层与嵌入/输出层）",
            ),
            "attn-only": TargetPreset(
                "attn-only",
                include=attn,
                description="仅主 block 注意力投影（官方对长时间训练的建议：更好保持提示词遵循）",
            ),
            "attn-mlp-text": TargetPreset(
                "attn-mlp-text",
                include=attn + mlp + text,
                description="主 block + 文本融合 transformer + 文本 MLP",
            ),
        }

    def default_preset(self) -> str:
        return "attn-mlp"

    def memory_layout(self, loaded: LoadedModel) -> MemoryLayout:
        return self.memory_layout_meta(loaded.backbone)

    def memory_layout_meta(self, backbone: nn.Module) -> MemoryLayout:
        blocks = list(backbone.blocks)
        nbytes = sum(p.numel() * p.element_size() for p in blocks[0].parameters()) if blocks else 0
        return MemoryLayout(
            blocks=blocks,
            keep_high_precision=(
                "first*",
                "last*",
                "tmlp*",
                "tproj*",
                "txtfusion*",
                "txtmlp*",
                "*norm*",
                "*mod*",
            ),
            block_param_bytes=nbytes,
        )

    def meta_backbone(self, cfg: ModelConfig) -> nn.Module:
        from .vendor.krea2_mmdit import KREA2_CONFIG, SingleStreamDiT, infer_config

        config = KREA2_CONFIG
        if cfg.dit_path and Path(cfg.dit_path).expanduser().exists():
            try:
                from safetensors import safe_open

                shapes: dict[str, Any] = {}
                with safe_open(str(Path(cfg.dit_path).expanduser()), framework="pt") as f:
                    for k in f.keys():
                        kk = _strip_prefix(k)
                        if kk == "scaled_fp8" or kk.endswith(SCALE_SUFFIXES):
                            continue
                        shapes[kk] = f.get_slice(k)
                config = infer_config(shapes)
            except Exception as e:  # noqa: BLE001
                log.warning("could not infer Krea 2 geometry from %s: %s", cfg.dit_path, e)
        return SingleStreamDiT(config)

    def linear_module_names(self) -> list[str]:
        from .vendor.krea2_mmdit import KREA2_CONFIG, SingleStreamDiT

        with torch.device("meta"):
            model = SingleStreamDiT(KREA2_CONFIG)
        return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


register("krea2", Krea2Family)
