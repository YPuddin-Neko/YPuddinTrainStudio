"""Anima model family: Cosmos-Predict2 style MiniTrainDIT + LLM adapter, Qwen3-0.6B text encoder,
Qwen-Image (Wan2.1 architecture) VAE, rectified flow.

Conventions verified against sd-scripts / diffusion-pipe / AnimaLoraStudio:
* backbone input ``(B, C=16, T=1, H, W)``, ``t ∈ (0,1)`` fp32 shape ``(B,)``, output same shape as input
* text conditioning = Qwen3 hidden states (padding zeroed) + old-T5 token ids consumed by the LLM
  adapter inside the backbone; ``padding_mask`` is all zeros ``(B, 1, h, w)``
* DiT geometry is inferred from the checkpoint (``x_embedder.proj.1.weight`` -> width, ``blocks.N`` count)
* checkpoint key prefixes ``net.`` (anima-base) and ``model.diffusion_model.`` (ComfyUI) are stripped
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import MemoryConfig, ModelConfig
from ypuddin.models.base import (
    LatentPipeline,
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

from .text import AnimaText

log = logging.getLogger(__name__)

PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.")


# --------------------------------------------------------------------------- DiT loading
def _read_state_dict(path: str | Path, dtype: torch.dtype | None = None) -> dict[str, Tensor]:
    from safetensors.torch import load_file

    sd = load_file(str(path))
    out: dict[str, Tensor] = {}
    for k, v in sd.items():
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p) :]
                break
        if dtype is not None and v.is_floating_point():
            v = v.to(dtype)
        out[k] = v
    return out


def build_dit(config: dict[str, Any]) -> nn.Module:
    from .vendor.cosmos_dit import Anima

    return Anima(**config)


def infer_config(state_dict: dict[str, Tensor]) -> dict[str, Any]:
    from .vendor.cosmos_dit import infer_dit_config

    return infer_dit_config(state_dict)


def default_config() -> dict[str, Any]:
    from .vendor.cosmos_dit import ANIMA_2B_CONFIG

    return dict(ANIMA_2B_CONFIG)


def load_dit(
    path: str | Path, *, device: torch.device | str, dtype: torch.dtype
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the DiT on the meta device (no 8 GB fp32 CPU allocation), then assign checkpoint tensors."""
    sd = _read_state_dict(path, dtype)
    config = infer_config(sd)
    with torch.device("meta"):
        model = build_dit(config)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"Anima checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    if unexpected:
        log.warning(
            "ignoring %d unexpected tensors in Anima checkpoint, e.g. %s", len(unexpected), unexpected[:5]
        )
    model = _materialize_meta_buffers(model, device)
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model, config


def _materialize_meta_buffers(model: nn.Module, device: torch.device | str) -> nn.Module:
    """Recreate the non-persistent RoPE tables that stay on ``meta`` after ``load_state_dict(assign=True)``.

    Both buffer families are deterministic functions of constructor arguments, so recomputing them
    here is exactly what ``__init__`` would have produced on a real device.
    """
    from .vendor.cosmos_dit import AdapterRotaryEmbedding, VideoRopePosition3DEmb

    device = torch.device(device)
    for module in model.modules():
        if isinstance(module, VideoRopePosition3DEmb):
            dim_h, dim_t = module._dim_h, module._dim_t
            module.seq = torch.arange(
                max(module.max_h, module.max_w, module.max_t), dtype=torch.float, device=device
            )
            module.dim_spatial_range = (
                torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h
            )
            module.dim_temporal_range = (
                torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t
            )
        elif isinstance(module, AdapterRotaryEmbedding):
            head_dim = module.inv_freq.shape[0] * 2
            module.inv_freq = 1.0 / (
                module.rope_theta
                ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim)
            )
    leftover = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if leftover:
        raise RuntimeError(f"buffers still on meta after load: {leftover[:5]}")
    return model


# --------------------------------------------------------------------------- VAE
class AnimaLatent(LatentPipeline):
    fingerprint = "anima-qwen-image-vae-f8c16-v1"
    channels = 16
    stride = 8

    def __init__(
        self,
        vae_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_2d: bool = True,
    ):
        self.path = Path(vae_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_2d = use_2d
        self.vae: nn.Module | None = None

    def _ensure(self) -> nn.Module:
        if self.vae is None:
            if self.use_2d:
                from .vendor.qwen_image_vae_2d import load_vae
            else:
                from .vendor.qwen_image_vae import load_vae
            vae = load_vae(str(self.path), device="cpu")
            vae = vae.to(device=self.device, dtype=self.dtype)
            vae.requires_grad_(False)
            vae.eval()
            self.vae = vae
        return self.vae

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.vae is not None:
            self.vae.to(self.device)

    def unload(self) -> None:
        self.vae = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        vae = self._ensure()
        return vae.encode_pixels_to_latents(pixels.to(self.device, self.dtype)).float()

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        vae = self._ensure()
        return vae.decode_to_pixels(latents.to(self.device, self.dtype)).float().clamp(-1, 1)


# --------------------------------------------------------------------------- family
class AnimaFamily(ModelFamily):
    spec = ModelSpec(
        name="anima",
        latent=LatentSpec(channels=16, stride=8, patch=2, fingerprint=AnimaLatent.fingerprint),
        text=TextSpec(max_len=512, fingerprint=AnimaText.fingerprint, pad_floor=True),
        sampling=SamplingDefaults(steps=25, cfg=4.0, shift=3.0, sampler="euler"),
        capabilities=frozenset(
            {
                "block_swap",
                "fp8_base",
                "activation_checkpointing",
                "online_text",
                "llm_adapter",
                "masked_loss",
                "compile",
            }
        ),
        architecture="anima",
        adapter_prefix="lora_unet",
    )

    # ----------------------------------------------------------------- loading
    def validate_config(self, cfg: ModelConfig) -> list[str]:
        problems = []
        for field in ("dit_path", "text_encoder_path", "vae_path"):
            value = getattr(cfg, field)
            if not value:
                problems.append(f"model.{field} is required for anima")
            elif not Path(value).expanduser().exists():
                problems.append(f"model.{field} does not exist: {value}")
        return problems

    def load(
        self, cfg: ModelConfig, memory: MemoryConfig, *, device: torch.device | str, dtype: torch.dtype
    ) -> LoadedModel:
        problems = self.validate_config(cfg)
        if problems:
            raise FileNotFoundError("; ".join(problems))
        dit, config = load_dit(cfg.dit_path, device=device, dtype=dtype)
        if memory.activation_checkpointing != "none" and hasattr(dit, "enable_gradient_checkpointing"):
            dit.enable_gradient_checkpointing()
        text = AnimaText(cfg.text_encoder_path, tokenizer_path=cfg.tokenizer_path, dtype=dtype, device=device)
        latent = AnimaLatent(
            cfg.vae_path, device=device, dtype=torch.float32 if torch.device(device).type == "cpu" else dtype
        )
        log.info(
            "loaded Anima DiT: width=%s blocks=%s heads=%s",
            config.get("model_channels"),
            config.get("num_blocks"),
            config.get("num_heads"),
        )
        return LoadedModel(
            backbone=dit,
            text=text,
            latent=latent,
            device=torch.device(device),
            dtype=dtype,
            extra={"dit_config": config},
        )

    # ----------------------------------------------------------------- forward
    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        dit = loaded.backbone
        b, _, h, w = x_t.shape
        x5 = x_t.unsqueeze(2)
        padding_mask = torch.zeros(b, 1, h, w, dtype=x5.dtype, device=x5.device)
        out = dit(
            x5,
            t.to(device=x5.device, dtype=torch.float32),
            cond["embeds"].to(device=x5.device, dtype=x5.dtype),
            padding_mask=padding_mask,
            target_input_ids=cond["t5_ids"].to(x5.device),
            target_attention_mask=cond["t5_mask"].to(x5.device),
            source_attention_mask=cond["attn_mask"].to(x5.device),
        )
        return out.squeeze(2)

    # ----------------------------------------------------------------- adapters / memory
    def presets(self) -> dict[str, TargetPreset]:
        attn = (
            "blocks.*.self_attn.{q_proj,k_proj,v_proj,output_proj}",
            "blocks.*.cross_attn.{q_proj,k_proj,v_proj,output_proj}",
        )
        mlp = ("blocks.*.mlp.layer1", "blocks.*.mlp.layer2")
        adaln = ("blocks.*.adaln_modulation_*.*",)
        adapter = (
            "llm_adapter.blocks.*.self_attn.*_proj",
            "llm_adapter.blocks.*.cross_attn.*_proj",
            "llm_adapter.blocks.*.mlp.*",
        )
        return {
            "attn-mlp": TargetPreset(
                "attn-mlp",
                include=attn + mlp,
                description="DiT 注意力 + MLP（默认，与 AnimaLoraStudio 一致）",
            ),
            "attn-only": TargetPreset("attn-only", include=attn, description="仅 DiT 注意力投影"),
            "full-linear": TargetPreset(
                "full-linear", include=attn + mlp + adaln, description="DiT 内全部 Linear（含 AdaLN 调制）"
            ),
            "with-adapter": TargetPreset(
                "with-adapter", include=attn + mlp + adapter, description="DiT 注意力 + MLP + LLM Adapter"
            ),
            "adapter-only": TargetPreset("adapter-only", include=adapter, description="仅 LLM Adapter"),
        }

    def memory_layout(self, loaded: LoadedModel) -> MemoryLayout:
        return self.memory_layout_meta(loaded.backbone)

    def memory_layout_meta(self, backbone: nn.Module) -> MemoryLayout:
        blocks = list(backbone.blocks)
        nbytes = sum(p.numel() * p.element_size() for p in blocks[0].parameters()) if blocks else 0
        return MemoryLayout(
            blocks=blocks,
            keep_high_precision=(
                "x_embedder*",
                "t_embedder*",
                "t_embedding_norm*",
                "final_layer*",
                "llm_adapter.embed*",
                "*norm*",
            ),
            block_param_bytes=nbytes,
        )

    def meta_backbone(self, cfg: ModelConfig) -> nn.Module:
        """Backbone on the meta device for planning: real geometry if the checkpoint is readable."""
        config = default_config()
        if cfg.dit_path and Path(cfg.dit_path).expanduser().exists():
            try:
                from safetensors import safe_open

                shapes: dict[str, Tensor] = {}
                with safe_open(str(Path(cfg.dit_path).expanduser()), framework="pt") as f:
                    for k in f.keys():
                        kk = k
                        for p in PREFIXES:
                            if kk.startswith(p):
                                kk = kk[len(p) :]
                                break
                        shapes[kk] = f.get_slice(k)
                config = _infer_from_shapes(shapes, config)
            except Exception as e:  # noqa: BLE001
                log.warning("could not infer Anima geometry from %s: %s", cfg.dit_path, e)
        return build_dit(config)

    def linear_module_names(self) -> list[str]:
        with torch.device("meta"):
            model = build_dit(default_config())
        return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def _infer_from_shapes(shapes: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Geometry from safetensors slices (no tensor data read)."""
    cfg = dict(base)
    xe = shapes.get("x_embedder.proj.1.weight")
    if xe is not None:
        out_dim, in_dim = xe.get_shape()
        cfg["model_channels"] = int(out_dim)
        cfg["in_channels"] = int(in_dim) // 4 - 1
    n_blocks = 0
    for k in shapes:
        if k.startswith("blocks."):
            try:
                n_blocks = max(n_blocks, int(k.split(".")[1]) + 1)
            except ValueError:
                continue
    if n_blocks:
        cfg["num_blocks"] = n_blocks
    heads = {2048: 16, 5120: 40}
    cfg["num_heads"] = heads.get(
        cfg.get("model_channels", 2048), max(1, cfg.get("model_channels", 2048) // 128)
    )
    cfg["use_llm_adapter"] = any(k.startswith("llm_adapter.") for k in shapes)
    return cfg


register("anima", AnimaFamily)
