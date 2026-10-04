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
import math
import time
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
from ypuddin.models.memory import release_model_memory
from ypuddin.models.registry import register

from .checkpoint import check_unquantized_checkpoint
from .text import AnimaText, require_qwen3_runtime

log = logging.getLogger(__name__)

PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.")


# --------------------------------------------------------------------------- DiT loading
def _read_state_dict(path: str | Path, dtype: torch.dtype | None = None) -> dict[str, Tensor]:
    from safetensors.torch import load_file

    check_unquantized_checkpoint(path, "Anima DiT")
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
    # einops initializes Dynamo when this vendor module is first imported.
    # Its real-scalar probes must not inherit the planner's meta context.
    with torch.device("cpu"):
        from .vendor.cosmos_dit import Anima

    return Anima(**config)


def infer_config(state_dict: dict[str, Tensor]) -> dict[str, Any]:
    with torch.device("cpu"):
        from .vendor.cosmos_dit import infer_dit_config

    return infer_dit_config(state_dict)


def default_config() -> dict[str, Any]:
    with torch.device("cpu"):
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
        vae_attention_chunking: bool = False,
        vae_tiling: bool = False,
        cache_encode_tiled: bool = False,
    ):
        self.path = Path(vae_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_2d = use_2d
        self.vae_attention_chunking = vae_attention_chunking
        self.vae_tiling = vae_tiling
        self.cache_encode_tiled = cache_encode_tiled
        self.vae: nn.Module | None = None
        self._loaded_once = False
        from ypuddin.models.fingerprints import content_fingerprint

        self.fingerprint = content_fingerprint(
            [self.path], namespace=f"{AnimaLatent.fingerprint}:2d={use_2d}:dtype={dtype}"
        )
        from ypuddin.models.vae_tiling import tiled_latent_fingerprint

        self.fingerprint = tiled_latent_fingerprint(
            self.fingerprint, vae_tiling=vae_tiling, cache_encode_tiled=cache_encode_tiled,
        )

    def _ensure(self) -> nn.Module:
        if self.vae is None:
            if self.use_2d:
                from .vendor.qwen_image_vae_2d import load_vae
            else:
                from .vendor.qwen_image_vae import load_vae
            started = time.perf_counter()
            # The loader describes the file the first time. Training releases the VAE after
            # caching and loads it again to decode each round of previews; repeating those
            # lines every round only buries the sampling log.
            vendor_log = logging.getLogger(__name__.rpartition(".")[0] + ".vendor")
            level = vendor_log.level
            if self._loaded_once:
                vendor_log.setLevel(logging.WARNING)
            try:
                # Loading frozen VAE weights initializes temporary CPU parameters.
                # A cold resume can reach this lazy load after restoring training
                # RNG; model construction must not advance that saved stream.
                with torch.random.fork_rng(devices=[]):
                    if self.use_2d:
                        vae = load_vae(
                            str(self.path), device="cpu", attention_chunking=self.vae_attention_chunking
                        )
                    else:
                        vae = load_vae(str(self.path), device="cpu")
            finally:
                vendor_log.setLevel(level)
            vae = vae.to(device=self.device, dtype=self.dtype)
            vae.requires_grad_(False)
            vae.eval()
            self.vae = vae
            if self._loaded_once:
                log.debug("reloaded the VAE in %.1fs", time.perf_counter() - started)
            self._loaded_once = True
        return self.vae

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.vae is not None:
            self.vae.to(self.device)

    def unload(self) -> None:
        if self.vae is None:
            return
        self.vae = None
        release_model_memory(self.device)

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        from ypuddin.models.vae_tiling import VAE_TILE_PIXELS

        return self._encode(pixels, tile=VAE_TILE_PIXELS if self.vae_tiling else None)

    @torch.no_grad()
    def encode_for_cache(self, pixels: Tensor) -> Tensor:
        from ypuddin.models.vae_tiling import CACHE_TILE_PIXELS, CACHE_TILE_THRESHOLD

        if self.cache_encode_tiled and pixels.shape[-2] * pixels.shape[-1] > CACHE_TILE_THRESHOLD:
            return self._encode(pixels, tile=CACHE_TILE_PIXELS)
        return self.encode(pixels)

    def _encode(self, pixels: Tensor, *, tile: int | None) -> Tensor:
        from ypuddin.models.vae_tiling import TILE_OVERLAP_PIXELS, spatial_tiled_apply

        with torch.autocast(device_type=self.device.type, enabled=False):
            vae = self._ensure()
            pixels = pixels.to(self.device, self.dtype)
            if tile is None:
                return vae.encode_pixels_to_latents(pixels).float()
            return spatial_tiled_apply(
                pixels, vae.encode_pixels_to_latents, tile=tile,
                overlap=TILE_OVERLAP_PIXELS, scale_den=self.stride,
            ).float()

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        with torch.autocast(device_type=self.device.type, enabled=False):
            vae = self._ensure()
            return self._decode_spatial(
                latents.to(self.device, self.dtype), vae.decode_to_pixels,
            ).float().clamp(-1, 1)

    def _decode_spatial(self, latents: Tensor, operation) -> Tensor:
        if not self.vae_tiling:
            return operation(latents)
        from ypuddin.models.vae_tiling import TILE_OVERLAP_PIXELS, VAE_TILE_PIXELS, spatial_tiled_apply

        return spatial_tiled_apply(
            latents, operation, tile=VAE_TILE_PIXELS // self.stride,
            overlap=TILE_OVERLAP_PIXELS // self.stride, scale_num=self.stride,
        )

    @torch.no_grad()
    def decode_comfy(self, latents: Tensor) -> Tensor:
        """Undo Wan21 normalization before casting to the VAE's compute precision."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            vae = self._ensure()
            raw = latents.to(self.device)
            shape = (1, self.channels, *([1] * (raw.ndim - 2)))
            mean = torch.tensor(vae.latents_mean, device=raw.device, dtype=raw.dtype).view(shape)
            std = torch.tensor(vae.latents_std, device=raw.device, dtype=raw.dtype).view(shape)
            raw = (raw * std + mean).to(self.dtype)
            image_only = not self.use_2d and raw.ndim == 4
            if image_only:
                raw = raw.unsqueeze(2)
            pixels = self._decode_spatial(raw, lambda tile: vae.decode(tile, return_dict=False)[0])
            if image_only:
                pixels = pixels.squeeze(2)
            return pixels.float().clamp(-1, 1)


# --------------------------------------------------------------------------- family
class AnimaFamily(ModelFamily):
    spec = ModelSpec(
        name="anima",
        attention_backends=("auto", "sdpa", "xformers", "flash_attn", "metal_flash"),
        latent=LatentSpec(channels=16, stride=8, patch=2, fingerprint=AnimaLatent.fingerprint),
        text=TextSpec(
            max_len=512, fingerprint=AnimaText.fingerprint, pad_floor=True, encoder_params=596_049_920
        ),
        sampling=SamplingDefaults(steps=25, cfg=4.0, shift=3.0, sampler="euler"),
        activation_units=34.0,
        checkpointing_modes=("none", "block", "unsloth"),
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
        label="Anima 2B",
        weights=(
            (
                "dit_path",
                "DiT",
                "anima-base / anima-preview.safetensors（键名 net.* 或 model.diffusion_model.*）",
            ),
            ("text_encoder_path", "Qwen3-0.6B", "HF 目录或单文件 safetensors"),
            ("vae_path", "Qwen-Image VAE", "qwen_image_vae.safetensors"),
            ("tokenizer_path", "分词器目录", "可选的旧版 T5 分词器；留空使用内置文件"),
        ),
        optional_weights=("tokenizer_path",),
    )

    # ----------------------------------------------------------------- loading
    def validate_config(self, cfg: ModelConfig) -> list[str]:
        from safetensors import SafetensorError

        problems = []
        for field in ("dit_path", "text_encoder_path", "vae_path"):
            value = getattr(cfg, field)
            if not value:
                problems.append(f"model.{field} is required for anima")
            elif not Path(value).expanduser().exists():
                problems.append(f"model.{field} does not exist: {value}")
            elif field in {"dit_path", "text_encoder_path"}:
                try:
                    check_unquantized_checkpoint(
                        value, "Anima DiT" if field == "dit_path" else "Anima 文字编码器"
                    )
                except (OSError, ValueError, KeyError, TypeError, SafetensorError) as error:
                    problems.append(f"model.{field}: {error}")
        return problems

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
        if cfg.attention == "metal_flash":
            from ypuddin.models.metal_attention import require_metal_flash

            require_metal_flash(device)
        problems = self.validate_config(cfg)
        if problems:
            raise FileNotFoundError("; ".join(problems))
        require_qwen3_runtime()
        # The trainer stages the backbone on CPU while VAE/text caches are built.
        # Direct family callers retain the usual load-on-device behaviour.
        dit, config = load_dit(cfg.dit_path, device=backbone_device or device, dtype=dtype)
        if memory.activation_checkpointing != "none":
            dit.enable_gradient_checkpointing(unsloth_offload=memory.activation_checkpointing == "unsloth")
        dit.attn_mode = self.resolve_attention(cfg.attention, device)
        text = AnimaText(cfg.text_encoder_path, tokenizer_path=cfg.tokenizer_path, dtype=dtype, device=device)
        latent = AnimaLatent(
            cfg.vae_path, device=device,
            dtype=torch.float32 if memory.no_half_vae or torch.device(device).type == "cpu" else dtype,
            vae_attention_chunking=memory.vae_attention_chunking,
            vae_tiling=memory.vae_tiling,
            cache_encode_tiled=memory.cache_encode_tiled,
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

    @staticmethod
    def resolve_attention(requested: str, device: torch.device | str) -> str:
        """Map the config value onto the vendored backend names; ``auto`` never picks an optional package.

        xFormers and FlashAttention are verified by :meth:`prepare_attention`, which keeps
        SDPA when the selected package cannot run on this device.
        """
        if requested == "metal_flash":
            from ypuddin.models.metal_attention import require_metal_flash

            require_metal_flash(device)
            return requested
        if requested == "sage":
            from .vendor.attention import backend_available

            if torch.device(device).type != "cuda":
                raise ValueError(f"model.attention={requested!r} requires CUDA")
            if not backend_available(requested):
                raise ValueError(
                    f"model.attention={requested!r} requires a compatible installed attention package; check Environment settings"
                )
            return requested
        if requested in ("xformers", "flash_attn"):
            return requested
        return "torch"

    def prepare_attention(self, loaded, configured, *, device, dtype, training, pinned=None):
        config = loaded.extra["dit_config"]
        heads = config["num_heads"]
        check_dit_attention(
            loaded, configured, [(heads, heads, config["model_channels"] // heads)],
            device=device, dtype=dtype, training=training, pinned=pinned,
        )

    # ----------------------------------------------------------------- forward
    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        from .vendor.attention import sampling_attention

        dit = loaded.backbone
        b, _, h, w = x_t.shape
        x5 = x_t.unsqueeze(2)
        padding_mask = torch.zeros(b, 1, h, w, dtype=x5.dtype, device=x5.device)
        with sampling_attention(not dit.training):
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
                description="训练图像模型的注意力和前馈层。适合大多数 LoRA / LoKr 训练。",
            ),
            "attn-only": TargetPreset(
                "attn-only", include=attn, description="只训练注意力层，训练的参数最少。"
            ),
            "full-linear": TargetPreset(
                "full-linear",
                include=attn + mlp + adaln,
                description="训练图像模型各主模块中的全部线性层，包括注意力、前馈和条件调制层。",
            ),
            "with-adapter": TargetPreset(
                "with-adapter",
                include=attn + mlp + adapter,
                description="训练图像模型的注意力、前馈层，以及连接文字特征的适配层。",
            ),
            "adapter-only": TargetPreset(
                "adapter-only",
                include=adapter,
                description="只训练连接文字特征与图像模型的适配层，不训练文本编码器本身。",
            ),
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
    """Use the loading geometry contract for planning, including the LLM adapter."""
    return {**base, **infer_config(shapes)}


def check_dit_attention(
    loaded: LoadedModel,
    configured: str,
    heads: list[tuple[int, int, int]],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    training: bool,
    pinned: str | None = None,
) -> None:
    """Run the shared DiT attention once per ``(query heads, key/value heads, head dim)`` and keep what passes.

    Head counts shrink to the grouped-query ratio; the head dimension, dtype and
    unmasked layout are the model's own. Anima and Krea 2 share this path.
    """
    from ypuddin.models.attention_check import check_attention, grad_mode

    from .vendor.attention import AttentionParams, attention

    selected = AnimaFamily.resolve_attention(configured, device)
    # Two key/value heads keep a multi-head layout; the query side keeps the grouped-query ratio.
    shapes = sorted({(2 * q // math.gcd(q, kv), 2 * kv // math.gcd(q, kv), dim) for q, kv, dim in heads})

    def run(backend: str) -> None:
        params = AttentionParams.create_attention_params("torch" if backend == "sdpa" else backend, False)
        with torch.autocast(torch.device(device).type, enabled=False), grad_mode(training):
            for q_heads, kv_heads, dim in shapes:
                q, k, v = (
                    torch.randn(1, 16, count, dim, device=device, dtype=dtype, requires_grad=training)
                    for count in (q_heads, kv_heads, kv_heads)
                )
                out = attention([q, k, v], attn_params=params)
                if training:
                    out.float().square().mean().backward()

    actual = check_attention(
        selected="sdpa" if selected == "torch" else selected,
        run=run,
        device=device,
        dtype=dtype,
        pinned=pinned,
    )
    mode = "torch" if actual == "sdpa" else actual
    loaded.backbone.attn_mode = mode
    # Krea 2 applies this again when it materializes a deferred backbone.
    loaded.extra["attention"] = mode


register("anima", AnimaFamily)
