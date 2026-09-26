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
from ypuddin.config import MemoryConfig, ModelConfig, TrainConfig
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
CUDA_STAGING_MARGIN = 2 * 1024**3


# --------------------------------------------------------------------------- loading
def _strip_prefix(key: str) -> str:
    from .vendor.krea2_mmdit import KREA2_KEY_PREFIXES

    for p in KREA2_KEY_PREFIXES:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _config_from_header(path: str | Path):
    """Infer geometry without reading the checkpoint's tensor payload."""
    from safetensors import safe_open

    from .vendor.krea2_mmdit import infer_config

    with safe_open(str(Path(path).expanduser()), framework="pt") as checkpoint:
        shapes = {
            _strip_prefix(key): checkpoint.get_slice(key)
            for key in checkpoint.keys()
            if _strip_prefix(key) != "scaled_fp8" and not key.endswith(SCALE_SUFFIXES)
        }
        return infer_config(shapes)


def _checkpoint_signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev


def _checkpoint_storage_bytes(path: Path, dtype: torch.dtype) -> int:
    """Target storage from the actual header, without reading tensor payloads."""
    from safetensors import safe_open

    compute_bytes = torch.empty((), dtype=dtype).element_size()
    unchanged_sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "U16": 2,
        "I16": 2,
        "U32": 4,
        "I32": 4,
        "U64": 8,
        "I64": 8,
        "C64": 8,
    }
    total = 0
    with safe_open(str(path), framework="pt") as checkpoint:
        for key in checkpoint.keys():
            if _strip_prefix(key) == "scaled_fp8":
                continue
            tensor = checkpoint.get_slice(key)
            count = math.prod(tensor.get_shape())
            source_dtype = tensor.get_dtype()
            if key.endswith(SCALE_SUFFIXES):
                if count != 1:
                    raise ValueError(f"Krea 2 FP8 scale {key} must contain exactly one value")
                total += 4  # _read_state_dict normalizes checkpoint scales to FP32
            elif source_dtype in {"F8_E4M3", "F8_E4M3FN", "F8_E5M2"}:
                total += count  # preserve the checkpoint's FP8 storage
            elif source_dtype in {"F64", "F32", "F16", "BF16", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}:
                total += count * compute_bytes
            elif source_dtype in unchanged_sizes:
                total += count * unchanged_sizes[source_dtype]
            else:
                raise ValueError(f"Unsupported Krea 2 checkpoint dtype: {source_dtype}")
    return total


def _materialization_device(loaded: LoadedModel) -> torch.device:
    """Avoid overlapping a complete source mmap with pinned masters when CUDA has room.

    This is only a temporary placement for a CUDA block-swap job. A CPU/MPS job,
    or a model that needs swapping to fit at all, keeps its requested placement.
    The free-memory observation is conservative admission, not an OOM guarantee.
    """
    requested = torch.device(loaded.extra["backbone_device"])
    decision: dict[str, Any] = {"device": str(requested), "reason": "requested_placement"}
    loaded.extra["materialization_placement"] = decision
    if loaded.device.type != "cuda" or requested.type != "cpu" or not loaded.extra.get("blocks_to_swap"):
        return requested
    size = _checkpoint_storage_bytes(loaded.extra["dit_path"], loaded.dtype)
    decision.update(weight_bytes=size, margin_bytes=CUDA_STAGING_MARGIN)
    try:
        free, _ = torch.cuda.mem_get_info(loaded.device)
    except (RuntimeError, OSError) as error:
        decision.update(reason="cuda_memory_unavailable")
        log.warning("Krea 2: keeping CPU materialization; CUDA free-memory query failed: %s", error)
        return requested
    decision["free_cuda_bytes"] = free
    if free < size + CUDA_STAGING_MARGIN:
        decision["reason"] = "insufficient_cuda_headroom"
        log.info(
            "Krea 2: keeping CPU materialization (weights %d + margin %d > free CUDA %d bytes)",
            size,
            CUDA_STAGING_MARGIN,
            free,
        )
        return requested
    decision.update(device=str(loaded.device), reason="cuda_staging_before_host_masters")
    log.info(
        "Krea 2: materializing on %s before block swap to release the source mmap before pinned allocation (weights %d, free %d, margin %d bytes)",
        loaded.device,
        size,
        free,
        CUDA_STAGING_MARGIN,
    )
    return loaded.device


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
            # A scalar view retains its original safetensors tensor via _base,
            # even after block swapping rebinds its .data. Detach that view so
            # moving the scale can release the checkpoint's shared mmap.
            scales[base] = v.reshape(()).to(torch.float32).detach()
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
    # Parameters/FrozenLinear buffers now own all required storages. Retaining
    # the input dictionaries would keep every old mmap view until after the
    # device move, even when the corresponding module tensor has moved.
    del weights, scales
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model, config.__dict__.copy()


# --------------------------------------------------------------------------- family
class Krea2Family(ModelFamily):
    spec = ModelSpec(
        name="krea2",
        attention_backends=("auto", "sdpa", "xformers", "flash_attn", "metal_flash"),
        latent=LatentSpec(channels=16, stride=8, patch=2, fingerprint=AnimaLatent.fingerprint),
        text=TextSpec(
            max_len=512, fingerprint=Krea2Text.fingerprint, pad_floor=False, encoder_params=4_022_000_000
        ),
        sampling=SamplingDefaults(steps=28, cfg=5.5, shift=None, sampler="euler"),
        activation_units=30.0,
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
        from .vendor.krea2_mmdit import SingleStreamDiT

        if cfg.attention == "metal_flash":
            from ypuddin.models.metal_attention import require_metal_flash

            require_metal_flash(device)
        problems = self.validate_config(cfg)
        if problems:
            raise FileNotFoundError("; ".join(problems))
        path = Path(cfg.dit_path).expanduser().resolve()
        signature = _checkpoint_signature(path)
        geometry = _config_from_header(path)
        config = geometry.__dict__.copy()
        # The 4B text encoder and VAE must finish caching before the 12.9B DiT
        # occupies host memory. Both Trainer and standalone sampling call the
        # materialization hook after releasing those encoders.
        with torch.device("meta"):
            dit = SingleStreamDiT(geometry).to(dtype=dtype)
        dit.eval().requires_grad_(False)
        attention = self.resolve_attention(cfg.attention, device)
        if memory.activation_checkpointing == "unsloth":
            log.warning(
                "krea2: unsloth offloaded checkpointing is not available for this family, using block checkpointing"
            )
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
            "prepared Krea 2 DiT metadata: features=%s layers=%s heads=%s/%s; weights deferred until after caching",
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
            extra={
                "dit_config": config,
                "variant": resolve_variant(cfg.dit_path, cfg.krea2_variant),
                "dit_path": path,
                "dit_signature": signature,
                "backbone_device": backbone_device or device,
                "blocks_to_swap": memory.blocks_to_swap,
                "checkpointing": memory.activation_checkpointing != "none",
                "attention": attention,
                "materialized": False,
            },
        )

    def materialize_backbone(self, loaded: LoadedModel) -> None:
        # Manually assembled LoadedModels already contain their real backbone.
        if loaded.extra.get("materialized", True):
            return
        loaded.text.unload()
        loaded.latent.unload()
        path = loaded.extra["dit_path"]
        if _checkpoint_signature(path) != loaded.extra["dit_signature"]:
            raise RuntimeError("Krea 2 checkpoint changed after metadata loading; restart this operation")
        dit, config = load_dit(path, device=_materialization_device(loaded), dtype=loaded.dtype)
        if config != loaded.extra["dit_config"]:
            raise RuntimeError("Krea 2 checkpoint geometry changed after metadata loading")
        if loaded.extra["checkpointing"]:
            dit.enable_gradient_checkpointing()
        dit.attn_mode = loaded.extra["attention"]
        loaded.backbone = dit
        loaded.extra["materialized"] = True

    @staticmethod
    def resolve_attention(requested: str, device: torch.device | str) -> str:
        from ypuddin.models.anima.family import AnimaFamily

        return AnimaFamily.resolve_attention(requested, device)

    # ----------------------------------------------------------------- forward
    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        from ypuddin.models.anima.vendor.attention import sampling_attention

        if loaded.extra.get("materialized") is False:
            raise RuntimeError("Call materialize_backbone after caching and before Krea 2 forward")
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
                description="训练模型中的全部线性层，包括文字融合、嵌入与输出层。",
            ),
            "attn-mlp": TargetPreset(
                "attn-mlp",
                include=attn + mlp,
                description="训练图像模型主模块的注意力和前馈层，不包含文字融合、嵌入与输出层。",
            ),
            "attn-only": TargetPreset(
                "attn-only",
                include=attn,
                description="仅训练图像模型主模块的注意力投影，训练参数更少。",
            ),
            "attn-mlp-text": TargetPreset(
                "attn-mlp-text",
                include=attn + mlp + text,
                description="训练主模块的注意力、前馈层，以及模型内部的文字融合层；不训练文本编码器本身。",
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
        from .vendor.krea2_mmdit import KREA2_CONFIG, SingleStreamDiT

        config = KREA2_CONFIG
        if cfg.dit_path and Path(cfg.dit_path).expanduser().exists():
            try:
                config = _config_from_header(cfg.dit_path)
            except Exception as e:  # noqa: BLE001
                log.warning("could not infer Krea 2 geometry from %s: %s", cfg.dit_path, e)
        return SingleStreamDiT(config)

    def prepare_backbone_for_plan(self, backbone: nn.Module, cfg: ModelConfig, dtype: torch.dtype) -> None:
        """Mirror load_dit's mixed FP8/compute-dtype storage using only safetensors headers."""
        from safetensors import safe_open

        super().prepare_backbone_for_plan(backbone, cfg, dtype)
        if not cfg.dit_path or not Path(cfg.dit_path).expanduser().is_file():
            return
        expected = backbone.state_dict()
        weights, scales = {}, {}
        fp8 = {"F8_E4M3": torch.float8_e4m3fn, "F8_E4M3FN": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
        with safe_open(str(Path(cfg.dit_path).expanduser()), framework="pt") as checkpoint:
            for source_key in checkpoint.keys():
                key = _strip_prefix(source_key)
                if key == "scaled_fp8":
                    continue
                tensor = checkpoint.get_slice(source_key)
                shape = tensor.get_shape()
                if key.endswith(SCALE_SUFFIXES):
                    if math.prod(shape) != 1:
                        raise ValueError(f"Krea 2 FP8 scale {key} must contain exactly one value")
                    scales[key.rsplit(".", 1)[0]] = torch.empty((), device="meta", dtype=torch.float32)
                elif key in expected:
                    target_dtype = fp8.get(tensor.get_dtype(), expected[key].dtype)
                    weights[key] = torch.empty(shape, device="meta", dtype=target_dtype)
        backbone.requires_grad_(False)
        missing, _ = backbone.load_state_dict(weights, strict=False, assign=True)
        if missing:
            raise ValueError(f"Krea 2 checkpoint is missing tensors: {missing[:5]}")
        _freeze_fp8_linears(backbone, scales)

    def cache_memory_estimate(self, cfg: TrainConfig, dtype: torch.dtype) -> dict[str, float]:
        from transformers import Qwen3VLTextModel

        from .text import text_config_for

        # The single-file loader discards the vision tower and lm_head. FP8
        # decoder weights, when supplied, are dequantized to the compute dtype.
        text_cfg = text_config_for(cfg.model.text_encoder_path or "")
        with torch.device("meta"):
            decoder = Qwen3VLTextModel(text_cfg)
        element_size = torch.empty((), dtype=dtype).element_size()
        weights = sum(p.numel() for p in decoder.parameters()) * element_size
        hidden, layers = text_cfg.hidden_size, text_cfg.num_hidden_layers
        # build_text_cache defaults to 16; even a small training batch does not
        # reduce it. Budget a full cold-cache batch, without assuming cache hits.
        batch, tokens = 16, self.spec.text.max_len + 34
        selected = min(12, layers)
        if cfg.model.dit_path and Path(cfg.model.dit_path).expanduser().is_file():
            selected = _config_from_header(cfg.model.dit_path).txtlayers
        retained = batch * tokens * hidden * (layers + 1 + selected) * element_size
        qkv = (text_cfg.num_attention_heads + 2 * text_cfg.num_key_value_heads) * text_cfg.head_dim
        scratch = batch * tokens * (qkv + 4 * hidden + 4 * text_cfg.intermediate_size) * element_size
        # Include a full FP32 attention-score workspace as a conservative
        # allowance, although fused SDPA normally needs less temporary storage.
        scratch += batch * text_cfg.num_attention_heads * tokens * tokens * 4
        phases = {"text_cache": (weights + retained + scratch) / 2**20 + 512}
        if cfg.model.vae_path and Path(cfg.model.vae_path).expanduser().is_file():
            from safetensors import safe_open

            with safe_open(str(Path(cfg.model.vae_path).expanduser()), framework="pt") as checkpoint:
                shapes = [checkpoint.get_slice(key).get_shape() for key in checkpoint.keys()]
            vae_weights = sum(math.prod(shape) for shape in shapes) * element_size
            pixels = (
                cfg.dataset.native_max_pixels
                if cfg.dataset.resolution_mode == "native"
                else max(cfg.dataset.resolutions) ** 2
            )
            # Qwen-Image's encoder processes one image at a time by default;
            # feature-map/residual workspaces are heuristic, weights are header based.
            vae_scratch = pixels * 96 * 8 * element_size
            phases["latent_cache"] = (vae_weights + vae_scratch) / 2**20 + 512
        return phases

    def training_tokens_for_plan(self, image_tokens: int) -> int:
        # Joint self-attention also processes the cached caption tokens. Their
        # actual trimmed length varies; planning budgets the supported maximum.
        return image_tokens + self.spec.text.max_len

    def linear_module_names(self) -> list[str]:
        from .vendor.krea2_mmdit import KREA2_CONFIG, SingleStreamDiT

        with torch.device("meta"):
            model = SingleStreamDiT(KREA2_CONFIG)
        return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


register("krea2", Krea2Family)
