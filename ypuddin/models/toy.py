"""Toy model family: a miniature DiT with the same interfaces and module naming as Anima.

Everything runs on CPU in milliseconds, which lets the whole pipeline (caching, bucketing,
adapters, training loop, checkpoints, sampling, service) be tested end-to-end.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import MemoryConfig, ModelConfig

from .base import (
    LatentPipeline,
    LatentSpec,
    LoadedModel,
    MemoryLayout,
    ModelFamily,
    ModelSpec,
    SamplingDefaults,
    TextCond,
    TextPipeline,
    TextSpec,
)
from .registry import register

DIM = 64
LATENT_CH = 4
STRIDE = 8
PATCH = 2
VOCAB = 4096
MAX_LEN = 32


# --------------------------------------------------------------------------- latent
class ToyLatent(LatentPipeline):
    fingerprint = "toy-latent-v1"
    channels = LATENT_CH
    stride = STRIDE

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(7)
        self.proj = torch.randn(LATENT_CH, 3, generator=g) / math.sqrt(3)
        self.inv = torch.linalg.pinv(self.proj)

    def encode(self, pixels: Tensor) -> Tensor:
        pooled = F.avg_pool2d(pixels.float(), STRIDE)
        return torch.einsum("bchw,dc->bdhw", pooled, self.proj.to(pooled.device))

    def decode(self, latents: Tensor) -> Tensor:
        rgb = torch.einsum("bdhw,cd->bchw", latents.float(), self.inv.to(latents.device))
        return F.interpolate(rgb, scale_factor=STRIDE, mode="nearest").clamp(-1, 1)


# --------------------------------------------------------------------------- text
def _token_ids(caption: str) -> list[int]:
    words = caption.replace(",", " , ").split()
    ids = [1 + int(hashlib.blake2b(w.encode(), digest_size=4).hexdigest(), 16) % (VOCAB - 2) for w in words]
    return ids[:MAX_LEN] or [VOCAB - 1]  # empty caption -> single "<empty>" token


class ToyText(TextPipeline):
    fingerprint = "toy-text-v1"

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(11)
        self.embed = nn.Parameter(torch.randn(VOCAB, DIM, generator=g) * 0.5, requires_grad=False)
        self.pos = nn.Parameter(torch.randn(MAX_LEN, DIM, generator=g) * 0.1, requires_grad=False)

    def trainable_modules(self) -> dict[str, nn.Module]:
        module = nn.Module()
        module.register_parameter("embed", self.embed)
        module.register_parameter("pos", self.pos)
        return {"text_encoder": module}

    def to(self, device: torch.device | str) -> None:
        # CPU-only toy is also used on CUDA for tiny smoke tests.
        for parameter in (self.embed, self.pos):
            parameter.data = parameter.data.to(device)

    def _embed(self, ids: list[int]) -> Tensor:
        t = torch.as_tensor(ids, device=self.embed.device)
        return self.embed[t] + self.pos[: len(ids)]

    def encode(self, captions: list[str], device: torch.device | str) -> TextCond:
        return self.cond_from_cache(self.encode_for_cache(captions), device)

    def encode_for_cache(self, captions: list[str]) -> list[dict[str, Tensor]]:
        return [{"embeds": self._embed(_token_ids(c))} for c in captions]

    def cond_from_cache(self, entries: list[dict[str, Tensor]], device: torch.device | str) -> TextCond:
        L = max(e["embeds"].shape[0] for e in entries)
        embeds = torch.zeros(len(entries), L, DIM)
        mask = torch.zeros(len(entries), L, dtype=torch.bool)
        for i, e in enumerate(entries):
            n = e["embeds"].shape[0]
            embeds[i, :n] = e["embeds"]
            mask[i, :n] = True
        return TextCond({"embeds": embeds, "mask": mask}).to(device)


# --------------------------------------------------------------------------- backbone
class _Attention(nn.Module):
    def __init__(self, dim: int, ctx_dim: int | None = None, heads: int = 4):
        super().__init__()
        ctx_dim = ctx_dim or dim
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(ctx_dim, dim, bias=False)
        self.v_proj = nn.Linear(ctx_dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, ctx: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        ctx = x if ctx is None else ctx
        B, L, D = x.shape
        h = self.heads
        q = self.q_proj(x).view(B, L, h, D // h).transpose(1, 2)
        k = self.k_proj(ctx).view(B, -1, h, D // h).transpose(1, 2)
        v = self.v_proj(ctx).view(B, -1, h, D // h).transpose(1, 2)
        attn_mask = None if mask is None else mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.output_proj(out.transpose(1, 2).reshape(B, L, D))


class _MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim * 4, bias=False)
        self.layer2 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


class _AdaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: Tensor) -> tuple[Tensor, ...]:
        return self.linear(F.silu(t_emb)).chunk(6, dim=-1)


class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = _Attention(dim)
        self.cross_attn = _Attention(dim, ctx_dim=DIM)
        self.mlp = _MLP(dim)
        self.adaln = _AdaLN(dim)

    def forward(self, x: Tensor, t_emb: Tensor, ctx: Tensor, ctx_mask: Tensor) -> Tensor:
        s1, sc1, g1, s2, sc2, g2 = (m[:, None, :] for m in self.adaln(t_emb))
        x = x + g1 * self.self_attn(self.norm1(x) * (1 + sc1) + s1)
        x = x + self.cross_attn(self.norm2(x), ctx, ctx_mask)
        x = x + g2 * self.mlp(self.norm3(x) * (1 + sc2) + s2)
        return x


def _sinusoidal(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ToyDiT(nn.Module):
    def __init__(self, dim: int = DIM, depth: int = 2):
        super().__init__()
        self.dim = dim
        self.x_embedder = nn.Linear(LATENT_CH * PATCH * PATCH, dim)
        self.t_embedder = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([ToyBlock(dim) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_layer = nn.Linear(dim, LATENT_CH * PATCH * PATCH)
        # A "pretrained" toy: small but non-zero output projection so adapters inside the blocks
        # actually influence the prediction (a zero-init DiT head would hide every adapter).
        nn.init.normal_(self.final_layer.weight, std=0.05)
        nn.init.zeros_(self.final_layer.bias)
        self.grad_checkpointing = False

    def patchify(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        B, C, H, W = x.shape
        h, w = H // PATCH, W // PATCH
        x = x.view(B, C, h, PATCH, w, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * PATCH * PATCH)
        return x, (h, w)

    def unpatchify(self, x: Tensor, hw: tuple[int, int]) -> Tensor:
        B = x.shape[0]
        h, w = hw
        x = (
            x.view(B, h, w, LATENT_CH, PATCH, PATCH)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, LATENT_CH, h * PATCH, w * PATCH)
        )
        return x

    def forward(self, x: Tensor, t: Tensor, ctx: Tensor, ctx_mask: Tensor) -> Tensor:
        tokens, hw = self.patchify(x)
        pos = torch.linspace(-1, 1, tokens.shape[1], device=x.device)[None, :, None]
        h = self.x_embedder(tokens) + pos
        t_emb = self.t_embedder(_sinusoidal(t, 256))
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, t_emb, ctx, ctx_mask, use_reentrant=False)
            else:
                h = block(h, t_emb, ctx, ctx_mask)
        out = self.final_layer(self.final_norm(h))
        return self.unpatchify(out, hw)


# --------------------------------------------------------------------------- family
class ToyFamily(ModelFamily):
    spec = ModelSpec(
        name="toy",
        latent=LatentSpec(channels=LATENT_CH, stride=STRIDE, patch=PATCH, fingerprint=ToyLatent.fingerprint),
        text=TextSpec(max_len=MAX_LEN, fingerprint=ToyText.fingerprint, pad_floor=False),
        sampling=SamplingDefaults(steps=8, cfg=2.0, shift=1.0),
        capabilities=frozenset(
            {"activation_checkpointing", "online_text", "masked_loss", "block_swap", "compile", "fp8_base"}
        ),
        architecture="toy-dit",
        label="Toy DiT（CPU 自检）",
    )

    def load(
        self,
        cfg: ModelConfig,
        memory: MemoryConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        backbone_device: torch.device | str | None = None,
    ) -> LoadedModel:
        torch.manual_seed(0)
        backbone = ToyDiT()
        if cfg.dit_path:
            from safetensors.torch import load_file

            backbone.load_state_dict(load_file(cfg.dit_path))
        backbone.to(device=backbone_device or device, dtype=dtype)
        backbone.grad_checkpointing = memory.activation_checkpointing != "none"
        text = ToyText()
        if cfg.text_encoder_path:
            from pathlib import Path

            from safetensors.torch import load_file

            source = Path(cfg.text_encoder_path)
            text.trainable_modules()["text_encoder"].load_state_dict(
                load_file(str(source / "model.safetensors" if source.is_dir() else source))
            )
        return LoadedModel(
            backbone=backbone, text=text, latent=ToyLatent(), device=torch.device(device), dtype=dtype
        )

    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        return loaded.backbone(x_t, t, cond["embeds"].to(x_t.dtype), cond["mask"])

    def presets(self) -> dict[str, TargetPreset]:
        return {
            "attn-mlp": TargetPreset(
                "attn-mlp",
                include=("blocks.*.self_attn.*_proj", "blocks.*.cross_attn.*_proj", "blocks.*.mlp.layer*"),
                description="注意力 + MLP",
            ),
            "attn-only": TargetPreset(
                "attn-only", include=("blocks.*.self_attn.*_proj", "blocks.*.cross_attn.*_proj")
            ),
            "full-linear": TargetPreset("full-linear", include=("blocks.*",)),
        }

    def meta_backbone(self, cfg: ModelConfig) -> nn.Module:
        return ToyDiT()

    def linear_module_names(self) -> list[str]:
        with torch.device("meta"):
            backbone = ToyDiT()
        return [name for name, module in backbone.named_modules() if isinstance(module, nn.Linear)]

    def memory_layout_meta(self, backbone: nn.Module) -> MemoryLayout:
        return MemoryLayout(blocks=list(backbone.blocks))

    def memory_layout(self, loaded: LoadedModel) -> MemoryLayout:
        blocks = list(loaded.backbone.blocks)
        nbytes = sum(p.numel() * p.element_size() for p in blocks[0].parameters()) if blocks else 0
        return MemoryLayout(
            blocks=blocks,
            keep_high_precision=("x_embedder", "t_embedder*", "final_layer"),
            block_param_bytes=nbytes,
        )


register("toy", ToyFamily)
