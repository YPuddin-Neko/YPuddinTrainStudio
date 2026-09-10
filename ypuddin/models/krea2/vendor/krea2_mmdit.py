# Vendored from kohya-ss/musubi-tuner (Apache-2.0) at commit 8934cfb; local modifications: attention import, block swap removed, config inference added.
# Copyright 2025-2026 Kohya S. and musubi-tuner contributors. Licensed under the Apache License, Version 2.0.
# Upstream: https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_mmdit.py
# ruff: noqa  -- vendored third-party code, kept byte-close to upstream; not linted to project style
#
# Local modifications (see NOTICE.md):
#   * ``musubi_tuner.modules.attention`` -> ``ypuddin.models.anima.vendor.attention`` (same kohya API; GQA expanded there).
#   * The musubi block-swap offloader hooks (``enable_block_swap`` & co.) were removed: ypuddin swaps blocks with its
#     own ``BlockSwapper`` (forward + backward hooks on ``blocks[i]``).
#   * ``KREA2_CONFIG`` (the public ``single_mmdit_large_wide`` checkpoint) and ``infer_config`` (geometry from tensor
#     shapes) were added so reduced test models and future variants load without hard-coded sizes.
#   * Module and parameter names are unchanged: ``state_dict()`` keys match the official / ComfyUI checkpoints.
"""Krea 2 (K2) single-stream MMDiT.

Ported from references/Krea2/mmdit.py, plus musubi training hooks (gradient checkpointing,
block swap) and the shared attention backend. The core attention now goes through
musubi's common ``modules.attention`` (SDPA / flash / sageattn, selectable via attn_mode),
with the combined sequence ordered image-first so that valid tokens form a contiguous
prefix per sample — this lets the shared varlen / cu_seqlens machinery handle text padding.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from ypuddin.models.anima.vendor.attention import AttentionParams, attention as common_attention


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(6, dim=-1)
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    # @torch.compile(fullgraph=True)
    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.zeros(features, device=device, dtype=torch.float32))

    # @torch.compile(fullgraph=True)
    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(self, qkv: Tensor, freqs: Tensor | None = None, attn_params: AttentionParams | None = None) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        # QKNorm + RoPE run in [B, H, L, D] (K2-native layout) to preserve the reference numerics.
        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)

        # The shared attention expects [B, L, H, D] and returns [B, L, H*D]. GQA (heads != kvheads)
        # is detected and handled inside it (enable_gqa for SDPA; native for flash/sageattn).
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = common_attention([q, k, v], attn_params=attn_params)
        out = self.wo(x * F.sigmoid(gate))

        return out


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    # @torch.compile(fullgraph=True)
    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, attn_params: AttentionParams | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), attn_params=attn_params)
        x = x + self.mlp(self.postnorm(x))

        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth — that's fixed at 2 + 2 blocks.
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(
        self,
        x: Tensor,
        attn_params_nomask: AttentionParams | None = None,
        attn_params: AttentionParams | None = None,
    ) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), attn_params=attn_params_nomask)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)
        x = x.squeeze(-1)

        for block in self.refiner_blocks:
            x = block(x, attn_params=attn_params)

        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor, attn_params: AttentionParams | None = None) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, attn_params)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)

        return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        self.config = config
        # Backend for the shared attention ("torch"=SDPA, "flash", "sageattn", "xformers").
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(config.features, axes, theta=config.theta, ntk=1.0)
        self.first = nn.Linear(config.channels * config.patch**2, config.features, bias=True)

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6))

        # training hooks
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        # cpu_offload is accepted for interface parity; not implemented for K2 yet.
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        # `mask`/`pos` arrive in image-first order: [img (all valid), text (valid prefix + pad)].
        # The text-only key-padding mask is therefore the tail beyond the image tokens.
        imglen = img.shape[1]
        txtmask = mask[:, imglen:]  # (B, txt_len) bool

        # Text fusion is a self-attention over text tokens only (img_len=0). The per-layer
        # blocks see every token (no mask); the refiner masks padding via txtmask.
        txt_attn_params_nomask = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, None)
        txt_attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, txtmask)
        context = self.txtfusion(context, txt_attn_params_nomask, txt_attn_params)
        context = self.txtmlp(context)

        combined = torch.cat((img, context), dim=1)  # image first, then text

        # Pad the combined sequence to a multiple of 256 to keep compiled kernel shapes stable.
        # The pad lands on the text tail; extending txtmask with False makes the shared attention
        # machinery (cu_seqlens / key-padding mask / trim) exclude it, so it is numerically inert.
        fulllen = combined.shape[1]
        padlen = (-fulllen) % 256
        if padlen > 0:
            combined = F.pad(combined, (0, 0, 0, padlen))
            pos = F.pad(pos, (0, 0, 0, padlen))
            txtmask = F.pad(txtmask, (0, padlen), value=False)

        # Main blocks: bidirectional attention over [image (img_len, all valid) + text (padded)].
        # Image-first ordering keeps each sample's valid tokens a contiguous prefix, which the
        # shared varlen path requires.
        attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, imglen, txtmask)

        freqs = self.posemb(pos)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                combined = torch.utils.checkpoint.checkpoint(block, combined, tvec, freqs, attn_params, use_reentrant=False)
            else:
                combined = block(combined, tvec, freqs, attn_params)

        final = self.last(combined, t)
        output = final[:, :imglen, :]  # image tokens are the leading slice now

        return output


# ----------------------------------------------------------------------------------------------------------------------
# ypuddin additions
# ----------------------------------------------------------------------------------------------------------------------

#: The single config shipped with the public Krea 2 checkpoints (``single_mmdit_large_wide``, 12.9B parameters).
KREA2_CONFIG = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

KREA2_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.")


def _shape(sd, key):
    obj = sd[key]
    s = obj.shape if not callable(getattr(obj, "get_shape", None)) else obj.get_shape()
    return tuple(int(d) for d in s)


def infer_config(state_dict, patch: int = 2) -> SingleMMDiTConfig:
    """Derive a :class:`SingleMMDiTConfig` from checkpoint tensor shapes (bare keys, prefixes already stripped).

    Works with ``safetensors.safe_open(...).get_slice(k)`` objects as well as tensors, so no weights need to be read.
    ``multiplier`` is recovered from the SwiGLU width (``mlpdim = round_up(int(2*features/3) * multiplier, 128)``).
    """
    features, in_dim = _shape(state_dict, "first.weight")
    headdim = _shape(state_dict, "blocks.0.attn.qknorm.qnorm.scale")[0]
    heads = features // headdim
    kvheads = _shape(state_dict, "blocks.0.attn.wk.weight")[0] // headdim
    layers = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks."))
    tdim = _shape(state_dict, "tmlp.0.weight")[1]
    txtdim = _shape(state_dict, "txtmlp.1.weight")[1]
    txtlayers = _shape(state_dict, "txtfusion.projector.weight")[1]
    txt_headdim = _shape(state_dict, "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale")[0]
    txtheads = txtdim // txt_headdim
    txtkvheads = _shape(state_dict, "txtfusion.layerwise_blocks.0.attn.wk.weight")[0] // txt_headdim
    mlpdim = _shape(state_dict, "blocks.0.mlp.gate.weight")[0]
    multiplier = next(
        (m for m in range(1, 17) if 128 * ((int(2 * features / 3) * m + 127) // 128) == mlpdim), 4
    )
    bias = "blocks.0.attn.wq.bias" in state_dict
    return SingleMMDiTConfig(
        features=features,
        tdim=tdim,
        txtdim=txtdim,
        heads=heads,
        kvheads=kvheads,
        multiplier=multiplier,
        layers=layers,
        patch=patch,
        channels=in_dim // (patch * patch),
        bias=bias,
        txtheads=txtheads,
        txtkvheads=txtkvheads,
        txtlayers=txtlayers,
    )


__all__ = [
    "KREA2_CONFIG",
    "KREA2_KEY_PREFIXES",
    "SingleMMDiTConfig",
    "SingleStreamDiT",
    "SingleStreamBlock",
    "TextFusionTransformer",
    "infer_config",
    "temb",
]
