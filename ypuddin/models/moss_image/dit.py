"""MOSS-Image DiT, derived from OpenMOSS's Apache-2.0 pipeline_moss_image.py.

Upstream HF revision 5f40f07a304541840b6019d2d34d30f966756787.
Weight names and computation are retained; attention masks permit true image batches,
and batches without text padding omit them so fused attention kernels apply.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _validate_skip_layers(layers, depth, name='skip_layers'):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f'{name} must contain integer block indices in [0, {depth}).')
    return frozenset(layers)


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, mask=None, mode="torch"):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    if mode != "torch":
        # The shared dispatcher runs the selected package on calls it supports and SDPA on the rest.
        from ypuddin.models.anima.vendor.attention import AttentionParams, attention

        q, k = (_rotate(t.reshape(shape), rope).to(v.dtype) for t in (q, k))
        return attention([q, k, v.reshape(shape)], attn_params=AttentionParams(mode, attention_mask=mask))
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask).transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.attn_mode = "torch"
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, mask=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, mask, self.attn_mode))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, mask=None):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, mask)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.attn_mode = "torch"
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')
        })
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj['q'].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads, mask, self.attn_mode)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer('bands', self._default_bands(torch.device('cpu')))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, lengths, device):
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        lengths = torch.as_tensor(lengths, device=device).reshape(-1)
        batch, text_length = len(lengths), int(lengths.max().item())
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.zeros_like(h), h, w), -1).flatten(0, 1).float()
        image = image[None].expand(batch, -1, -1).clone()
        image[:, :, 1] = lengths[:, None]
        image[:, :, 2] -= (height - 1) / 2
        image[:, :, 3] -= (width - 1) / 2
        text = torch.zeros(batch, text_length, 4, device=device, dtype=torch.float32)
        text[:, :, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 1).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)


class MOSSImageTransformer(nn.Module):
    """MOSSImage DiT with training-compatible state-dict names and SDPA or FlashAttention 2.

    Each sample is evaluated at its true text length, including text refinement;
    padded tokens cannot influence either text or joint attention.
    """
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0, checkpointing="none"):
        super().__init__()
        if checkpointing not in {"none", "block"}:
            raise ValueError("MOSS-Image supports none or block checkpointing")
        self.checkpointing = checkpointing
        self.config = SimpleNamespace(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, num_layers=num_layers,
                                      text_dim=text_dim, freq_dim=freq_dim, latent_channels=latent_channels,
                                      rope_base=rope_base)
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError('Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.')
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def set_attention_mode(self, mode):
        """``torch`` is SDPA; ``flash_attn`` uses FlashAttention 2 where the shared dispatcher allows."""
        for module in self.modules():
            if isinstance(module, (_Block, _TextBlock)):
                module.attn_mode = mode

    def load_training_state_dict(self, state_dict, *, prefix=''):
        """Load a consolidated raw/EMA model mapping, strictly; no silent key drops."""
        if prefix:
            if any(not key.startswith(prefix) for key in state_dict):
                raise ValueError('All keys must start with the explicit prefix.')
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
        return self.load_state_dict(state_dict, strict=True)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError('hidden_states must be [B, latent_channels, H, W].')
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device="cpu" if times.device.type == "mps" else times.device, dtype=torch.float64) / half)
        phase = times.to(freq.device).double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float().to(times.device)
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        lengths = mask.bool().sum(1)
        if (lengths == 0).any():
            raise ValueError("Each sample needs at least one valid text token.")
        context = encoder_hidden_states.new_zeros(batch, int(lengths.max().item()), self.config.text_dim)
        for index in range(batch):
            context[index, :lengths[index]] = encoder_hidden_states[index, mask[index].bool()]
        valid = torch.arange(context.shape[1], device=hidden_states.device)[None] < lengths[:, None]
        text_mask = joint_mask = None
        if not bool(valid.all()):
            # An all-true mask selects the same keys as none, but keeps SDPA off its FlashAttention kernel.
            text_mask = valid[:, None, None, :]
            joint_mask = torch.cat((torch.ones(batch, height * width, device=valid.device, dtype=torch.bool), valid), 1)
            joint_mask = joint_mask[:, None, None, :]
        video = self.video_embedder(hidden_states.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous())
        rope = self.rope(height, width, lengths, hidden_states.device).to(video.dtype)
        text = self.text_embedder.proj(context.to(self.text_embedder.proj.weight.dtype))
        recompute = self.checkpointing == "block" and self.training and torch.is_grad_enabled()
        for block in self.text_embedder.blocks:
            args = (text, rope[:, height * width:], text_mask)
            text = checkpoint(block, *args, use_reentrant=False) if recompute else block(*args)
        x = torch.cat((video, text), 1)
        for layer_index, layer in enumerate(self.layers):
            if layer_index not in skip_layers:
                args = (x, rope, mod, joint_mask)
                x = checkpoint(layer, *args, use_reentrant=False) if recompute else layer(*args)
        x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        sample = x.transpose(1, 2).reshape(batch, -1, height, width)
        return SimpleNamespace(sample=sample) if return_dict else (sample,)
