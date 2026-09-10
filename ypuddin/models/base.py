"""Model-family contract: what the training loop needs to know about a diffusion model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import MemoryConfig, ModelConfig


@dataclass(frozen=True)
class LatentSpec:
    channels: int
    stride: int  # pixels per latent cell
    patch: int  # DiT patch size in latent cells
    fingerprint: str  # identifies the encoder + normalization for caching

    @property
    def align(self) -> int:
        """Image side lengths must be multiples of this."""
        return self.stride * self.patch


@dataclass(frozen=True)
class TextSpec:
    max_len: int
    fingerprint: str
    pad_floor: bool = True  # pad every batch up to max_len (model expects fixed length)


@dataclass(frozen=True)
class SamplingDefaults:
    steps: int = 25
    cfg: float = 4.0
    shift: float = 3.0
    sampler: str = "euler"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    latent: LatentSpec
    text: TextSpec
    sampling: SamplingDefaults
    capabilities: frozenset[str]
    objective: str = "rectified_flow"
    t_convention: str = "unit"  # backbone receives t in (0, 1)
    architecture: str = "unknown"  # modelspec.architecture base tag
    adapter_prefix: str = "lora_unet"


KNOWN_CAPABILITIES = frozenset(
    {"block_swap", "fp8_base", "activation_checkpointing", "text_encoder_train", "masked_loss", "online_text", "llm_adapter", "compile"}
)


class TextCond:
    """Opaque batch of text conditioning tensors (batch-first). Only the family interprets it."""

    def __init__(self, tensors: Mapping[str, Tensor]):
        self.tensors: dict[str, Tensor] = dict(tensors)

    def __getitem__(self, key: str) -> Tensor:
        return self.tensors[key]

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    @property
    def batch_size(self) -> int:
        return next(iter(self.tensors.values())).shape[0]

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> TextCond:
        out = {}
        for k, v in self.tensors.items():
            v = v.to(device)
            if dtype is not None and v.is_floating_point():
                v = v.to(dtype)
            out[k] = v
        return TextCond(out)

    def select(self, indices: list[int]) -> TextCond:
        idx = torch.as_tensor(indices)
        return TextCond({k: v[idx.to(v.device)] for k, v in self.tensors.items()})

    @staticmethod
    def cat(conds: list[TextCond]) -> TextCond:
        keys = conds[0].tensors.keys()
        return TextCond({k: torch.cat([c.tensors[k] for c in conds], dim=0) for k in keys})

    def repeat(self, n: int) -> TextCond:
        return TextCond({k: v.repeat_interleave(n, dim=0) for k, v in self.tensors.items()})


class TextPipeline(ABC):
    """Tokenize + encode captions. Cache entries are per-caption, CPU, trimmed to real length."""

    fingerprint: str

    @abstractmethod
    def encode(self, captions: list[str], device: torch.device | str) -> TextCond: ...

    @abstractmethod
    def encode_for_cache(self, captions: list[str]) -> list[dict[str, Tensor]]: ...

    @abstractmethod
    def cond_from_cache(self, entries: list[dict[str, Tensor]], device: torch.device | str) -> TextCond: ...

    def to(self, device: torch.device | str) -> None:  # noqa: D401
        """Move the encoder (no-op for stateless pipelines)."""

    def unload(self) -> None:
        """Release encoder weights (after caching)."""


class LatentPipeline(ABC):
    fingerprint: str
    channels: int
    stride: int

    @abstractmethod
    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, 3, H, W)`` in ``[-1, 1]`` -> normalized latents ``(B, C, H/stride, W/stride)``."""

    @abstractmethod
    def decode(self, latents: Tensor) -> Tensor:
        """Normalized latents -> pixels in ``[-1, 1]``."""

    def to(self, device: torch.device | str) -> None:
        pass

    def unload(self) -> None:
        pass


@dataclass
class MemoryLayout:
    blocks: list[nn.Module] = field(default_factory=list)  # swappable transformer blocks, in forward order
    keep_high_precision: tuple[str, ...] = ()  # module-name globs never quantized to fp8
    block_param_bytes: int = 0


@dataclass
class LoadedModel:
    backbone: nn.Module
    text: TextPipeline
    latent: LatentPipeline
    device: torch.device
    dtype: torch.dtype
    extra: dict[str, Any] = field(default_factory=dict)


class ModelFamily(ABC):
    spec: ModelSpec

    @abstractmethod
    def load(self, cfg: ModelConfig, memory: MemoryConfig, *, device: torch.device | str, dtype: torch.dtype) -> LoadedModel: ...

    @abstractmethod
    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        """Velocity prediction for ``x_t`` at ``t ∈ (0,1)``."""

    @abstractmethod
    def presets(self) -> dict[str, TargetPreset]: ...

    def default_preset(self) -> str:
        return "attn-mlp"

    def memory_layout(self, loaded: LoadedModel) -> MemoryLayout:
        return MemoryLayout()

    def blocks(self, loaded: LoadedModel) -> Iterator[nn.Module]:
        return iter(self.memory_layout(loaded).blocks)

    def validate_config(self, cfg: ModelConfig) -> list[str]:
        """Return human-readable problems (missing paths etc.) without loading weights."""
        return []
