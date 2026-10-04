"""Model-family contract: what the training loop needs to know about a diffusion model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import MemoryConfig, ModelConfig, ObjectiveConfig, SamplingConfig, TrainConfig


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
    encoder_params: int = 0  # resident on the accelerator in online text mode (planner VRAM estimate)


@dataclass(frozen=True)
class SamplingDefaults:
    steps: int = 25
    cfg: float = 4.0
    shift: float | None = 3.0  # None: resolution dependent, see ``ModelFamily.sampling_shift``
    sampler: str = "euler"
    guidance: float | None = None  # model guidance embedding (distinct from classifier-free guidance)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    latent: LatentSpec
    text: TextSpec
    sampling: SamplingDefaults
    capabilities: frozenset[str]
    # The shared caption reader renders supported sidecars to text before encoding.
    caption_formats: tuple[str, ...] = ("txt", "json")
    objective: str = "rectified_flow"
    t_convention: str = "unit"  # backbone receives t in (0, 1)
    architecture: str = "unknown"  # modelspec.architecture base tag
    adapter_prefix: str = "lora_unet"
    label: str = ""  # human-readable name for UIs ("Krea 2 Raw 12.9B"); falls back to ``name``
    retired_reason: str | None = None  # readable legacy configs, unavailable for new execution
    weights: tuple[tuple[str, str, str], ...] = ()  # (ModelConfig field, label, hint) the family needs
    optional_weights: tuple[str, ...] = ()  # fields supplied by a bundled checkpoint unless overridden
    directory_only_weights: tuple[str, ...] = ()  # cannot be prepared by single-file downloads
    attention_backends: tuple[str, ...] = ("auto", "sdpa", "xformers", "flash_attn")
    sampling_samplers: tuple[str, ...] = ("euler", "euler_ancestral", "heun", "er_sde")
    sampling_schedulers: tuple[str, ...] = ("uniform", "simple", "sgm_uniform", "normal")
    objective_timestep_sampling: tuple[str, ...] = (
        "uniform",
        "logit_normal",
        "shift",
        "resolution_shift",
        "mode",
        "cosmap",
    )
    objective_weighting: tuple[str, ...] = ("none", "sigma_sqrt", "cosmap", "snr_like", "cosmos")
    # Activations one transformer block keeps for backward, in units of tokens x width x activation bytes,
    # when every block linear carries an adapter. Measured with saved-tensor hooks on the real block.
    activation_units: float = 14.0
    # A backbone without uniform blocks (the SDXL UNet) gives its whole activation peak instead, per latent token
    # and activation byte, for each checkpointing mode it implements. Measured on the real architecture.
    backbone_activation_units: tuple[tuple[str, float], ...] = ()
    # Checkpointing modes the family implements; "unsloth" also moves block inputs to system memory.
    checkpointing_modes: tuple[str, ...] = ("none", "block")


KNOWN_CAPABILITIES = frozenset(
    {
        "block_swap",
        "fp8_base",
        "activation_checkpointing",
        "text_encoder_train",
        "masked_loss",
        "online_text",
        "llm_adapter",
        "compile",
    }
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

    def trainable_modules(self) -> dict[str, nn.Module]:
        """Materialize the actual encoder modules, retaining their original parameter names."""
        raise NotImplementedError("This text pipeline does not implement text-encoder training")

    def enable_training(self) -> dict[str, nn.Module]:
        self.training_enabled = True
        if hasattr(self, "dtype"):
            self.dtype = torch.float32
        modules = self.trainable_modules()
        for module in modules.values():
            module.to(dtype=torch.float32).requires_grad_(True).train()
        return modules

    def enable_adapter_training(self) -> dict[str, nn.Module]:
        """Keep encoder base precision and gradients through online conditioning.

        The caller injects/optimizes only adapter parameters. Frozen embeddings and
        other encoder weights must not become full-training parameters by accident.
        """
        modules = self.trainable_modules()
        for module in modules.values():
            module.requires_grad_(False).train()
        self.training_enabled = True
        return modules


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
    def load(
        self,
        cfg: ModelConfig,
        memory: MemoryConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        backbone_device: torch.device | str | None = None,
    ) -> LoadedModel: ...

    @abstractmethod
    def forward(self, loaded: LoadedModel, x_t: Tensor, t: Tensor, cond: TextCond, **extra: Any) -> Tensor:
        """Predict this family's training target at unit-interval time ``t``.

        The family converts unit time into its backbone's time convention. ``geometry``
        may provide per-image original size, crop offset and target size for conditioning.
        """

    def build_objective(self, loaded: LoadedModel, cfg: ObjectiveConfig) -> Any:
        """Construct the noising/target/loss contract for these loaded weights."""
        from ypuddin.objectives import Objective

        if self.spec.objective != "rectified_flow":
            raise NotImplementedError(f"{self.spec.name}: implement its {self.spec.objective} objective")
        return Objective(cfg)

    def materialize_backbone(self, loaded: LoadedModel) -> None:
        """Finish deferred weight loading after VAE/text caches have released their encoders."""

    def prepare_attention(
        self,
        loaded: LoadedModel,
        configured: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        training: bool,
        pinned: str | None = None,
    ) -> None:
        """Check the selected attention backend once, after loading and before the backbone computes.

        ``dtype`` is the precision of the backbone's attention inputs. Families with a
        selectable backend run one tiny call through their own attention path and keep
        SDPA when it fails (see :mod:`ypuddin.models.attention_check`).
        """

    def prepare_backbone_for_plan(self, backbone: nn.Module, cfg: ModelConfig, dtype: torch.dtype) -> None:
        """Match loaded storage on meta, without reading checkpoint tensor payloads."""
        backbone.to(dtype=dtype)

    def cache_memory_estimate(self, cfg: TrainConfig, dtype: torch.dtype) -> dict[str, float]:
        """Optional independent encoder-phase estimates, in MiB (not added to training residency)."""
        return {}

    def training_tokens_for_plan(self, image_tokens: int) -> int:
        return image_tokens

    def adapter_checkpoint_blocks_for_plan(self, backbone: nn.Module) -> list[nn.Module] | None:
        """Adapter recomputation boundaries; None uses the family's memory-layout blocks."""
        return None

    def auxiliary_activation_bytes_for_plan(
        self, backbone: nn.Module, batch_size: int, dtype: torch.dtype
    ) -> int:
        """Saved training activations outside the block layout, excluding weights and adapter fusion."""
        return 0

    def sample_latents(
        self,
        loaded: LoadedModel,
        predict: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        **options: Any,
    ) -> Tensor:
        """Sample using the same prediction convention as this family's training objective."""
        from ypuddin.sampling import sample

        if self.spec.objective != "rectified_flow":
            raise NotImplementedError(f"{self.spec.name}: implement its {self.spec.objective} sampler")
        return sample(predict, shape, **options)

    @abstractmethod
    def presets(self) -> dict[str, TargetPreset]: ...

    def default_preset(self) -> str:
        return "attn-mlp"

    def linear_module_names(self) -> list[str]:
        """Dotted names of the backbone's linear layers on the official geometry (built on the meta device)."""
        return []

    def adaptable_modules(self) -> dict[str, tuple[int, ...]]:
        """The backbone layers adapters can train on the official geometry: kernel size by dotted name,
        ``()`` for a linear layer."""
        return dict.fromkeys(self.linear_module_names(), ())

    def memory_layout(self, loaded: LoadedModel) -> MemoryLayout:
        return MemoryLayout()

    def blocks(self, loaded: LoadedModel) -> Iterator[nn.Module]:
        return iter(self.memory_layout(loaded).blocks)

    def sampling_shift(self, num_tokens: int, objective: Any | None = None) -> float:
        """Timestep shift used for previews of an image with ``num_tokens`` DiT tokens.

        Families with a fixed inference shift return ``spec.sampling.shift``; resolution-aware ones
        (Flux / Krea 2 style ``mu`` interpolation) override this.
        """
        if self.spec.sampling.shift is None:
            raise NotImplementedError(
                f"{self.spec.name}: sampling shift is resolution dependent, override sampling_shift()"
            )
        return float(self.spec.sampling.shift)

    def sampling_defaults(self, loaded: LoadedModel) -> SamplingDefaults:
        """Resolved model variant defaults; explicit user preview settings take precedence."""
        return self.spec.sampling

    def sampling_needs_uncond(self, loaded: LoadedModel, cfg: float) -> bool:
        """Whether the family's public CFG convention needs negative conditioning."""
        return cfg != 1.0

    def sampling_shift_for_model(
        self, loaded: LoadedModel, num_tokens: int, objective: Any | None = None, *, steps: int | None = None
    ) -> float:
        shift = self.sampling_defaults(loaded).shift
        return float(shift) if shift is not None else self.sampling_shift(num_tokens, objective)

    def validate_config(self, cfg: ModelConfig) -> list[str]:
        """Return human-readable problems (missing paths etc.) without loading weights."""
        return []

    def sampling_errors(self, cfg: SamplingConfig) -> list[dict[str, str]]:
        if self.spec.retired_reason:
            return [{"loc": "model.family", "msg": self.spec.retired_reason}]
        problems = []
        for key, allowed in (
            ("sampler", self.spec.sampling_samplers),
            ("scheduler", self.spec.sampling_schedulers),
        ):
            if getattr(cfg, key) not in allowed:
                problems.append(
                    {
                        "loc": f"sampling.{key}",
                        "msg": f"{self.spec.label or self.spec.name}: choose {', '.join(allowed)}",
                    }
                )
        if self.spec.objective == "ddpm" and cfg.shift not in (None, 1.0):
            problems.append(
                {
                    "loc": "sampling.shift",
                    "msg": "SDXL uses its diffusion schedule; Flow timestep shift must be unset or 1",
                }
            )
        return problems

    def training_options_errors(self, cfg: TrainConfig) -> list[dict[str, str]]:
        if self.spec.retired_reason:
            return [{"loc": "model.family", "msg": self.spec.retired_reason}]
        problems = []
        for key in ("scale_v_pred_loss_like_noise_pred", "v_pred_like_loss", "debiased_estimation_loss"):
            if getattr(cfg.objective, key) and self.spec.objective != "ddpm":
                problems.append(
                    {"loc": f"objective.{key}", "msg": "此损失加权仅适用于 SDXL，不适用于 Flow 模型"}
                )
        if self.spec.objective == "ddpm":
            if (
                cfg.objective.scale_v_pred_loss_like_noise_pred
                and cfg.model.prediction_type != "v_prediction"
            ):
                problems.append(
                    {"loc": "objective.scale_v_pred_loss_like_noise_pred", "msg": "此损失缩放仅适用于 v 预测"}
                )
            if cfg.objective.v_pred_like_loss and cfg.model.prediction_type != "epsilon":
                problems.append(
                    {"loc": "objective.v_pred_like_loss", "msg": "附加 v 预测损失仅适用于 ε 预测"}
                )
        for key, allowed in (
            ("timestep_sampling", self.spec.objective_timestep_sampling),
            ("weighting", self.spec.objective_weighting),
        ):
            if getattr(cfg.objective, key) not in allowed:
                problems.append(
                    {
                        "loc": f"objective.{key}",
                        "msg": f"{self.spec.label or self.spec.name}: choose {', '.join(allowed)}",
                    }
                )
        if cfg.sampling.enabled:
            problems.extend(self.sampling_errors(cfg.sampling))
        if cfg.memory.base_precision.startswith("fp8") and "fp8_base" not in self.spec.capabilities:
            problems.append(
                {"loc": "memory.base_precision", "msg": "This model family does not support FP8 base weights"}
            )
        return problems

    def latent_fingerprint(self, cfg: ModelConfig, *, dtype: torch.dtype) -> str:
        """Identity used by cache-coverage queries without loading the VAE."""
        return self.spec.latent.fingerprint
