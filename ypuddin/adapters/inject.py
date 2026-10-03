"""Rewrite a model's module tree with ``AdaptedLinear``/``AdaptedConv`` wrappers and manage the resulting set."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.config import AdapterConfig

from .base import AdapterModule, refuse_tucker
from .conv import CONV_TYPES, AdaptedConv
from .dora import resolve_merge_dtype
from .frozen import FrozenLinear
from .full import Full
from .linear import AdaptedLinear
from .loha import LoHa
from .lokr import LoKr
from .lora import LoRA
from .ortho import OrthoLoRA
from .rules import ResolvedTarget, TargetPreset, resolve_targets
from .tlora import TLoRA, rank_mask

log = logging.getLogger(__name__)

ALGOS: dict[str, type[AdapterModule]] = {
    "lora": LoRA,
    "lokr": LoKr,
    "loha": LoHa,
    "full": Full,
    "ortho": OrthoLoRA,
    "tlora": TLoRA,
}
PARAM_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
_FLOAT_DTYPES = {**PARAM_DTYPES, "fp16": torch.float16}


def build_adapter(
    algo: str,
    out_features: int,
    in_features: int,
    params: dict[str, Any],
    dtype: torch.dtype,
    *,
    kernel: tuple[int, ...] = (),
    bias: bool = False,
) -> AdapterModule:
    """``kernel`` is a convolution's kernel size (``()`` for a linear layer); ``bias`` whether the
    layer has one, which LyCORIS Full trains with the weight."""
    cls = ALGOS[algo]
    kw = dict(params)
    lr = kw.pop("lr", None)  # consumed by param groups, not by the module
    tlora = {key: kw.pop(f"tlora_{key}", None) for key in ("min_rank", "power", "ortho")}
    if algo == "tlora":
        kw.update({key: value for key, value in tlora.items() if value is not None})
    if algo != "lokr":
        kw.pop("factor", None)
        kw.pop("decompose_both", None)
        if kw.get("rank") == "full":
            raise ValueError(f"rank='full' is only valid for lokr (module algo={algo})")
    if algo == "full":
        kw = {"bias": bias}
    mod = cls(out_features, in_features, kernel=kernel, dtype=dtype, **kw)
    mod.lr_override = lr  # type: ignore[attr-defined]
    return mod


def kohya_key(name: str, prefix: str) -> str:
    return f"{prefix}_{name.replace('.', '_')}"


@dataclass
class AdapterSet:
    """All adapted layers of one model plus bookkeeping for training and IO."""

    model: nn.Module
    layers: dict[str, AdaptedLinear | AdaptedConv]
    targets: list[ResolvedTarget]
    config: AdapterConfig
    prefix: str = "lora_unet"
    _originals: dict[str, nn.Module] | None = None
    # File keys name layers below this module path ("model." of a causal-LM text encoder).
    export_root: str = ""
    # The prefix files used before, read back with the full module path.
    legacy_prefix: str | None = None

    def export_key(self, name: str) -> str:
        return kohya_key(name.removeprefix(self.export_root), self.prefix)

    def file_keys(self, name: str) -> tuple[str, ...]:
        """Keys a file may carry for a layer: the current name, then the one older files used."""
        keys = (self.export_key(name),)
        return keys + (kohya_key(name, self.legacy_prefix),) if self.legacy_prefix else keys

    # ----------------------------------------------------------------- training surface
    def parameters(self) -> list[nn.Parameter]:
        return [p for layer in self.layers.values() for p in layer.parameters() if p.requires_grad]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        group_lr: dict[str, float] | None = None,
        *,
        name_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Group parameters by (lr, weight_decay). Priority: rule ``lr`` > ``group_lr`` by name
        substring > ``base_lr``; then ``adapter.lr_scale`` by parameter kind. ``w1`` (LoKr) and
        DoRA magnitudes are excluded from weight decay."""
        buckets: dict[tuple[float, float, str], list[nn.Parameter]] = {}
        for name, layer in self.layers.items():
            lr = getattr(layer.adapter, "lr_override", None)
            if lr is None and group_lr:
                for key, v in group_lr.items():
                    if key in name_prefix + name:
                        lr = v
                        break
            lr = base_lr if lr is None else lr
            kinds = layer.adapter.param_kinds()
            for pname, p in layer.adapter.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                kind = kinds.get(pname, pname)
                scale = self.config.lr_scale.get(kind, 1.0)
                # OrthoLoRA's scales start at 1; decaying them would shrink the principal directions.
                wd = 0.0 if kind in ("w1", "scalar", "scale") else weight_decay
                buckets.setdefault((lr * scale, wd, kind), []).append(p)
            if layer.dora is not None:
                buckets.setdefault((lr, 0.0, "dora"), []).append(layer.dora.dora_scale)
        groups = [
            {"params": ps, "lr": lr, "weight_decay": wd, "name": kind}
            for (lr, wd, kind), ps in buckets.items()
        ]
        groups.sort(key=lambda g: (g["name"], -g["lr"]))
        return groups

    def set_noise_level(self, level: Tensor | None) -> None:
        """Each sample's noise level (0 clean … 1 pure noise) for T-LoRA layers; ``None`` uses every rank."""
        masks: dict[tuple[int, int, float], Tensor] = {}
        for layer in self.layers.values():
            adapter = layer.adapter
            if not isinstance(adapter, TLoRA):
                continue
            if level is None:
                adapter.set_mask(None)
                continue
            key = (adapter.rank, adapter.min_rank, adapter.power)
            if key not in masks:
                masks[key] = rank_mask(level, *key)
            adapter.set_mask(masks[key])

    def set_multiplier(self, m: float) -> None:
        for layer in self.layers.values():
            layer.multiplier = float(m)

    def train(self, mode: bool = True) -> None:
        for layer in self.layers.values():
            layer.adapter.train(mode)

    # ----------------------------------------------------------------- io surface
    def training_state_dict(self) -> dict[str, Tensor]:
        """Raw trainable state, without the scalar/alpha folding used by inference exports."""
        state: dict[str, Tensor] = {}
        for name, layer in self.layers.items():
            for key, value in layer.adapter.state_dict().items():
                state[f"{name}.adapter.{key}"] = value.detach().clone()
            if layer.dora is not None:
                for key, value in layer.dora.state_dict().items():
                    state[f"{name}.dora.{key}"] = value.detach().clone()
        return state

    def load_training_state(self, state: dict[str, Tensor]) -> None:
        expected = {
            f"{name}.{kind}.{key}"
            for name, layer in self.layers.items()
            for kind, module in (("adapter", layer.adapter), ("dora", layer.dora))
            if module is not None
            for key in module.state_dict()
        }
        if state.keys() != expected:
            missing = sorted(expected - state.keys())
            extra = sorted(state.keys() - expected)
            raise ValueError(
                f"checkpoint adapter structure changed: missing={missing[:3]}, extra={extra[:3]}"
            )
        for name, layer in self.layers.items():
            prefix = f"{name}.adapter."
            layer.adapter.load_state_dict(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            )
            if layer.dora is not None:
                layer.dora.load_tensor(state[f"{name}.dora.dora_scale"])

    def export_state(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        tensors: dict[str, Tensor] = {}
        targets_meta: dict[str, Any] = {}
        for name, layer in self.layers.items():
            key = self.export_key(name)
            for suffix, t in layer.adapter.export_tensors().items():
                tensors[f"{key}.{suffix}"] = t
            if layer.dora is not None:
                tensors[f"{key}.dora_scale"] = layer.dora.export_tensor()
            targets_meta[name] = layer.adapter.extra_metadata() | {
                "dora": layer.dora is not None,
                "mode": layer.mode,
            }
        return tensors, targets_meta

    def load_state(
        self, tensors: dict[str, Tensor], *, strict: bool = True, unused_dora: list[str] | None = None,
    ) -> list[str]:
        """Load exported tensors into existing layers (same architecture). Returns missing layer names;
        ``unused_dora`` collects layers whose file has a DoRA magnitude the layer does not train."""
        missing = []
        for name, layer in self.layers.items():
            for key in self.file_keys(name):
                sub = {k[len(key) + 1 :]: v for k, v in tensors.items() if k.startswith(key + ".")}
                if sub:
                    break
            if not sub:
                missing.append(name)
                continue
            dora = sub.pop("dora_scale", None)
            refuse_tucker(sub)
            rebuilt = type(layer.adapter).from_tensors(
                sub,
                layer.adapter.extra_metadata(),
                dtype=layer.adapter.param_dtype,
                kernel=layer.adapter.kernel,
            )
            if isinstance(rebuilt, Full):
                rebuilt.bind_base(layer.frozen_weight(torch.float32), bias=layer.bias)
            layer.adapter.load_state_dict(rebuilt.state_dict(), strict=False)
            # Inference files fold scalar into a factor. A warm start uses that factor with gain 1;
            # exact training resume uses load_training_state instead.
            if layer.adapter.scalar is not None:
                with torch.no_grad():
                    layer.adapter.scalar.fill_(1.0)
            if dora is not None and layer.dora is not None:
                layer.dora.load_tensor(dora)
            elif dora is not None and unused_dora is not None:
                unused_dora.append(name)
        if missing and strict:
            raise KeyError(f"{len(missing)} adapted layers have no tensors in the file, e.g. {missing[:3]}")
        return missing

    # ----------------------------------------------------------------- restore
    def eject(self) -> None:
        """Put the original ``nn.Linear``/convolution modules back (weights untouched)."""
        for name, layer in self.layers.items():
            parent, attr = _locate(self.model, name)
            original = (self._originals or {}).get(name)
            setattr(parent, attr, layer.unwrapped() if original is None else original)
        self.layers.clear()

    def summary(self) -> dict[str, Any]:
        by_algo: dict[str, int] = {}
        for layer in self.layers.values():
            by_algo[layer.adapter.kind] = by_algo.get(layer.adapter.kind, 0) + 1
        summary: dict[str, Any] = {
            "layers": len(self.layers),
            "trainable_params": self.num_params(),
            "by_algo": by_algo,
        }
        conv = sum(isinstance(layer, AdaptedConv) for layer in self.layers.values())
        if conv:
            summary["conv_layers"] = conv
        return summary


# DoRA rescales low-rank updates. A rule that trains a layer with LyCORIS Full or T-LoRA leaves the
# run's DoRA off there, as LyCORIS does: its Full files carry no dora_scale, T-LoRA cannot merge per
# sample. Choosing such an algorithm for the whole run together with DoRA is still refused.
DORA_ALGOS = frozenset({"lora", "loha", "lokr", "ortho"})


def wants_dora(cfg: AdapterConfig, algo: str, out_features: int, *, keep: bool = False) -> bool:
    """Whether a layer trains the run's DoRA.

    A layer with one output channel trains without it. ComfyUI and Forge tell the axes apart by the
    magnitude's first dimension, which a per-input magnitude ``(1, in)`` shares with such a layer, so they
    apply it per output; per input channel, normalising one row leaves only the signs of ``W₀ + ΔW``.
    ``keep`` retains DoRA on such a layer where a resumed training state already holds its magnitude.
    """
    return bool(cfg.dora) and (algo in DORA_ALGOS or algo == cfg.algo) and (out_features > 1 or keep)


def adaptable_modules(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Every layer adapters can train, in module order: its kernel size, ``()`` for a linear layer.

    Linear layers include those a loader already froze (fp8_scaled checkpoints keep their own scales).
    """
    return {
        name: tuple(module.kernel_size) if isinstance(module, CONV_TYPES) else ()
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, FrozenLinear, *CONV_TYPES))
    }


def _locate(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject(
    model: nn.Module,
    cfg: AdapterConfig,
    preset: TargetPreset,
    *,
    prefix: str = "lora_unet",
    base_precision: str = "keep",
    extra_exclude: tuple[str, ...] = (),
    keep_originals: bool = False,
    dora_save_dtype: torch.dtype | str | None = None,
    compute_dtype: torch.dtype | None = None,
    keep_dora: Collection[str] = (),
) -> AdapterSet:
    """Replace every selected ``nn.Linear`` with ``AdaptedLinear`` and convolution with ``AdaptedConv``.

    ``base_precision`` controls the storage of frozen linear base weights (``keep``/``bf16``/``fp8_e4m3``...);
    a convolution keeps its own module and dtype. ``dora_save_dtype`` is the actual
    checkpoint output dtype. ``compute_dtype`` resolves automatic DoRA fusion for FP8 bases.
    ``keep_dora`` names single-output layers (module names or export keys) whose resumed state holds a
    DoRA magnitude; they keep training it.
    """
    modules = adaptable_modules(model)
    targets = resolve_targets(modules, cfg, preset, extra_exclude=extra_exclude)
    if not targets:
        raise ValueError(f"no modules matched preset {preset.name!r} and rules")
    dtype = PARAM_DTYPES[cfg.param_dtype]
    save_dtype = _FLOAT_DTYPES[dora_save_dtype] if isinstance(dora_save_dtype, str) else dora_save_dtype
    layers: dict[str, AdaptedLinear | AdaptedConv] = {}
    originals: dict[str, nn.Module] = {}
    for t in targets:
        parent, attr = _locate(model, t.name)
        module = getattr(parent, attr)
        kernel = modules[t.name]
        out_features, in_features = module.weight.shape[:2]  # a grouped convolution's per-group inputs
        # Planning injects into a meta-device model; its adapters need no real storage.
        with torch.device("meta") if module.weight.is_meta else contextlib.nullcontext():
            adapter = build_adapter(
                t.algo,
                int(out_features),
                int(in_features),
                t.params,
                dtype,
                kernel=kernel,
                bias=module.bias is not None,
            )
        adapter.to(module.weight.device)
        if kernel:
            base, wrapper = module, AdaptedConv
        else:
            # A layer a loader already froze (fp8 weight + scale from the checkpoint) is kept as-is.
            frozen = isinstance(module, FrozenLinear)
            base = module if frozen else FrozenLinear.from_linear(module, precision=base_precision)
            wrapper = AdaptedLinear
        compute_mode = getattr(cfg, "dora_compute_mode", "standard")
        merge_dtype = resolve_merge_dtype(getattr(cfg, "dora_merge_dtype", "auto"), base, compute_dtype)
        kept = t.name in keep_dora or kohya_key(t.name, prefix) in keep_dora
        layer = wrapper(
            base,
            adapter,
            mode=cfg.mode,
            dora=wants_dora(cfg, t.algo, int(out_features), keep=kept),
            dora_axis=cfg.dora_axis,
            dora_compute_mode=compute_mode,
            dora_merge_dtype=merge_dtype,
            dora_save_dtype=save_dtype,
            module_dropout=cfg.module_dropout,
            name=t.name,
        )
        setattr(parent, attr, layer)
        layers[t.name] = layer
        if keep_originals:
            originals[t.name] = module
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers.values():
        for p in layer.adapter.parameters():
            p.requires_grad_(True)
        if layer.dora is not None:
            layer.dora.dora_scale.requires_grad_(True)
    aset = AdapterSet(
        model, layers, targets, cfg, prefix=prefix, _originals=originals if keep_originals else None
    )
    summary = aset.summary()
    log.info(
        "injected adapters into %d layers (%s)%s: %s trainable parameters",
        summary["layers"],
        ", ".join(f"{kind} {count}" for kind, count in summary["by_algo"].items()),
        f", {summary['conv_layers']} of them convolutions" if summary.get("conv_layers") else "",
        f"{summary['trainable_params']:,}",
    )
    return aset
