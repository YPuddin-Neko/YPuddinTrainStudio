"""Rewrite a model's module tree with ``AdaptedLinear`` wrappers and manage the resulting set."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.config import AdapterConfig

from .base import AdapterModule
from .frozen import FrozenLinear
from .full import Full
from .linear import AdaptedLinear
from .loha import LoHa
from .lokr import LoKr
from .lora import LoRA
from .rules import ResolvedTarget, TargetPreset, resolve_targets

log = logging.getLogger(__name__)

ALGOS: dict[str, type[AdapterModule]] = {"lora": LoRA, "lokr": LoKr, "loha": LoHa, "full": Full}
PARAM_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def build_adapter(
    algo: str, out_features: int, in_features: int, params: dict[str, Any], dtype: torch.dtype
) -> AdapterModule:
    cls = ALGOS[algo]
    kw = dict(params)
    lr = kw.pop("lr", None)  # consumed by param groups, not by the module
    if algo != "lokr":
        kw.pop("factor", None)
        kw.pop("decompose_both", None)
        if kw.get("rank") == "full":
            raise ValueError(f"rank='full' is only valid for lokr (module algo={algo})")
    if algo == "full":
        kw = {}
    mod = cls(out_features, in_features, dtype=dtype, **kw)
    mod.lr_override = lr  # type: ignore[attr-defined]
    return mod


def kohya_key(name: str, prefix: str) -> str:
    return f"{prefix}_{name.replace('.', '_')}"


@dataclass
class AdapterSet:
    """All adapted layers of one model plus bookkeeping for training and IO."""

    model: nn.Module
    layers: dict[str, AdaptedLinear]
    targets: list[ResolvedTarget]
    config: AdapterConfig
    prefix: str = "lora_unet"
    _originals: dict[str, nn.Linear] | None = None

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
                wd = 0.0 if kind in ("w1", "scalar") else weight_decay
                buckets.setdefault((lr * scale, wd, kind), []).append(p)
            if layer.dora is not None:
                buckets.setdefault((lr, 0.0, "dora"), []).append(layer.dora.dora_scale)
        groups = [
            {"params": ps, "lr": lr, "weight_decay": wd, "name": kind}
            for (lr, wd, kind), ps in buckets.items()
        ]
        groups.sort(key=lambda g: (g["name"], -g["lr"]))
        return groups

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
            for kind, module in (("adapter", layer.adapter), ("dora", layer.dora)):
                if module is None:
                    continue
                prefix = f"{name}.{kind}."
                module.load_state_dict(
                    {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
                )

    def export_state(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        tensors: dict[str, Tensor] = {}
        targets_meta: dict[str, Any] = {}
        for name, layer in self.layers.items():
            key = kohya_key(name, self.prefix)
            for suffix, t in layer.adapter.export_tensors().items():
                tensors[f"{key}.{suffix}"] = t
            if layer.dora is not None:
                tensors[f"{key}.dora_scale"] = layer.dora.export_tensor()
            targets_meta[name] = layer.adapter.extra_metadata() | {
                "dora": layer.dora is not None,
                "mode": layer.mode,
            }
        return tensors, targets_meta

    def load_state(self, tensors: dict[str, Tensor], *, strict: bool = True) -> list[str]:
        """Load exported tensors into existing layers (same architecture). Returns missing layer names."""
        missing = []
        for name, layer in self.layers.items():
            key = kohya_key(name, self.prefix)
            sub = {k[len(key) + 1 :]: v for k, v in tensors.items() if k.startswith(key + ".")}
            if not sub:
                missing.append(name)
                continue
            dora = sub.pop("dora_scale", None)
            rebuilt = type(layer.adapter).from_tensors(
                sub, layer.adapter.extra_metadata(), dtype=layer.adapter.param_dtype
            )
            if isinstance(rebuilt, Full):
                rebuilt.bind_base(layer.base.dequant(torch.float32))
            layer.adapter.load_state_dict(rebuilt.state_dict(), strict=False)
            # Inference files fold scalar into a factor. A warm start uses that factor with gain 1;
            # exact training resume uses load_training_state instead.
            if layer.adapter.scalar is not None:
                with torch.no_grad():
                    layer.adapter.scalar.fill_(1.0)
            if dora is not None and layer.dora is not None:
                layer.dora.load_tensor(dora)
        if missing and strict:
            raise KeyError(f"{len(missing)} adapted layers have no tensors in the file, e.g. {missing[:3]}")
        return missing

    # ----------------------------------------------------------------- restore
    def eject(self) -> None:
        """Put the original ``nn.Linear`` modules back (weights untouched)."""
        for name, layer in self.layers.items():
            parent, attr = _locate(self.model, name)
            original = (self._originals or {}).get(name)
            if original is None:
                original = layer.base.to_linear()
            setattr(parent, attr, original)
        self.layers.clear()

    def summary(self) -> dict[str, Any]:
        by_algo: dict[str, int] = {}
        for layer in self.layers.values():
            by_algo[layer.adapter.kind] = by_algo.get(layer.adapter.kind, 0) + 1
        return {"layers": len(self.layers), "trainable_params": self.num_params(), "by_algo": by_algo}


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
) -> AdapterSet:
    """Replace every selected ``nn.Linear`` with ``AdaptedLinear``.

    ``base_precision`` controls the storage of frozen base weights (``keep``/``bf16``/``fp8_e4m3``...).
    """
    # plain Linears plus layers a loader already froze (e.g. fp8_scaled checkpoints keep their own scales)
    linear_names = [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, FrozenLinear))]
    targets = resolve_targets(linear_names, cfg, preset, extra_exclude=extra_exclude)
    if not targets:
        raise ValueError(f"no modules matched preset {preset.name!r} and rules")
    dtype = PARAM_DTYPES[cfg.param_dtype]
    layers: dict[str, AdaptedLinear] = {}
    originals: dict[str, nn.Linear] = {}
    for t in targets:
        parent, attr = _locate(model, t.name)
        linear = getattr(parent, attr)
        if isinstance(linear, FrozenLinear):
            frozen = linear  # already frozen (fp8 weight + scale from the checkpoint): keep as-is
        else:
            frozen = FrozenLinear.from_linear(linear, precision=base_precision)
        adapter = build_adapter(t.algo, linear.out_features, linear.in_features, t.params, dtype)
        adapter.to(linear.weight.device)
        layer = AdaptedLinear(
            frozen, adapter, mode=cfg.mode, dora=cfg.dora, module_dropout=cfg.module_dropout, name=t.name
        )
        setattr(parent, attr, layer)
        layers[t.name] = layer
        if keep_originals:
            originals[t.name] = linear
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
    log.info("injected adapters: %s", aset.summary())
    return aset
