"""One portable adapter artifact spanning backbone and selected text encoders."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping

import torch
from torch import nn

from .inject import AdapterSet, inject
from .rules import TargetPreset

TEXT_ADAPTER_PREFIXES = {"text_encoder": "lora_te1", "text_encoder_2": "lora_te2"}
# A family with one text encoder (Anima, Krea 2, FLUX.2) names it as kohya and ComfyUI do, below
# the model root; SDXL's pair keeps te1/te2. Files from before read back with TEXT_ADAPTER_PREFIXES.
SINGLE_TEXT_ADAPTER_PREFIX = "lora_te"
TEXT_ADAPTER_PRESET = TargetPreset(
    "text_encoder", ("*",), ("lm_head",), "文本编码器的线性层，不训练词嵌入或语言模型输出头"
)


class ComponentAdapterSet:
    """Retain each component's export prefix and isolate its raw training keys."""

    def __init__(self, components: dict[str, AdapterSet]):
        if not components or len({item.prefix for item in components.values()}) != len(components):
            raise ValueError("adapter components require unique, nonempty export prefixes")
        self.components = components
        self.modules = {name: item.model for name, item in components.items()}
        self.layers = {
            f"{component}.{name}": layer
            for component, item in components.items()
            for name, layer in item.layers.items()
        }

    def parameters(self):
        return list({id(p): p for item in self.components.values() for p in item.parameters()}.values())

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, base_lr, weight_decay, group_lr=None):
        groups = []
        for component, item in self.components.items():
            for group in item.param_groups(base_lr, weight_decay, group_lr, name_prefix=component + "."):
                groups.append({**group, "name": f"{component}.{group['name']}"})
        return groups

    def train(self, mode=True):
        for item in self.components.values():
            item.model.train(mode)
            item.train(mode)

    def set_noise_level(self, level):
        # The noise level conditions the image model; text encoder layers always use every rank.
        backbone = self.components.get("backbone")
        if backbone is not None:
            backbone.set_noise_level(level)

    def set_multiplier(self, multiplier):
        for item in self.components.values():
            item.set_multiplier(multiplier)

    def training_state_dict(self):
        return {
            f"{component}.{key}": value
            for component, item in self.components.items()
            for key, value in item.training_state_dict().items()
        }

    def load_training_state(self, state):
        expected = self.training_state_dict()
        if state.keys() != expected.keys() or any(state[k].shape != v.shape for k, v in expected.items()):
            raise ValueError("checkpoint adapter component selection or parameter structure changed")
        for component, item in self.components.items():
            prefix = component + "."
            item.load_training_state({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})

    def export_state(self):
        tensors, targets = {}, {}
        for item in self.components.values():
            values, metadata = item.export_state()
            if tensors.keys() & values.keys():
                raise ValueError("duplicate adapter export keys across components")
            tensors.update(values)
            targets.update({item.export_key(k): v for k, v in metadata.items()})
        return tensors, targets

    def load_state(self, tensors, *, strict=True, unused_dora=None):
        missing, unused = [], []
        for component, item in self.components.items():
            skipped = [] if unused_dora is not None else None
            missing += [f"{component}.{name}" for name in item.load_state(tensors, strict=False, unused_dora=skipped)]
            unused += [f"{component}.{name}" for name in skipped or ()]
        if unused_dora is not None:
            unused_dora.extend(unused)
        if strict and missing:
            raise KeyError(f"adapter file is missing component layers: {missing[:3]}")
        return missing

    def eject(self):
        for item in self.components.values():
            item.eject()
        self.layers.clear()

    def summary(self):
        counts = Counter()
        for item in self.components.values():
            counts.update(item.summary()["by_algo"])
        return {
            "layers": len(self.layers),
            "trainable_params": self.num_params(),
            "by_algo": dict(counts),
            "components": sorted(self.components),
        }


def text_export_root(module: nn.Module) -> str:
    """``model.`` when a causal-LM wrapper nests the encoder layers there (FLUX.2), else empty."""
    root = getattr(module, "base_model_prefix", "") or ""
    return f"{root}." if root and isinstance(getattr(module, root, None), nn.Module) else ""


def inject_text_adapters(
    modules: dict[str, nn.Module], cfg, *, dora_save_dtype: torch.dtype | str | None = None,
    compute_dtype: torch.dtype | None = None, keep_dora: Mapping[str, Collection[str]] | None = None,
):
    """``keep_dora`` names, per component, single-output layers whose resumed state holds a DoRA magnitude."""
    if not modules or not modules.keys() <= TEXT_ADAPTER_PREFIXES.keys():
        raise ValueError("text pipeline returned unsupported component names")
    keep_dora = keep_dora or {}
    if len(modules) == 1:
        ((name, module),) = modules.items()
        adapters = inject(module, cfg, TEXT_ADAPTER_PRESET, prefix=SINGLE_TEXT_ADAPTER_PREFIX,
                          dora_save_dtype=dora_save_dtype, compute_dtype=compute_dtype,
                          keep_dora=keep_dora.get(name, ()))
        adapters.export_root = text_export_root(module)
        adapters.legacy_prefix = TEXT_ADAPTER_PREFIXES[name]
        return {name: adapters}
    return {
        name: inject(module, cfg, TEXT_ADAPTER_PRESET, prefix=TEXT_ADAPTER_PREFIXES[name],
                     dora_save_dtype=dora_save_dtype, compute_dtype=compute_dtype,
                     keep_dora=keep_dora.get(name, ()))
        for name, module in modules.items()
    }
