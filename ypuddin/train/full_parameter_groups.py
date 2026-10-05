"""Learning-rate selection shared by full training and meta-device planning."""

from __future__ import annotations

from torch import nn

FULL_LR_FIELDS = (
    "backbone_lr", "text_encoder_lr", "text_encoder_2_lr", "llm_adapter_lr",
    "self_attn_lr", "cross_attn_lr", "mlp_lr", "modulation_lr",
)


def parameter_category(name: str, family: str) -> str:
    component = name.split(".", 1)[0]
    if component != "backbone" or family != "anima":
        return component
    parts = name.split(".")
    if "llm_adapter" in parts:
        return "llm_adapter"
    if any(part.startswith("adaln_modulation") for part in parts):
        return "modulation"
    for category in ("self_attn", "cross_attn", "mlp"):
        if category in parts:
            return category
    return component


def parameter_rate(name: str, cfg, base_lr: float, group_lr: dict | None):
    """Return effective rate and its source; preserve ordered legacy substring rules."""
    component = name.split(".", 1)[0]
    category = parameter_category(name, cfg.model.family) if cfg else component
    selection = cfg.training if cfg else None
    specialized = getattr(selection, f"{category}_lr", None) if category != component else None
    if specialized is not None:
        return specialized, f"training.{category}_lr"
    for key, value in (group_lr or {}).items():
        if key in name:
            return value, f"optimizer.group_lr.{key}"
    component_rate = getattr(selection, f"{component}_lr", None)
    if component_rate is not None:
        return component_rate, f"training.{component}_lr"
    return base_lr, "optimizer.lr"


def no_decay_names(modules: dict[str, nn.Module]) -> set[str]:
    names = set()
    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._NormBase,
                  nn.modules.instancenorm._InstanceNorm)
    if hasattr(nn, "RMSNorm"):
        norm_types += (nn.RMSNorm,)
    for component, module in modules.items():
        for path, child in module.named_modules():
            # Native model RMSNorm classes need not subclass torch.nn.RMSNorm.
            norm = isinstance(child, norm_types) or child.__class__.__name__.lower().endswith("norm")
            for name, _ in child.named_parameters(recurse=False):
                if norm or name == "bias":
                    names.add(".".join(filter(None, (component, path, name))))
    return names
