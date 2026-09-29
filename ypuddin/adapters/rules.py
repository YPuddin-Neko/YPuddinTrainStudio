"""Target selection: family preset (include/exclude patterns) + ordered user rules."""

from __future__ import annotations

import fnmatch
import itertools
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ypuddin.config import AdapterConfig, AdapterRule


@dataclass(frozen=True)
class TargetPreset:
    """A named set of module-name patterns provided by a model family.

    ``include`` selects linear layers. ``conv`` names what training convolutions as well adds:
    convolution layers, and linear layers that belong to the same convolution blocks (SDXL's
    ``time_emb_proj``), as kohya's LoCon scope does. It applies only when the configuration chooses
    linear and convolution layers; convolutions are never selected otherwise.
    """

    name: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    description: str = ""
    conv: tuple[str, ...] = ()


@dataclass
class ResolvedTarget:
    name: str
    algo: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "algo": self.algo, **self.params}


def _expand_braces(pattern: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    head, tail = pattern[: m.start()], pattern[m.end() :]
    return list(
        itertools.chain.from_iterable(_expand_braces(head + opt + tail) for opt in m.group(1).split(","))
    )


def match_name(pattern: str, name: str) -> bool:
    """``re:`` prefix -> regex fullmatch; otherwise glob with ``{a,b}`` brace expansion."""
    if pattern.startswith("re:"):
        return re.fullmatch(pattern[3:], name) is not None
    return any(fnmatch.fnmatchcase(name, p) for p in _expand_braces(pattern))


def _rule_params(rule: AdapterRule) -> dict[str, Any]:
    return {k: v for k, v in rule.model_dump().items() if k not in ("match", "algo") and v is not None}


def trains_convolutions(cfg: AdapterConfig, preset: TargetPreset) -> bool:
    """Whether this configuration adds adapters to the preset's convolution scope."""
    return cfg.layer_types == "linear_conv" and bool(preset.conv)


def resolve_targets(
    modules: Mapping[str, tuple[int, ...]] | Iterable[str],
    cfg: AdapterConfig,
    preset: TargetPreset,
    *,
    extra_exclude: tuple[str, ...] = (),
) -> list[ResolvedTarget]:
    """Decide, for every candidate module, whether and how it is adapted.

    ``modules`` maps each candidate layer to its kernel size, ``()`` for a linear layer; a plain list
    names linear layers. Order of precedence per module: first matching user rule (if any) > preset. A
    rule may include a module the preset would not have selected; ``algo="none"`` excludes it.
    Convolutions are candidates only when the configuration trains them and the preset has a
    convolution scope; a kernel larger than 1×1 takes ``conv_rank``/``conv_alpha`` (kohya and LyCORIS
    ``conv_dim``/``conv_alpha``), a 1×1 convolution the linear rank.
    """
    if not isinstance(modules, Mapping):
        modules = dict.fromkeys(modules, ())
    conv = trains_convolutions(cfg, preset)
    defaults: dict[str, Any] = {
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "factor": cfg.factor,
        "dropout": cfg.dropout,
        "rank_dropout": cfg.rank_dropout,
        "decompose_both": cfg.decompose_both,
        "rs_lora": cfg.rs_lora,
        "init": cfg.init,
        "tlora_min_rank": cfg.tlora_min_rank,
        "tlora_power": cfg.tlora_power,
        "tlora_ortho": cfg.tlora_ortho,
    }
    kernel_defaults = defaults | {
        "rank": cfg.rank if cfg.conv_rank is None else cfg.conv_rank,
        "alpha": cfg.alpha if cfg.conv_alpha is None else cfg.conv_alpha,
    }
    linear_patterns = preset.include + (preset.conv if conv else ())
    excluded = (*preset.exclude, *extra_exclude)
    out: list[ResolvedTarget] = []
    for name, kernel in modules.items():
        if kernel and not conv:
            continue
        rule = next((r for r in cfg.rules if match_name(r.match, name)), None)
        patterns = preset.conv if kernel else linear_patterns
        in_preset = any(match_name(p, name) for p in patterns) and not any(
            match_name(p, name) for p in excluded
        )
        base = kernel_defaults if any(size != 1 for size in kernel) else defaults
        if rule is None:
            if not in_preset:
                continue
            out.append(ResolvedTarget(name, cfg.algo, dict(base)))
            continue
        algo = rule.algo or cfg.algo
        if algo == "none":
            continue
        params = dict(base)
        params.update(_rule_params(rule))
        out.append(ResolvedTarget(name, algo, params))
    return out
