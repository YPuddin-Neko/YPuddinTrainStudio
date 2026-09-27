"""Target selection: family preset (include/exclude patterns) + ordered user rules."""

from __future__ import annotations

import fnmatch
import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from ypuddin.config import AdapterConfig, AdapterRule


@dataclass(frozen=True)
class TargetPreset:
    """A named set of module-name patterns provided by a model family."""

    name: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    description: str = ""


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


def resolve_targets(
    module_names: list[str],
    cfg: AdapterConfig,
    preset: TargetPreset,
    *,
    extra_exclude: tuple[str, ...] = (),
) -> list[ResolvedTarget]:
    """Decide, for every candidate linear module, whether and how it is adapted.

    Order of precedence per module: first matching user rule (if any) > preset. A rule may
    include a module the preset would not have selected; ``algo="none"`` excludes it.
    """
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
    out: list[ResolvedTarget] = []
    for name in module_names:
        rule = next((r for r in cfg.rules if match_name(r.match, name)), None)
        in_preset = any(match_name(p, name) for p in preset.include) and not any(
            match_name(p, name) for p in (*preset.exclude, *extra_exclude)
        )
        if rule is None:
            if not in_preset:
                continue
            out.append(ResolvedTarget(name, cfg.algo, dict(defaults)))
            continue
        algo = rule.algo or cfg.algo
        if algo == "none":
            continue
        params = dict(defaults)
        params.update(_rule_params(rule))
        out.append(ResolvedTarget(name, algo, params))
    return out
