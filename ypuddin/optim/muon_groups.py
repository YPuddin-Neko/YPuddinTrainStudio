"""Semantic matrix selection shared by Muon training and memory planning."""

from __future__ import annotations

from typing import Any

from torch import nn

# Whole component input/output projections stay with AdamW. Hidden attention,
# MLP and modulation matrices remain intact, including fused QKV and AdaLN.
_BOUNDARIES = (
    "x_embedder", "context_embedder", "time_text_embed", "time_embedding", "t_embedder",
    "t_embedding", "video_embedder", "text_embedder.proj", "final_linear_video",
    "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "emb_in", "ref_time_in",
    "final_layer", "proj_in", "proj_out", "conv_in", "conv_out", "pos_embed",
    "first", "last", "tmlp", "txtmlp", "text_projection", "lm_head", "llm_adapter.in_proj", "llm_adapter.out_proj",
)


def _native_name(name: str) -> str:
    return ".".join(part for part in name.split(".") if part != "_orig_mod")


def _boundary(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in _BOUNDARIES)


def _parameter_identity(training_set) -> dict[int, tuple[str, str]]:
    identities: dict[int, tuple[str, str]] = {}
    if hasattr(training_set, "components") or hasattr(training_set, "model"):
        components = getattr(training_set, "components", None) or {"backbone": training_set}
        for component, item in components.items():
            for path, layer in item.layers.items():
                path = _native_name(path)
                for name, parameter in layer.adapter.named_parameters(recurse=False):
                    # LoRA/LoKr optimize their factors, not the reconstructed weight.
                    full_boundary = layer.adapter.__class__.__name__ == "Full" and _boundary(path)
                    role = "muon" if parameter.ndim == 2 and not full_boundary else "adamw"
                    identities[id(parameter)] = (f"{component}.{path}.adapter.{name}", role)
                if layer.dora is not None:
                    identities[id(layer.dora.dora_scale)] = (f"{component}.{path}.dora.dora_scale", "adamw")
        return identities
    for component, module in training_set.modules.items():
        for path, child in module.named_modules():
            path = _native_name(path)
            norm = "norm" in child.__class__.__name__.lower()
            auxiliary = norm or isinstance(child, (nn.Embedding, nn.EmbeddingBag)) or _boundary(path)
            for name, parameter in child.named_parameters(recurse=False):
                role = "adamw" if auxiliary or name == "bias" or parameter.ndim != 2 else "muon"
                full_name = ".".join(filter(None, (component, path, name)))
                previous = identities.get(id(parameter))
                # A tied embedding/output matrix retains its auxiliary role.
                if previous is None or previous[1] != "adamw":
                    identities[id(parameter)] = (full_name, role)
    return identities


def prepare_muon_groups(cfg, groups, training_set=None) -> list[dict[str, Any]]:
    """Split existing groups without changing their internal order or LR rules.

    The Trainer and planner supply the training set for semantic roles. Direct
    optimizer callers can supply explicit roles, or use dimension-based routing.
    """
    identities = _parameter_identity(training_set) if training_set is not None else {}
    result = []
    for group_index, original in enumerate(groups):
        if "muon_role" in original:
            result.append(dict(original))
            continue
        routed: dict[str, list] = {"muon": [], "adamw": []}
        names: dict[str, list[str]] = {"muon": [], "adamw": []}
        provided_names = original.get("param_names")
        if provided_names is not None and len(provided_names) != len(original["params"]):
            raise ValueError("Muon parameter names must match the parameter group")
        for index, parameter in enumerate(original["params"]):
            if training_set is not None:
                if id(parameter) not in identities:
                    raise ValueError("Muon could not identify a training parameter")
                name, role = identities[id(parameter)]
            else:
                name = provided_names[index] if provided_names is not None else f"group{group_index}.parameter{index}"
                role = "muon" if parameter.ndim == 2 else "adamw"
            routed[role].append(parameter)
            names[role].append(name)
        for role in ("muon", "adamw"):
            if not routed[role] and (original["params"] or role == "adamw"):
                continue
            ratio = cfg.aux_lr_ratio if role == "adamw" else 1.0
            group = {
                **original, "params": routed[role], "param_names": names[role],
                "name": f"{original.get('name', group_index)}/{role}", "muon_role": role,
                "lr": original.get("lr", cfg.lr) * ratio, "aux_lr_ratio": cfg.aux_lr_ratio,
            }
            if "initial_lr" in group:
                group["initial_lr"] *= ratio
            result.append(group)
    return result


def muon_parameter_roles(training_set) -> dict[int, str]:
    """Include frozen parameters when reporting full-training groups."""
    return {identity: role for identity, (_, role) in _parameter_identity(training_set).items()}
