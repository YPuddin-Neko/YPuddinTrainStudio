"""Safe recipes for a new model family, independent from dataset snapshotting."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig
from ypuddin.models import get_family

from .environment import environment_attention_default
from .errors import ApiError

SUPPORTED_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2", "toy"})
MODEL_PATHS = {
    "dit": "dit_path",
    "text_encoder": "text_encoder_path",
    "text_encoder_2": "text_encoder_2_path",
    "vae": "vae_path",
    "tokenizer": "tokenizer_path",
}


def _check_family(family: str) -> None:
    if family not in SUPPORTED_FAMILIES:
        raise ApiError(f"unsupported training family: {family}", code="version.family", status=422)


def _default_model_paths(c: Any, family: str) -> dict[str, str]:
    """Only reuse registered defaults whose role and current filesystem type still match."""
    paths: dict[str, str] = {}
    for row in c.db.fetchall(
        "SELECT kind,path FROM models WHERE family=? AND is_default=1 ORDER BY created_at DESC,id",
        (family,),
    ):
        kind = row["kind"]
        field = MODEL_PATHS.get(kind)
        if field is None or field in paths:
            continue
        try:
            path = Path(row["path"]).expanduser().resolve()
            valid = (
                path.is_file()
                if kind in {"dit", "vae"} and family not in {"sdxl", "flux", "flux2"}
                else path.is_dir()
                if kind == "tokenizer"
                else path.is_file() or path.is_dir()
            )
            if valid and c.is_allowed(path):
                if family == "flux2":
                    from .model_inspection import inspect_model, training_rejection

                    if kind == "text_encoder" and not path.is_dir():
                        continue
                    if training_rejection(inspect_model(path, allowed=c.is_allowed), family):
                        continue  # Retain legacy dev rows without selecting them for a new Klein recipe.
                paths[field] = str(path)
        except (OSError, RuntimeError, ValueError):
            continue
    return paths


def initial_family_config(c: Any, family: str) -> dict[str, Any]:
    """Return a complete family recipe without loading weights or assuming a CUDA device."""
    _check_family(family)
    model_family = get_family(family)
    spec = model_family.spec
    config = TrainConfig().to_dict()
    config["dataset"]["image_fit"] = "pad"
    config["model"].update(family=family, attention=environment_attention_default(c))
    if config["model"]["attention"] not in spec.attention_backends:
        config["model"]["attention"] = "auto"
    config["adapter"]["preset"] = model_family.default_preset()
    config["sampling"].update(
        steps=spec.sampling.steps,
        cfg=spec.sampling.cfg,
        shift=spec.sampling.shift,
        sampler=spec.sampling.sampler,
        guidance=spec.sampling.guidance,
    )
    config["dataset"]["text_encoding"] = "auto" if "online_text" in spec.capabilities else "cached"
    if family == "toy":
        config["model"]["dtype"] = "fp32"
        config["dataset"].update(resolutions=[64], bucket_step=16, batch_size=2, num_workers=0)
        config["loop"].update(epochs=1, mixed_precision="no")
        config["sampling"].update(width=64, height=64)
    if spec.objective == "ddpm":
        config["objective"].update(timestep_sampling="uniform", weighting="none")
        config["sampling"].update(shift=1.0, scheduler="uniform")
        config["dataset"]["text_encoding"] = "cached"
        config["memory"]["activation_checkpointing"] = "block"
    elif spec.sampling.shift is None:
        config["objective"].update(
            timestep_sampling="resolution_shift", res_shift_tokens=[256, 6400 if family == "krea2" else 4096]
        )
    else:
        config["objective"].update(timestep_sampling="shift", shift=spec.sampling.shift)
    if family in {"krea2", "flux2"}:
        config["memory"]["activation_checkpointing"] = "block"
    if family == "flux2":
        # Resolve Klein sampling defaults from the selected checkpoint.
        config["sampling"].update(steps=None, cfg=None, guidance=None)
    config["model"].update(_default_model_paths(c, family))
    return config


def change_config_family(c: Any, config: dict[str, Any], family: str) -> dict[str, Any]:
    """Same-family recipes are preserved; cross-family recipes retain only data preparation."""
    _check_family(family)
    if config.get("model", {}).get("family", "anima") == family:
        return deepcopy(config)
    changed = initial_family_config(c, family)
    text_encoding = changed["dataset"]["text_encoding"]
    for section in ("dataset", "validation"):
        changed[section].update(deepcopy(config.get(section, {})))
    # Family changes retain the previous data transform, including legacy files
    # with no explicit image_fit; only an actually new recipe defaults to padding.
    changed["dataset"]["image_fit"] = config.get("dataset", {}).get("image_fit", "crop")
    # The text-conditioning implementation belongs to the target family, not the copied images.
    changed["dataset"]["text_encoding"] = text_encoding
    return changed


def version_family(c: Any, row: dict[str, Any]) -> str | None:
    """Project-list projection: unavailable or malformed configuration is not a list failure."""
    try:
        path = c.config_path(row["project_id"], row["id"])
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return "anima" if row["status"] == "ready" else None
        if not isinstance(config, dict) or not isinstance(config.get("model", {}), dict):
            return None
        family = config.get("model", {}).get("family", "anima")
        return family if isinstance(family, str) and family in SUPPORTED_FAMILIES | {"flux"} else None
    except (OSError, UnicodeError, ValueError, ApiError):
        return None
