"""Resolve class-prior models from the same registry used by model settings."""

from __future__ import annotations

from pathlib import Path

from ypuddin.config import ModelConfig, SamplingConfig

from .errors import ApiError
from .family_config import MODEL_PATHS, _default_model_paths


def _allowed(c, path):
    if not c.is_allowed(Path(path).expanduser().resolve()):
        raise ApiError("Model path is outside permitted roots", status=403, code="regularization.path")


def _package_paths(c, model):
    """Keep a package's bindings ahead of unrelated component defaults."""
    root = Path(model.dit_path).expanduser()
    paths = {}
    package = False
    if model.family in {"flux2", "sdxl"} and (root / "model_index.json").is_file():
        from .model_inspection import CONFIG_LIMIT, _json

        _allowed(c, root / "model_index.json")
        class_name = str(_json(root / "model_index.json", CONFIG_LIMIT).get("_class_name", ""))
        package = class_name == "Flux2KleinPipeline" if model.family == "flux2" else class_name.startswith("StableDiffusionXL")
    if model.family in {"flux3", "moss_image"}:
        if model.family == "flux3":
            from ypuddin.models.flux3.loading import component_paths
        else:
            from ypuddin.models.moss_image.loading import component_paths
        directory = root if root.is_dir() else root.parent
        if model.family == "moss_image" and directory.name == "transformer":
            directory = directory.parent
        descriptor = directory / f".ypuddin-{model.family.replace('_', '-')}-components.json"
        bound = descriptor.exists() or descriptor.is_symlink()
        if bound:
            _allowed(c, descriptor)
        _, _, text, vae, tokenizer = component_paths(model)
        for role, path in (("text_encoder_path", text), ("vae_path", vae), ("tokenizer_path", tokenizer)):
            exists = (path / "tokenizer_config.json").is_file() if role == "tokenizer_path" else path.exists()
            if bound or exists:
                paths[role] = str(path)
    elif model.family == "flux2":
        for role, name in (("text_encoder_path", "text_encoder"), ("vae_path", "vae"),
                           ("tokenizer_path", "tokenizer")):
            if package or (root / name).exists():
                paths[role] = str(root / name)
    elif model.family == "sdxl":
        from ypuddin.models.sdxl.loading import component_keys

        keys = None
        if root.is_file() and root.suffix.lower() == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(root), framework="pt", device="cpu") as weights:
                keys = list(weights.keys())
        for role, name in (("text_encoder_path", "text_encoder"), ("text_encoder_2_path", "text_encoder_2"),
                           ("vae_path", "vae")):
            if package or (root / name).is_dir():
                paths[role] = str(root / name)
            elif keys is not None:
                try:
                    component_keys(keys, name)
                except ValueError:
                    continue
                paths[role] = str(root)
        if package or ((root / "tokenizer").is_dir() and (root / "tokenizer_2").is_dir()):
            paths["tokenizer_path"] = str(root)
    return paths


def _component_defaults(c, family):
    from .routes_core import _check_model_admission, _model_row

    paths = _default_model_paths(c, family)
    for row in c.db.fetchall(
        "SELECT * FROM models WHERE family!=? AND kind!='dit' AND is_default=1 ORDER BY created_at DESC,id", (family,)
    ):
        field = MODEL_PATHS.get(row["kind"])
        if field is None or field in paths or field == "tokenizer_path":
            continue
        projected = _model_row(row, c)
        if (not projected["exists"] or projected.get("unsupported_reason")
                or family not in projected.get("compatible_families", [])):
            continue
        try:
            path = Path(row["path"]).expanduser().resolve()
            _allowed(c, path)
            processor = projected.get("processor_path")
            _check_model_admission(path, family, row["kind"], c, processor_path=processor)
        except (ApiError, OSError, ValueError, RuntimeError):
            continue
        paths[field] = str(path)
        if row["kind"] == "text_encoder" and processor:
            paths.setdefault("tokenizer_path", processor)
    return paths


def _registered_config(c, row):
    from .model_recommendations import component_metadata

    model = ModelConfig(family=row["family"], dit_path=row["path"])
    _allowed(c, model.dit_path)
    paths = _package_paths(c, model)
    defaults = _component_defaults(c, model.family)
    for field, path in defaults.items():
        if field not in {"dit_path", "tokenizer_path"}:
            paths.setdefault(field, path)
    text = paths.get("text_encoder_path")
    if text and "tokenizer_path" not in paths:
        processor = component_metadata(c, Path(text)).get("processor_path")
        if processor:
            paths["tokenizer_path"] = processor
        elif text == defaults.get("text_encoder_path") and defaults.get("tokenizer_path"):
            paths["tokenizer_path"] = defaults["tokenizer_path"]
    for field, path in paths.items():
        setattr(model, field, path)
    return model


def resolve_model(c, config, *, model_id=None, model=None):
    """Return the frozen model and sampling recipe, or the same error shown in options."""
    from ypuddin.models import get_family

    from .routes_core import _model_row
    from .xyz import adopt_backbone_objective

    if model_id is not None and model is not None:
        raise ApiError("Choose a registered model or provide model paths, not both", status=422,
                       code="regularization.models")
    selected = model_id is not None or model is not None
    explicit = model
    row = None
    try:
        if model_id is not None:
            row = c.db.fetchone("SELECT * FROM models WHERE id=? AND kind='dit'", (model_id,))
            if row is None:
                raise ApiError("The selected model is no longer registered", status=422,
                               code="regularization.models")
            projected = _model_row(row, c)
            if not projected["exists"] or projected.get("unsupported_reason"):
                raise ApiError(projected.get("unsupported_reason") or "Model file does not exist",
                               status=422, code="regularization.models")
            model = _registered_config(c, row)
        else:
            model = model.model_copy(deep=True) if model is not None else ModelConfig.model_validate(config.get("model", {}))
        model.attention = "auto"
        family = get_family(model.family)
        for field in MODEL_PATHS.values():
            if path := getattr(model, field):
                _allowed(c, path)
        if selected and model.dit_path:
            if row is None:
                target = Path(model.dit_path).expanduser().resolve()
                row = next((asset for asset in c.db.fetchall("SELECT * FROM models WHERE family=? AND kind='dit'", (model.family,))
                            if Path(asset["path"]).expanduser().resolve() == target), None)
            if row is not None:
                projected = _model_row(row, c)
                if projected.get("unsupported_reason"):
                    raise ApiError(projected["unsupported_reason"], status=422, code="regularization.models")
                if model.family == "krea2" and projected.get("variant") in {"raw", "turbo"}:
                    if (explicit is not None and "krea2_variant" in explicit.model_fields_set
                            and model.krea2_variant not in {"auto", projected["variant"]}):
                        raise ApiError(f"此 Krea 2 模型已登记为 {projected['variant'].title()}，请选择对应类型。",
                                       status=422, code="regularization.variant")
                    model.krea2_variant = projected["variant"]
            if explicit is None or not {"prediction_type", "zero_terminal_snr"} & explicit.model_fields_set:
                adopt_backbone_objective(model, Path(model.dit_path).expanduser().resolve())
        issues = [family.spec.retired_reason] if family.spec.retired_reason else []
        if model.family == "moss_image":
            if row is None and model.dit_path:
                target = Path(model.dit_path).expanduser().resolve()
                row = next((asset for asset in c.db.fetchall("SELECT * FROM models WHERE family='moss_image' AND kind='dit'")
                            if Path(asset["path"]).expanduser().resolve() == target), None)
            projected = _model_row(row, c) if row is not None else {}
            authenticated = projected.get("exists") and not projected.get("unsupported_reason")
            # A current registry receipt authenticates the main weight; components still undergo validation.
            issues.extend(family.validate_config(model, authenticate=not authenticated))
        else:
            issues.extend(family.validate_config(model))
        if issues:
            raise ApiError("; ".join(issues), status=422, code="regularization.models")
        sampling = (SamplingConfig(sampler=family.spec.sampling.sampler).model_dump(mode="json")
                    if selected else config.get("sampling", {}))
        return model, family, sampling
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        raise ApiError(str(error), status=422, code="regularization.models") from error


def model_options(c, pid, vid):
    from ypuddin.models import get_family

    from .routes_core import _model_row
    from .routes_work import get_project_config

    c.require_project_type(pid, "image")
    c.resolve_version(pid, vid)
    config = get_project_config(pid, c, vid)

    def option(row=None):
        family = row["family"] if row else config.get("model", {}).get("family", "anima")
        item = {"model_id": row["id"] if row else None, "family": family,
                "name": (row.get("display_name") or Path(row["path"]).name) if row else get_family(family).spec.label,
                "variant": row.get("variant") if row else None, "purpose": row.get("purpose") if row else None,
                "available": True, "reason": None}
        try:
            resolved, _, _ = resolve_model(c, config, model_id=item["model_id"])
            if resolved.family == "krea2":
                item["variant"] = resolved.krea2_variant
        except ApiError as error:
            item.update(available=False, reason=str(error))
        return item

    return {"current": option(), "models": [option(_model_row(row, c)) for row in c.db.fetchall(
        "SELECT * FROM models WHERE kind='dit' ORDER BY family,created_at,id")]}
