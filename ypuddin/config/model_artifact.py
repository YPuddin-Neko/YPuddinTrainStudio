"""Resolve generated component paths when a full model artifact is relocated."""

from pathlib import Path


def rebind_artifact_components(config: dict, manifest: dict, root: Path) -> dict:
    root = root.resolve()
    model = config.get("model", {})
    family = manifest.get("family")
    if not isinstance(model, dict) or model.get("family") != family:
        raise ValueError("full model artifact configuration and manifest families differ")
    components = manifest.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("full model artifact components must be an object")
    fields = {
        "backbone": "dit_path",
        "text_encoder": "text_encoder_path",
        "text_encoder_2": "text_encoder_2_path",
    }
    for component, relative in components.items():
        if component not in fields or not isinstance(relative, str):
            raise ValueError("invalid full model artifact component")
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError("full model artifact component is outside the artifact or missing")
        directory = family in {"sdxl", "flux2"} or (family == "toy" and component != "backbone")
        model[fields[component]] = str(source.parent if directory else source)
    return config
