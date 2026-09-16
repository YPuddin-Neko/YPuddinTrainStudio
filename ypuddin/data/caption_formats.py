"""Caption-sidecar capabilities shared by training preflight and inspection."""

from pathlib import Path


class UnsupportedCaptionFormat(ValueError):
    code = "caption_model_unsupported"


def family_caption_formats(family: str) -> tuple[str, ...]:
    from ypuddin.models import get_family

    return get_family(family).spec.caption_formats


def effective_caption_extension(extension: str, formats: tuple[str, ...]) -> str:
    """Auto follows this family's formats; explicit selections never silently change."""
    if extension.lower() == "auto" and "json" not in formats:
        return ".txt"
    return extension


def require_caption_format(path: str | Path, formats: tuple[str, ...], family: str) -> None:
    # Existing custom suffixes are plain-text captions, not arbitrary JSON schemas.
    kind = "json" if Path(path).suffix.lower() == ".json" else "txt"
    if kind not in formats:
        raise UnsupportedCaptionFormat(
            f"Caption {path}: model family {family!r} does not support {kind.upper()} captions; "
            "select a supported caption format explicitly"
        )
