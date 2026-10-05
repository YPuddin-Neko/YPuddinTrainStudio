"""Explicit user-supplied ModelSpec fields shared by weight exporters."""

from .schema import CheckpointConfig

USER_METADATA_FIELDS = (
    "title", "author", "description", "license", "merged_from", "tags", "usage_hint", "trigger_phrase",
)


def user_modelspec_metadata(checkpoint: CheckpointConfig) -> dict[str, str]:
    if not checkpoint.save_training_metadata:
        return {}
    return {
        f"modelspec.{name}": value
        for name in USER_METADATA_FIELDS
        if (value := getattr(checkpoint, f"metadata_{name}")).strip()
    }
