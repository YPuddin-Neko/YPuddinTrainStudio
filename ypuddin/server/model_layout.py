"""Shared category-first layout for model downloads and file browsing."""
from pathlib import Path


def model_category(kind: str) -> str:
    if kind == 'vae':
        return 'vae'
    if kind in {'text_encoder', 'text_encoder_2', 'tokenizer'}:
        return 'text_encoders'
    if kind == 'dit':
        return 'diffusion_models'
    raise ValueError(f'Unknown model component: {kind}')


def model_folder(root: Path, family: str, kind: str) -> Path:
    # Both families use the Qwen Image VAE architecture.
    owner = 'shared' if kind == 'vae' and family in {'anima', 'krea2'} else family
    if owner not in {'anima', 'krea2', 'sdxl', 'flux2', 'shared'}:
        raise ValueError(f'Unknown model family: {family}')
    return root / model_category(kind) / owner
