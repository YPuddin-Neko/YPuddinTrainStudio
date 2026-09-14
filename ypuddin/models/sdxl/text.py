"""SDXL conditioning: penultimate CLIP-L/G hidden states and projected CLIP-G pooled output."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from ypuddin.models.base import TextCond, TextPipeline
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.memory import release_model_memory

from .loading import ASSETS, component_config, config_asset, load_clip


def tokenizer_paths(checkpoint: Path, override: str | None = None) -> tuple[Path, Path]:
    root = (
        Path(override).expanduser() if override else checkpoint if checkpoint.is_dir() else checkpoint.parent
    )
    if override and not (root / "tokenizer").is_dir():
        raise ValueError("SDXL tokenizer_path must contain both tokenizer/ and tokenizer_2/ directories")
    if (root / "tokenizer").is_dir() or (root / "tokenizer_2").is_dir():
        return root / "tokenizer", root / "tokenizer_2"
    return ASSETS / "tokenizer", ASSETS / "tokenizer_2"


class SDXLText(TextPipeline):
    def __init__(
        self,
        clip_l: Path,
        clip_g: Path,
        tokenizers: tuple[Path, Path],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.paths = (clip_l, clip_g)
        self.tokenizer_paths = tokenizers
        self.device, self.dtype = torch.device(device), dtype
        self.models: list[nn.Module] = []
        self.tokenizers: list = []
        configs = [
            component_config(path, component)
            for path, component in zip(self.paths, self.components, strict=True)
        ]
        self.max_len = int(configs[0]["max_position_embeddings"])
        if self.max_len != int(configs[1]["max_position_embeddings"]):
            raise ValueError("SDXL CLIP encoders must use the same sequence length")
        self.hidden_size = sum(int(config["hidden_size"]) for config in configs)
        self.pooled_size = int(configs[1]["projection_dim"])
        self.fingerprint = content_fingerprint(
            [
                *self.paths,
                *tokenizers,
                *(config_asset(p, c) for p, c in zip(self.paths, self.components, strict=True)),
            ],
            namespace=f"sdxl-dual-clip-penultimate-pooled-v1:dtype={dtype}",
        )

    components = ("text_encoder", "text_encoder_2")

    def _ensure(self) -> None:
        if not self.tokenizers:
            from transformers import CLIPTokenizerFast

            self.tokenizers = [
                CLIPTokenizerFast.from_pretrained(str(p), local_files_only=True) for p in self.tokenizer_paths
            ]
        if not self.models:
            models = []
            for path, component in zip(self.paths, self.components, strict=True):
                models.append(load_clip(path, component, device=self.device, dtype=self.dtype))
            self.models = models

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        for model in self.models:
            model.to(self.device)

    def unload(self) -> None:
        if not self.models:
            return
        self.models.clear()
        release_model_memory(self.device)

    def trainable_modules(self) -> dict[str, nn.Module]:
        self._ensure()
        return dict(zip(self.components, self.models, strict=True))

    def encode(self, captions: list[str], device: torch.device | str) -> TextCond:
        with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self, "training_enabled", False)):
            return self._encode(captions, device)

    def _encode(self, captions: list[str], device: torch.device | str) -> TextCond:
        self._ensure()
        hidden = []
        pooled = None
        for tokenizer, model in zip(self.tokenizers, self.models, strict=True):
            ids = tokenizer(
                captions, padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            # SDXL's CLIP padding is meaningful conditioning. Match upstream: no attention
            # mask, no trimming, and no final layer norm applied to penultimate hidden states.
            output = model(ids, output_hidden_states=True)
            hidden.append(output.hidden_states[-2])
            if hasattr(output, "text_embeds"):
                pooled = output.text_embeds
        if pooled is None:
            raise RuntimeError("SDXL CLIP-G did not return projected pooled embeddings")
        return TextCond({"embeds": torch.cat(hidden, dim=-1), "pooled": pooled}).to(device)

    @torch.no_grad()
    def encode_for_cache(self, captions: list[str]) -> list[dict[str, Tensor]]:
        cond = self.encode(captions, "cpu")
        return [
            {key: tensor[i].contiguous() for key, tensor in cond.tensors.items()}
            for i in range(len(captions))
        ]

    def cond_from_cache(self, entries: list[dict[str, Tensor]], device: torch.device | str) -> TextCond:
        if not entries:
            raise ValueError("SDXL text cache batch is empty")
        for entry in entries:
            if entry["embeds"].shape != (self.max_len, self.hidden_size) or entry["pooled"].shape != (
                self.pooled_size,
            ):
                raise ValueError("SDXL text cache has incompatible CLIP conditioning dimensions")
        return TextCond(
            {key: torch.stack([entry[key] for entry in entries]) for key in ("embeds", "pooled")}
        ).to(device)
