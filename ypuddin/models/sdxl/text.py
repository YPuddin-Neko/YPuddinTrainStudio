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
        max_token_length: int = 75,
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
        self.context_length = int(configs[0]["max_position_embeddings"])
        if self.context_length != int(configs[1]["max_position_embeddings"]):
            raise ValueError("SDXL CLIP encoders must use the same sequence length")
        if max_token_length not in (75, 150, 225):
            raise ValueError("SDXL max_token_length must be 75, 150 or 225")
        if max_token_length != 75 and self.context_length != 77:
            raise ValueError("SDXL long captions require CLIP encoders with a 77-token context")
        self.chunks = max_token_length // 75
        self.max_len = self.context_length if self.chunks == 1 else max_token_length + 2
        self.hidden_size = sum(int(config["hidden_size"]) for config in configs)
        self.pooled_size = int(configs[1]["projection_dim"])
        self.fingerprint = content_fingerprint(
            [
                *self.paths,
                *tokenizers,
                *(config_asset(p, c) for p, c in zip(self.paths, self.components, strict=True)),
            ],
            # Preserve the existing 75-token cache identity and numerical path.
            namespace=(
                f"sdxl-dual-clip-penultimate-pooled-v1:dtype={dtype}"
                if self.chunks == 1
                else f"sdxl-dual-clip-chunks-first-pooled-v1:tokens={max_token_length}:dtype={dtype}"
            ),
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
            ids = self._tokenize(tokenizer, captions).to(self.device)
            # SDXL's CLIP padding is meaningful conditioning. Match upstream: no attention
            # mask, no trimming, and no final layer norm applied to penultimate hidden states.
            output = model(ids, output_hidden_states=True)
            states = output.hidden_states[-2]
            if self.chunks > 1:
                states = states.reshape(len(captions), self.chunks, self.context_length, -1)
                # Keep the first BOS, each chunk's 75 content/padding positions,
                # and the final EOS/PAD. Do not modify hidden states in place:
                # online text-encoder training needs their complete gradient graph.
                states = torch.cat(
                    [states[:, 0, :1], states[:, :, 1:-1].flatten(1, 2), states[:, -1, -1:]], dim=1
                )
            hidden.append(states)
            if hasattr(output, "text_embeds"):
                pooled = output.text_embeds
                if self.chunks > 1:
                    # SDXL's pooled condition remains the first CLIP-G chunk.
                    pooled = pooled.reshape(len(captions), self.chunks, -1)[:, 0]
        if pooled is None:
            raise RuntimeError("SDXL CLIP-G did not return projected pooled embeddings")
        return TextCond({"embeds": torch.cat(hidden, dim=-1), "pooled": pooled}).to(device)

    def _tokenize(self, tokenizer, captions: list[str]) -> Tensor:
        tokens = tokenizer(
            captions,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        ids = tokens.input_ids
        if self.chunks == 1:
            return ids
        chunks = []
        for start in range(1, self.max_len - 1, self.context_length - 2):
            chunk = torch.cat([ids[:, :1], ids[:, start : start + 75], ids[:, -1:]], dim=1)
            if tokenizer.pad_token_id != tokenizer.eos_token_id:
                # CLIP-G's zero padding ID is also a valid content token ("!").
                # Use tokenizer positions only to distinguish real content from
                # padding; the CLIP forward still deliberately receives no mask.
                valid = tokens.attention_mask[:, start : start + 75].bool()
                full = valid[:, -1] & (chunk[:, -2] != tokenizer.eos_token_id)
                chunk[full, -1] = tokenizer.eos_token_id
                chunk[~valid[:, 0], 1] = tokenizer.eos_token_id
            chunks.append(chunk)
        return torch.stack(chunks, dim=1).flatten(0, 1)

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
