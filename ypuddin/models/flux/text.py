"""FLUX.1 frozen CLIP-L pooler and T5-XXL sequence embeddings, loaded only during encoding."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from ypuddin.models.base import TextCond, TextPipeline
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.memory import release_model_memory

from .loading import ASSETS, component_config, config_file, load_component


def tokenizer_paths(root: Path, override: str | None, paths: tuple[Path, Path]) -> tuple[Path, Path]:
    location = Path(override).expanduser() if override else root if root.is_dir() else root.parent
    if override or (location / "tokenizer").is_dir() or (location / "tokenizer_2").is_dir():
        result = location / "tokenizer", location / "tokenizer_2"
    else:
        defaults = ASSETS / "sdxl/assets/tokenizer", ASSETS / "anima/assets/t5_old"
        result = tuple(
            p if p.is_dir() and (p / "tokenizer.json").is_file() else fallback
            for p, fallback in zip(paths, defaults, strict=True)
        )
    for path in result:
        if not (path / "tokenizer.json").is_file() and not all(
            (path / f).is_file() for f in ("vocab.json", "merges.txt")
        ):
            raise ValueError(f"FLUX.1 tokenizer assets are missing: {path}")
    return result


class FluxText(TextPipeline):
    components = ("text_encoder", "text_encoder_2")

    def __init__(
        self, paths: tuple[Path, Path], tokenizers: tuple[Path, Path], *, max_len: int, device, dtype
    ):
        self.paths, self.tokenizer_paths = paths, tokenizers
        self.device, self.dtype = torch.device(device), dtype
        self.max_len = max_len
        self.configs = [component_config(p, c) for p, c in zip(paths, self.components, strict=True)]
        self.pooled_size = int(self.configs[0]["hidden_size"])
        self.hidden_size = int(self.configs[1]["d_model"])
        self.clip_len = int(self.configs[0]["max_position_embeddings"])
        self.models, self.tokenizers = [], []
        files = [file for p, c in zip(paths, self.components, strict=True) if (file := config_file(p, c))]
        self.fingerprint = content_fingerprint(
            [*paths, *tokenizers, *files],
            namespace=f"flux1-clip-pooler-t5-padding-v1:{max_len}:{dtype}:"
            + json.dumps(self.configs, sort_keys=True),
        )

    def _ensure(self):
        if not self.tokenizers:
            from transformers import CLIPTokenizerFast, T5TokenizerFast

            self.tokenizers = [
                cls.from_pretrained(str(path), local_files_only=True)
                for cls, path in zip((CLIPTokenizerFast, T5TokenizerFast), self.tokenizer_paths, strict=True)
            ]
            if any(len(t) > c["vocab_size"] for t, c in zip(self.tokenizers, self.configs, strict=True)):
                raise ValueError("FLUX.1 tokenizer vocabulary exceeds its encoder embeddings")
        if not self.models:
            self.models = [
                load_component(path, component, device=self.device, dtype=self.dtype, config=config)
                for path, component, config in zip(self.paths, self.components, self.configs, strict=True)
            ]

    @torch.no_grad()
    def encode(self, captions: list[str], device) -> TextCond:
        self._ensure()
        outputs = []
        for tokenizer, model, length in zip(
            self.tokenizers, self.models, (self.clip_len, self.max_len), strict=True
        ):
            ids = tokenizer(
                captions, padding="max_length", max_length=length, truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            # Match FluxPipeline: padded T5 rows are meaningful; neither encoder receives a mask.
            outputs.append(model(ids, output_hidden_states=False))
        return TextCond({"embeds": outputs[1].last_hidden_state, "pooled": outputs[0].pooler_output}).to(
            device
        )

    def encode_for_cache(self, captions: list[str]) -> list[dict[str, Tensor]]:
        cond = self.encode(captions, "cpu")
        return [
            {key: value[i].contiguous() for key, value in cond.tensors.items()} for i in range(len(captions))
        ]

    def cond_from_cache(self, entries: list[dict[str, Tensor]], device) -> TextCond:
        if not entries or any(
            e["embeds"].shape != (self.max_len, self.hidden_size) or e["pooled"].shape != (self.pooled_size,)
            for e in entries
        ):
            raise ValueError("FLUX.1 text cache has incompatible conditioning dimensions")
        return TextCond({key: torch.stack([e[key] for e in entries]) for key in ("embeds", "pooled")}).to(
            device
        )

    def to(self, device):
        self.device = torch.device(device)
        for model in self.models:
            model.to(self.device)

    def unload(self):
        if not self.models:
            return
        self.models.clear()
        release_model_memory(self.device)
