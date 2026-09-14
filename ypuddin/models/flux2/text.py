"""Lazy Klein Qwen3 conditioning using the upstream chat template."""

import math
from pathlib import Path

import torch

from ypuddin.models.base import TextCond, TextPipeline
from ypuddin.models.fingerprints import content_fingerprint

from .loading import DEV_UNSUPPORTED, KLEIN_VARIANTS, read_json, shapes


class Flux2Text(TextPipeline):
    max_len = 512

    def __init__(self, path: Path, tokenizer_path: Path, *, variant: str, dtype, device="cpu"):
        if variant == "dev":
            raise ValueError(DEV_UNSUPPORTED)
        if variant not in KLEIN_VARIANTS:
            raise ValueError("FLUX.2 text encoding requires Klein base 4B or 9B")
        self.path, self.tokenizer_path, self.variant = path, tokenizer_path, variant
        self.dtype, self.device = dtype, torch.device(device)
        self.model, self.tokenizer = None, None
        if not path.is_dir():
            raise ValueError("FLUX.2 text encoder must be a complete local HF directory")
        config = read_json(path / "config.json")
        if config.get("model_type") != "qwen3":
            raise ValueError(f"FLUX.2 {variant} requires a Qwen3 text encoder")
        self.layers = (9, 18, 27)
        if int(config.get("num_hidden_layers", 0)) < max(self.layers):
            raise ValueError("FLUX.2 text encoder lacks required intermediate hidden layers")
        self.hidden_size = int(config["hidden_size"]) * 3
        if config.get("quantization_config"):
            raise ValueError("FLUX.2 quantized text encoders are unsupported; select BF16/FP16 local weights")
        self.weight_elements = sum(math.prod(shape) for shape in shapes(path).values())
        if not tokenizer_path.is_dir() or not (tokenizer_path / "tokenizer_config.json").is_file():
            raise ValueError("FLUX.2 needs a local tokenizer/processor directory with its chat template")
        token_cfg = read_json(tokenizer_path / "tokenizer_config.json")
        if not token_cfg.get("chat_template") and not any(tokenizer_path.glob("*chat_template*")):
            raise ValueError("FLUX.2 tokenizer is missing the required chat template")
        self.fingerprint = content_fingerprint(
            [path, tokenizer_path], namespace=f"flux2-{variant}-layers{self.layers}-len512-v1:{dtype}"
        )

    def _ensure(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path), local_files_only=True)
            if self.tokenizer.pad_token_id is None:
                raise ValueError("FLUX.2 Qwen3 tokenizer has no padding token")
        if self.model is None:
            from transformers import Qwen3ForCausalLM

            self.model, info = Qwen3ForCausalLM.from_pretrained(
                str(self.path),
                local_files_only=True,
                use_safetensors=True,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                output_loading_info=True,
            )
            if any(
                info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                self.model = None
                raise ValueError(f"FLUX.2 text encoder weights do not match config: {info}")
            self.model.to(self.device).requires_grad_(False).eval()

    def to(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)

    def unload(self):
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def trainable_modules(self):
        self._ensure()
        return {"text_encoder": self.model}

    def encode(self, captions, device):
        with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self, "training_enabled", False)):
            return self._encode(captions, device)

    def _encode(self, captions, device):
        self._ensure()
        from diffusers import Flux2KleinPipeline

        embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            self.model,
            self.tokenizer,
            captions,
            device=self.device,
            dtype=self.dtype,
            max_sequence_length=self.max_len,
            hidden_states_layers=self.layers,
        )
        return TextCond({"embeds": embeds}).to(device)

    @torch.no_grad()
    def encode_for_cache(self, captions):
        embeds = self.encode(captions, "cpu")["embeds"]
        return [{"embeds": row.contiguous()} for row in embeds]

    def cond_from_cache(self, entries, device):
        if not entries or any(e["embeds"].shape != (self.max_len, self.hidden_size) for e in entries):
            raise ValueError("FLUX.2 text cache has incompatible variant/sequence/hidden dimensions")
        return TextCond({"embeds": torch.stack([e["embeds"] for e in entries])}).to(device)
