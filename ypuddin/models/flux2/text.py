"""Lazy dev Mistral3 / Klein Qwen3 conditioning using upstream chat templates."""

from pathlib import Path

import torch

from ypuddin.models.base import TextCond, TextPipeline
from ypuddin.models.fingerprints import content_fingerprint

from .loading import read_json, shapes


class Flux2Text(TextPipeline):
    max_len = 512

    def __init__(self, path: Path, tokenizer_path: Path, *, variant: str, dtype, device="cpu"):
        self.path, self.tokenizer_path, self.variant = path, tokenizer_path, variant
        self.dtype, self.device = dtype, torch.device(device)
        self.model, self.tokenizer = None, None
        if not path.is_dir():
            raise ValueError("FLUX.2 text encoder must be a complete local HF directory")
        config = read_json(path / "config.json")
        expected = "mistral3" if variant == "dev" else "qwen3"
        if config.get("model_type") != expected:
            raise ValueError(f"FLUX.2 {variant} requires {expected} text encoder")
        text_config = config.get("text_config", config)
        self.layers = (10, 20, 30) if variant == "dev" else (9, 18, 27)
        if int(text_config.get("num_hidden_layers", 0)) < max(self.layers):
            raise ValueError("FLUX.2 text encoder lacks required intermediate hidden layers")
        self.hidden_size = int(text_config["hidden_size"]) * 3
        if config.get("quantization_config"):
            raise ValueError("FLUX.2 quantized text encoders are unsupported; select BF16/FP16 local weights")
        shapes(path)
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
            from transformers import AutoProcessor, AutoTokenizer

            cls = AutoProcessor if self.variant == "dev" else AutoTokenizer
            self.tokenizer = cls.from_pretrained(str(self.tokenizer_path), local_files_only=True)
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            if self.variant == "dev":
                # Mistral's pad=EOS fallback changes conditioning; official FLUX.2 uses id 11.
                tokenizer.padding_side = "right"
                tokenizer.pad_token_id = 11
            elif tokenizer.pad_token_id is None:
                raise ValueError("FLUX.2 Qwen3 tokenizer has no padding token")
        if self.model is None:
            from transformers import Mistral3ForConditionalGeneration, Qwen3ForCausalLM

            cls = Mistral3ForConditionalGeneration if self.variant == "dev" else Qwen3ForCausalLM
            self.model, info = cls.from_pretrained(
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

    @torch.no_grad()
    def encode(self, captions, device):
        self._ensure()
        if self.variant == "dev":
            from diffusers import Flux2Pipeline

            embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                self.model,
                self.tokenizer,
                captions,
                device=self.device,
                dtype=self.dtype,
                max_sequence_length=self.max_len,
                hidden_states_layers=self.layers,
            )
        else:
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

    def encode_for_cache(self, captions):
        embeds = self.encode(captions, "cpu")["embeds"]
        return [{"embeds": row.contiguous()} for row in embeds]

    def cond_from_cache(self, entries, device):
        if not entries or any(e["embeds"].shape != (self.max_len, self.hidden_size) for e in entries):
            raise ValueError("FLUX.2 text cache has incompatible variant/sequence/hidden dimensions")
        return TextCond({"embeds": torch.stack([e["embeds"] for e in entries])}).to(device)
