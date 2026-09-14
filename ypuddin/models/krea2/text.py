"""Krea 2 text pipeline: Qwen3-VL-4B-Instruct hidden states from 12 selected layers.

Conventions (identical to the official pipeline / musubi-tuner ``krea2_encoder.py``):
* prompt = fixed system prefix (34 tokens) + caption, right-padded with the tokenizer's pad token to a
  fixed length (``max_len + 34 - 5``), then a 5-token ``<|im_end|>\\n<|im_start|>assistant\\n`` suffix;
* hidden states of layers ``(2, 5, 8, ..., 35)`` are stacked -> ``(seq, 12, 2560)``; the 34 prefix
  positions are dropped;
* padding positions are dropped too ("gather valid"): the DiT gives text tokens a zero RoPE position and
  masks padding, so only the set of valid vectors matters -- caching becomes ~50x smaller.
Cache entries hold the trimmed stack in bf16; a batch is re-padded to its longest caption.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ypuddin.models.anima.text import ASSETS as ANIMA_ASSETS
from ypuddin.models.anima.text import tokenizer_assets
from ypuddin.models.base import TextCond, TextPipeline
from ypuddin.models.memory import release_model_memory

log = logging.getLogger(__name__)

PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
QWEN3_VL_4B_TEXT_CONFIG: dict[str, Any] = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
    "rope_theta": 5000000,
    "tie_word_embeddings": True,
    "use_cache": False,
    "vocab_size": 151936,
}


def _strip_comfy_prefixes(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """ComfyUI single-file Qwen3-VL checkpoints use bare ``model.`` / ``visual.`` keys; HF uses
    ``model.language_model.`` / ``model.visual.``. Only the language model is kept."""
    out: dict[str, Tensor] = {}
    for k, v in sd.items():
        if k.startswith("model.language_model."):
            out[k[len("model.language_model.") :]] = v
        elif k.startswith("language_model."):
            out[k[len("language_model.") :]] = v
        elif k.startswith("model.visual.") or k.startswith("visual.") or k == "lm_head.weight":
            continue
        elif k.startswith("model."):
            out[k[len("model.") :]] = v
        else:
            out[k] = v
    return out


def _dequant_fp8_scaled(sd: dict[str, Tensor], dtype: torch.dtype) -> dict[str, Tensor]:
    """Comfy-Org ``fp8_scaled`` text encoders: ``W(fp8) * scale_weight`` -> compute dtype."""
    out: dict[str, Tensor] = {}
    for k, v in sd.items():
        if k.endswith((".scale_weight", ".weight_scale")) or k == "scaled_fp8":
            continue
        if v.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            scale = sd.get(base + ".scale_weight", sd.get(base + ".weight_scale"))
            v = v.to(torch.float32) * (scale.to(torch.float32) if scale is not None else 1.0)
        out[k] = v.to(dtype) if v.is_floating_point() else v
    return out


def text_config_for(path: str | Path):
    """Decoder config of the checkpoint at ``path``: an HF directory, or a single file with the model's ``config.json``
    next to it; a lone single file is assumed to be Qwen3-VL-4B-Instruct (the only encoder Krea 2 ships with)."""
    from transformers import Qwen3VLConfig, Qwen3VLTextConfig

    p = Path(path)
    cfg_dir = p if p.is_dir() else p.parent
    if (cfg_dir / "config.json").exists():
        import json

        config = json.loads((cfg_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") == "qwen3_vl_text":
            return Qwen3VLTextConfig(**config)
        cfg = Qwen3VLConfig.from_pretrained(str(cfg_dir), local_files_only=True)
        return getattr(cfg, "text_config", cfg)
    return Qwen3VLTextConfig(**QWEN3_VL_4B_TEXT_CONFIG)


def select_layers_for(num_hidden_layers: int, txtlayers: int) -> tuple[int, ...]:
    """Which hidden states feed the DiT's text fusion: every ``step``-th layer ending at the last one.

    Reproduces the official ``(2, 5, ..., 35)`` for 36 decoder layers / 12 fusion inputs and degrades to the
    same rule for reduced test geometries (``hidden_states[0]`` is the embedding output, so indices are 1-based).
    """
    if num_hidden_layers == 36 and txtlayers == 12:
        return SELECT_LAYERS
    if txtlayers > num_hidden_layers:
        raise ValueError(
            f"the DiT expects {txtlayers} text layers but the encoder only has {num_hidden_layers}"
        )
    step = num_hidden_layers // txtlayers
    return tuple(range(num_hidden_layers - step * (txtlayers - 1), num_hidden_layers + 1, step))


def _load_language_model(path: str | Path, dtype: torch.dtype, device: torch.device | str) -> nn.Module:
    """The Qwen3-VL text decoder only (the vision tower is never used for conditioning)."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLTextModel

    p = Path(path)
    if p.is_dir():
        full = Qwen3VLForConditionalGeneration.from_pretrained(str(p), dtype=dtype)
        model = full.model.language_model
        del full
    else:
        from safetensors.torch import load_file

        text_cfg = text_config_for(p)
        # A lone file uses the official 4B geometry. Build only its shapes until the
        # checkpoint has passed validation; allocating/randomizing that model first
        # wastes many GiB even when a tiny or incompatible file will be rejected.
        with torch.device("meta"):
            model = Qwen3VLTextModel(text_cfg)
        sd = _dequant_fp8_scaled(_strip_comfy_prefixes(load_file(str(p))), dtype)
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        except RuntimeError as e:  # shape mismatch: not the 4B geometry
            raise RuntimeError(
                f"{p.name} does not match the Qwen3-VL-4B-Instruct text decoder geometry; place the model's "
                f"config.json next to the file (or point model.text_encoder_path at the HF directory): {e}"
            ) from e
        missing = [m for m in missing if "rotary" not in m]
        if missing or unexpected:
            raise RuntimeError(
                f"Qwen3-VL text encoder checkpoint does not match: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        # RoPE frequencies are derived non-persistent buffers, absent from the
        # checkpoint. Recreate this small module on CPU, using the installed
        # transformers implementation (including its original/dynamic frequencies).
        with torch.device("cpu"):
            model.rotary_emb = type(model.rotary_emb)(config=text_cfg)
        remaining = [
            name for name, tensor in (*model.named_parameters(), *model.named_buffers()) if tensor.is_meta
        ]
        if remaining:
            raise RuntimeError(f"Qwen3-VL text encoder has uninitialized tensors: {remaining[:5]}")
        model = model.to(dtype)
    model.config.use_cache = False
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


class Krea2Text(TextPipeline):
    fingerprint = "krea2-qwen3vl-4b-12layers-v1"

    def __init__(
        self,
        text_encoder_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        max_len: int = 512,
        select_layers: tuple[int, ...] | None = None,
        txtlayers: int = len(SELECT_LAYERS),
    ):
        from transformers import AutoTokenizer

        p = Path(text_encoder_path)
        tok_dir = (
            Path(tokenizer_path)
            if tokenizer_path
            else (
                p
                if p.is_dir() and (p / "tokenizer.json").exists()
                else p.parent
                if (p.parent / "tokenizer.json").exists()
                else ANIMA_ASSETS / "qwen3_06b"
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        self.tokenizer.padding_side = "right"  # the 34-token prefix must stay at the front
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|endoftext|>"
        self.max_len = max_len
        text_cfg = text_config_for(text_encoder_path)
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_hidden_layers = int(text_cfg.num_hidden_layers)
        self.select_layers = (
            tuple(select_layers) if select_layers else select_layers_for(self.num_hidden_layers, txtlayers)
        )
        if max(self.select_layers) > self.num_hidden_layers:
            raise ValueError(
                f"select_layers {self.select_layers} exceed the encoder depth {self.num_hidden_layers}"
            )
        if self.select_layers != SELECT_LAYERS:  # a different layer set must never share the official cache
            self.fingerprint = f"{Krea2Text.fingerprint}-L{'.'.join(map(str, self.select_layers))}"
        self.dtype = dtype
        self.device = torch.device(device)
        self._path = text_encoder_path
        self.encoder: nn.Module | None = None
        self.prefix_len = len(self.tokenizer(PREFIX)["input_ids"])
        self.suffix_ids = torch.tensor(self.tokenizer(SUFFIX)["input_ids"], dtype=torch.long)
        from ypuddin.models.fingerprints import content_fingerprint

        assets = [p, *tokenizer_assets(tok_dir)]
        if p.is_file() and (p.parent / "config.json").exists():
            assets.append(p.parent / "config.json")
        self.fingerprint = content_fingerprint(
            assets,
            namespace=f"{Krea2Text.fingerprint}:{self.select_layers}:{max_len}:{dtype}:{PREFIX}:{SUFFIX}:{text_cfg.to_json_string()}",
        )

    # ----------------------------------------------------------------- weights
    def _ensure_loaded(self) -> nn.Module:
        if self.encoder is None:
            self.encoder = _load_language_model(self._path, self.dtype, self.device)
        return self.encoder

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.encoder is not None:
            self.encoder.to(self.device)

    def unload(self) -> None:
        if self.encoder is None:
            return
        self.encoder = None
        release_model_memory(self.device)

    # ----------------------------------------------------------------- tokenize / encode
    def _tokenize(self, captions: list[str]) -> tuple[Tensor, Tensor]:
        body_len = self.max_len + self.prefix_len - len(self.suffix_ids)
        enc = self.tokenizer(
            [PREFIX + c for c in captions],
            truncation=True,
            padding="max_length",
            max_length=body_len,
            return_tensors="pt",
        )
        suffix = self.suffix_ids.unsqueeze(0).expand(len(captions), -1)
        ids = torch.cat([enc["input_ids"], suffix], dim=1)
        mask = torch.cat([enc["attention_mask"].bool(), torch.ones_like(suffix, dtype=torch.bool)], dim=1)
        return ids, mask

    @torch.no_grad()
    def encode_for_cache(self, captions: list[str]) -> list[dict[str, Tensor]]:
        enc = self._ensure_loaded()
        ids, mask = self._tokenize(captions)
        out = enc(
            input_ids=ids.to(self.device), attention_mask=mask.to(self.device), output_hidden_states=True
        )
        stack = torch.stack([out.hidden_states[i] for i in self.select_layers], dim=2)  # (B, T, L, D)
        stack = stack[:, self.prefix_len :]
        mask = mask[:, self.prefix_len :]
        entries = []
        for i in range(len(captions)):
            valid = stack[i][mask[i].to(stack.device)]  # (n_i, L, D): prompt tokens + suffix, padding dropped
            entries.append({"embeds": valid.to(torch.bfloat16).cpu().contiguous()})
        return entries

    def cond_from_cache(self, entries: list[dict[str, Tensor]], device: torch.device | str) -> TextCond:
        n_max = max(e["embeds"].shape[0] for e in entries)
        layers, dim = entries[0]["embeds"].shape[1:]
        storage_device = entries[0]["embeds"].device
        embeds = torch.zeros(len(entries), n_max, layers, dim, dtype=torch.bfloat16, device=storage_device)
        mask = torch.zeros(len(entries), n_max, dtype=torch.bool, device=storage_device)
        for i, e in enumerate(entries):
            n = e["embeds"].shape[0]
            embeds[i, :n] = e["embeds"].to(torch.bfloat16)
            mask[i, :n] = True
        return TextCond({"embeds": embeds, "attn_mask": mask}).to(device)

    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"text_encoder": self._ensure_loaded()}

    def encode(self, captions: list[str], device: torch.device | str) -> TextCond:
        if not getattr(self, "training_enabled", False):
            return self.cond_from_cache(self.encode_for_cache(captions), device)
        ids, mask = self._tokenize(captions)
        output = self._ensure_loaded()(
            input_ids=ids.to(self.device),
            attention_mask=mask.to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        stack = torch.stack([output.hidden_states[i] for i in self.select_layers], dim=2)[
            :, self.prefix_len :
        ]
        mask = mask[:, self.prefix_len :].to(stack.device)
        # Preserve the model's BF16 conditioning convention without detaching from encoder parameters.
        entries = [{"embeds": stack[i][mask[i]].to(torch.bfloat16)} for i in range(len(captions))]
        return self.cond_from_cache(entries, device)
