"""Anima text pipeline: Qwen3-0.6B hidden states + old-T5 token ids for the LLM adapter.

Conventions (must match the official ComfyUI / sd-scripts behaviour):
* Qwen3 embeddings are the final ``last_hidden_state`` (post final RMSNorm) with padded positions
  zeroed; no chat template, no special prefix.
* Both sequences are padded to at least ``PAD_FLOOR`` (512) tokens; longer captions are allowed.
* Cache entries store trimmed tensors; padding is re-applied when a batch is assembled.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor, nn

from ypuddin.models.base import TextCond, TextPipeline

log = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"
PAD_FLOOR = 512
QWEN_HIDDEN = 1024
T5_PAD_ID = 0
T5_EOS_ID = 1


def _load_qwen3(path: str | Path, dtype: torch.dtype, device: torch.device | str) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    p = Path(path)
    if p.is_dir():
        model = AutoModelForCausalLM.from_pretrained(str(p), torch_dtype=dtype)
    else:
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(str(ASSETS / "qwen3_06b"))
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        sd = load_file(str(p))
        if not any(k.startswith("model.") for k in sd):
            # bare decoder-stack keys (layers.0...., embed_tokens.weight) -> HF causal-LM layout
            sd = {("lm_head.weight" if k == "lm_head.weight" else f"model.{k}"): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("lm_head")]
        if missing or unexpected:
            log.warning(
                "Qwen3 load: %d missing, %d unexpected keys (e.g. %s / %s)",
                len(missing),
                len(unexpected),
                missing[:2],
                unexpected[:2],
            )
    encoder = model.model  # decoder stack without the LM head
    encoder.config.use_cache = False
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.to(device)


def _ensure_one_token(ids: Tensor, mask: Tensor, fill_id: int) -> tuple[Tensor, Tensor]:
    """Empty captions (unconditional samples) keep exactly one valid token (EOS).

    A fully masked attention row is undefined for fused SDPA kernels (NaN on some backends), so the
    adapter's cross-attention must always see at least one key.
    """
    if ids.shape[1] == 0:
        ids = torch.full((ids.shape[0], 1), fill_id, dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return ids, mask
    empty = mask.sum(1) == 0
    if empty.any():
        ids = ids.clone()
        mask = mask.clone()
        ids[empty, 0] = fill_id
        mask[empty, 0] = True
    return ids, mask


class AnimaText(TextPipeline):
    fingerprint = "anima-qwen3-0.6b-last-hidden+t5old-v1"

    def __init__(
        self,
        text_encoder_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        max_len: int = 1024,
    ):
        from transformers import AutoTokenizer, T5TokenizerFast

        qwen_tok_dir = (
            Path(text_encoder_path)
            if Path(text_encoder_path).is_dir() and (Path(text_encoder_path) / "tokenizer.json").exists()
            else ASSETS / "qwen3_06b"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(qwen_tok_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path or (ASSETS / "t5_old")))
        self.max_len = max_len
        self.dtype = dtype
        self.device = torch.device(device)
        self._path = text_encoder_path
        self.encoder: nn.Module | None = None

    # ----------------------------------------------------------------- weights
    def _ensure_loaded(self) -> nn.Module:
        if self.encoder is None:
            self.encoder = _load_qwen3(self._path, self.dtype, self.device)
        return self.encoder

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.encoder is not None:
            self.encoder.to(self.device)

    def unload(self) -> None:
        self.encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------- tokenize / encode
    def _tokenize(self, captions: list[str]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        q = self.tokenizer(
            captions, padding="longest", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        t5 = self.t5_tokenizer(
            captions, padding="longest", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        q_ids, q_mask = _ensure_one_token(
            q["input_ids"], q["attention_mask"].bool(), int(self.tokenizer.eos_token_id)
        )
        t5_ids, t5_mask = _ensure_one_token(t5["input_ids"], t5["attention_mask"].bool(), T5_EOS_ID)
        return q_ids, q_mask, t5_ids, t5_mask

    @torch.no_grad()
    def encode_for_cache(self, captions: list[str]) -> list[dict[str, Tensor]]:
        enc = self._ensure_loaded()
        q_ids, q_mask, t5_ids, t5_mask = self._tokenize(captions)
        out = enc(
            input_ids=q_ids.to(self.device), attention_mask=q_mask.to(self.device), output_hidden_states=False
        )
        hidden = out.last_hidden_state.float().cpu()
        entries = []
        for i in range(len(captions)):
            n = int(q_mask[i].sum())
            m = int(t5_mask[i].sum())
            entries.append({"embeds": hidden[i, :n].contiguous(), "t5_ids": t5_ids[i, :m].contiguous()})
        return entries

    def cond_from_cache(self, entries: list[dict[str, Tensor]], device: torch.device | str) -> TextCond:
        lq = max(PAD_FLOOR, max(e["embeds"].shape[0] for e in entries))
        lt = max(PAD_FLOOR, max(e["t5_ids"].shape[0] for e in entries))
        b = len(entries)
        embeds = torch.zeros(b, lq, QWEN_HIDDEN, dtype=torch.float32)
        q_mask = torch.zeros(b, lq, dtype=torch.bool)
        t5_ids = torch.full((b, lt), T5_PAD_ID, dtype=torch.long)
        t5_mask = torch.zeros(b, lt, dtype=torch.bool)
        for i, e in enumerate(entries):
            n, m = e["embeds"].shape[0], e["t5_ids"].shape[0]
            embeds[i, :n] = e["embeds"].float()
            q_mask[i, :n] = True
            t5_ids[i, :m] = e["t5_ids"]
            t5_mask[i, :m] = True
        return TextCond({"embeds": embeds, "attn_mask": q_mask, "t5_ids": t5_ids, "t5_mask": t5_mask}).to(
            device
        )

    def encode(self, captions: list[str], device: torch.device | str) -> TextCond:
        return self.cond_from_cache(self.encode_for_cache(captions), device)
