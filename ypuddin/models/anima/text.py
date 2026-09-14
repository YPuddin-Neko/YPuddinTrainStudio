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


def tokenizer_assets(root: str | Path) -> list[Path]:
    """Tokenization inputs, excluding unrelated checkpoints in a shared models directory."""
    root = Path(root)
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
        "config.json",
        "chat_template.jinja",
    }
    return sorted(p for p in root.iterdir() if p.is_file() and p.name in names)


def _load_qwen3(path: str | Path, dtype: torch.dtype, device: torch.device | str) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    p = Path(path)
    if p.is_dir():
        model = AutoModelForCausalLM.from_pretrained(str(p), dtype=dtype)
    else:
        from safetensors.torch import load_file

        config_root = p.parent if (p.parent / "config.json").is_file() else ASSETS / "qwen3_06b"
        config = AutoConfig.from_pretrained(str(config_root), local_files_only=True)
        # The complete decoder is supplied by the checkpoint: allocating and randomly
        # initializing it first wastes CPU time and a second copy of its weights.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        sd = load_file(str(p))
        if not any(k.startswith("model.") for k in sd):
            # bare decoder-stack keys (layers.0...., embed_tokens.weight) -> HF causal-LM layout
            sd = {("lm_head.weight" if k == "lm_head.weight" else f"model.{k}"): v for k, v in sd.items()}
        # Safetensors may retain either name of a tied embedding. The decoder's
        # explicit weight takes precedence; an unused, separate LM head is ignored.
        embedding_key = "model.embed_tokens.weight"
        if config.tie_word_embeddings and embedding_key not in sd and "lm_head.weight" in sd:
            sd[embedding_key] = sd["lm_head.weight"]
        decoder_sd = {
            k.removeprefix("model."): v.to(dtype=dtype) if v.is_floating_point() else v
            for k, v in sd.items()
            if k != "lm_head.weight"
        }
        missing, unexpected = model.model.load_state_dict(decoder_sd, strict=False, assign=True)
        del sd, decoder_sd
        if missing:
            raise ValueError(
                f"Qwen3 checkpoint is missing required encoder weights: {', '.join(missing[:8])}"
            )
        if unexpected:
            log.warning(
                "Qwen3 load: %d unexpected keys (e.g. %s)",
                len(unexpected),
                unexpected[:2],
            )
        # RoPE frequencies are nonpersistent, so assign=True cannot materialize them.
        # Rebuild using the installed Transformers implementation, preserving FP32
        # frequencies and any version-specific original_inv_freq bookkeeping.
        with torch.device("cpu"):
            model.model.rotary_emb = type(model.model.rotary_emb)(config=config)
        if any(t.is_meta for t in (*model.model.parameters(), *model.model.buffers())):
            raise ValueError("Qwen3 encoder contains unmaterialized parameters or buffers after loading")
        if config.tie_word_embeddings:
            model.tie_weights()
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
        from ypuddin.models.fingerprints import content_fingerprint

        assets = [
            Path(text_encoder_path),
            *tokenizer_assets(qwen_tok_dir),
            *tokenizer_assets(tokenizer_path or ASSETS / "t5_old"),
        ]
        if Path(text_encoder_path).is_file():
            adjacent_config = Path(text_encoder_path).parent / "config.json"
            assets.append(
                adjacent_config if adjacent_config.is_file() else ASSETS / "qwen3_06b" / "config.json"
            )
        self.fingerprint = content_fingerprint(assets, namespace=f"{AnimaText.fingerprint}:{max_len}:{dtype}")

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
        hidden = entries[0]["embeds"].shape[1] if entries else QWEN_HIDDEN
        storage_device = entries[0]["embeds"].device
        embeds = torch.zeros(b, lq, hidden, dtype=torch.float32, device=storage_device)
        q_mask = torch.zeros(b, lq, dtype=torch.bool, device=storage_device)
        t5_ids = torch.full((b, lt), T5_PAD_ID, dtype=torch.long, device=storage_device)
        t5_mask = torch.zeros(b, lt, dtype=torch.bool, device=storage_device)
        for i, e in enumerate(entries):
            n, m = e["embeds"].shape[0], e["t5_ids"].shape[0]
            embeds[i, :n] = e["embeds"].float()
            q_mask[i, :n] = True
            t5_ids[i, :m] = e["t5_ids"]
            t5_mask[i, :m] = True
        return TextCond({"embeds": embeds, "attn_mask": q_mask, "t5_ids": t5_ids, "t5_mask": t5_mask}).to(
            device
        )

    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"text_encoder": self._ensure_loaded()}

    def encode(self, captions: list[str], device: torch.device | str) -> TextCond:
        if not getattr(self, "training_enabled", False):
            return self.cond_from_cache(self.encode_for_cache(captions), device)
        encoder = self._ensure_loaded()
        q_ids, q_mask, t5_ids, t5_mask = self._tokenize(captions)
        output = encoder(
            input_ids=q_ids.to(self.device), attention_mask=q_mask.to(self.device), use_cache=False
        )
        # Padding is applied with differentiable cat/copy operations, not the detached disk cache.
        hidden = output.last_hidden_state
        entries = [
            {"embeds": hidden[i, : int(q_mask[i].sum())], "t5_ids": t5_ids[i, : int(t5_mask[i].sum())]}
            for i in range(len(captions))
        ]
        return self.cond_from_cache(entries, device)
