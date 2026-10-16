"""Frozen Qwen3.5 last-state conditioning with unpadded per-caption evaluation."""
from pathlib import Path

import torch
from torch import Tensor

from ypuddin.models.base import TextCond, TextPipeline
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.memory import release_model_memory

from .loading import read_json, tokenizer_directory, validate_weights


class MOSSImageText(TextPipeline):
    max_len = 4096
    encoder_hidden_size = 4096
    encoder_layers = 32

    def __init__(self, path, tokenizer_path=None, *, device="cpu"):
        self.path = Path(path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else tokenizer_directory(self.path)
        self.config, self.weight_elements = self.validate_paths(self.path, self.tokenizer_path)
        self.hidden_size = self.encoder_hidden_size
        self.dtype, self.device = torch.bfloat16, torch.device(device)
        self.model, self.tokenizer = None, None
        self.fingerprint = content_fingerprint(
            [self.path, self.tokenizer_path, (self.path if self.path.is_dir() else self.path.parent) / "config.json"], namespace=f"moss-image-qwen35-last-thinking-unpadded-{self.max_len}-bf16-v1",
        )

    @classmethod
    def validate_paths(cls, path, tokenizer_path=None):
        path = Path(path)
        tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else tokenizer_directory(path)
        config = read_json((path if path.is_dir() else path.parent) / "config.json")
        text = config.get("text_config", {})
        if (config.get("model_type") != "qwen3_5" or text.get("hidden_size") != cls.encoder_hidden_size
                or text.get("num_hidden_layers") != cls.encoder_layers
                or config.get("quantization_config") or text.get("quantization_config")):
            raise ValueError("MOSS-Image 需要配套的未量化 Qwen3.5 文本编码器。")
        tok = read_json(tokenizer_path / "tokenizer_config.json")
        if not (tokenizer_path / "tokenizer.json").is_file():
            raise ValueError("MOSS-Image 分词器缺少 tokenizer.json。")
        if not tok.get("chat_template") and not (tokenizer_path / "chat_template.jinja").is_file():
            raise ValueError("MOSS-Image 分词器缺少 chat template。")
        try:
            from transformers import Qwen3_5Config, Qwen3_5Model
        except ImportError as error:
            raise ValueError("当前 Transformers 缺少 Qwen3.5 模型支持。") from error
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            model = Qwen3_5Model(Qwen3_5Config.from_dict(config))
        elements = validate_weights(path, model.state_dict())
        return config, elements

    def _ensure(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path), local_files_only=True, trust_remote_code=False)
            if not self.tokenizer.chat_template:
                raise ValueError("MOSS-Image 分词器缺少 chat template。")
        if self.model is None:
            from safetensors.torch import load_file
            from transformers import Qwen3_5Config, Qwen3_5Model

            kwargs = {}
            source = str(self.path)
            if self.path.is_file():
                source, kwargs = None, {"config": Qwen3_5Config.from_dict(self.config), "state_dict": load_file(str(self.path))}
            # Qwen3.5 initializes GDN constants before loading their saved values.
            with torch.random.fork_rng(devices=[]):
                model, info = Qwen3_5Model.from_pretrained(
                    source, local_files_only=True, use_safetensors=True, dtype=self.dtype,
                    low_cpu_mem_usage=True, output_loading_info=True, **kwargs,
                )
            if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"MOSS-Image 文本权重与配置不符：{info}")
            self.model = model.to(self.device).requires_grad_(False).eval()

    def to(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)

    def unload(self):
        if self.model is not None:
            self.model = None
            release_model_memory(self.device)

    @torch.no_grad()
    def _encode_caption(self, caption):
        self._ensure()
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": caption}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=self.max_len).to(self.device)
        with torch.autocast(self.device.type, enabled=False):
            return self.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].to(self.dtype)

    @torch.no_grad()
    def encode(self, captions, device):
        return self.cond_from_cache([{"embeds": self._encode_caption(caption)} for caption in captions], device)

    @torch.no_grad()
    def encode_for_cache(self, captions):
        return [{"embeds": self._encode_caption(caption).cpu().contiguous()} for caption in captions]

    def cond_from_cache(self, entries, device):
        rows = [entry.get("embeds") for entry in entries]
        if not rows or any(not isinstance(row, Tensor) or row.ndim != 2 or row.shape[1] != self.hidden_size
                           or not row.is_floating_point() or not 0 < row.shape[0] <= self.max_len for row in rows):
            raise ValueError("MOSS-Image 文本缓存的长度或隐藏维度不符。")
        length = max(row.shape[0] for row in rows)
        embeds = torch.zeros(len(rows), length, self.hidden_size, dtype=self.dtype, device=device)
        mask = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            embeds[index, :len(row)] = row.to(device=device, dtype=self.dtype)
            mask[index, :len(row)] = True
        return TextCond({"embeds": embeds, "mask": mask})
