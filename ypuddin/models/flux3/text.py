"""Frozen Qwen3-VL conditioning with FLUX.3's per-caption padded lengths."""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from ypuddin.models.base import TextCond, TextPipeline
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.memory import release_model_memory
from ypuddin.runtime_imports import LIBRARY_IMPORTS


class Flux3Text(TextPipeline):
    max_len = 4096
    pad_multiple = 80
    layers = (4, 8, 12, 16, 20, 24, 28, 32)
    encoder_hidden_size = 2560

    def __init__(self, path: str | Path, processor_path: str | Path | None = None, *, device="cpu"):
        self.path, self.processor_path, self.config_path = self._local_paths(path, processor_path)
        self.config, self.weight_elements = self.validate_paths(self.path, self.processor_path)
        self.dtype, self.device = torch.bfloat16, torch.device(device)
        self.hidden_size = self.encoder_hidden_size * len(self.layers)
        self.model, self.processor = None, None
        assets = [self.path, self.processor_path]
        if self.path.is_file():
            assets.append(self.config_path)
        self.fingerprint = content_fingerprint(
            assets,
            namespace=f"flux3-qwen3vl-layers{self.layers}-len{self.max_len}-pad{self.pad_multiple}-positions4-bf16-v1",
        )

    @staticmethod
    def _local_paths(path: str | Path, processor_path: str | Path | None = None):
        path = Path(path).expanduser()
        directory = path if path.is_dir() else path.parent
        processor_path = Path(processor_path).expanduser() if processor_path is not None else directory
        config_path = directory / "config.json"
        if path.is_file() and not config_path.is_file():
            config_path = processor_path / "config.json"
        return path, processor_path, config_path

    @classmethod
    def validate_paths(cls, path: str | Path, processor_path: str | Path | None = None):
        path, processor_path, config_path = cls._local_paths(path, processor_path)
        if not config_path.is_file():
            raise ValueError("FLUX.3 文本编码器需要本地 Qwen3-VL config.json。")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("FLUX.3 text encoder config must be an object")
        text = config.get("text_config", {})
        if (
            config.get("model_type") != "qwen3_vl"
            or not isinstance(text, dict)
            or text.get("hidden_size") != cls.encoder_hidden_size
            or not isinstance(text.get("num_hidden_layers"), int)
            or text["num_hidden_layers"] < max(cls.layers)
        ):
            raise ValueError("FLUX.3 requires Qwen3-VL text width 2560 and hidden states through layer 32")
        if config.get("quantization_config") or text.get("quantization_config"):
            raise ValueError("FLUX.3 requires unquantized Qwen3-VL text encoder weights")
        if not processor_path.is_dir() or not (processor_path / "tokenizer_config.json").is_file():
            raise ValueError("FLUX.3 requires a complete local processor directory with its chat template")
        processor_configs = []
        for name in ("tokenizer_config.json", "processor_config.json", "preprocessor_config.json"):
            asset = processor_path / name
            if asset.is_file():
                parsed = json.loads(asset.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError(f"FLUX.3 processor {name} must be an object")
                processor_configs.append(parsed)
        if not (processor_path / "tokenizer.json").is_file() and not all(
            (processor_path / name).is_file() for name in ("vocab.json", "merges.txt")
        ):
            raise ValueError("FLUX.3 processor is missing its local tokenizer vocabulary")
        if not any((processor_path / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
            raise ValueError("FLUX.3 processor is missing its local processor configuration")
        if not any(item.get("chat_template") for item in processor_configs) and not any(
            (processor_path / name).is_file() for name in ("chat_template.jinja", "chat_template.json", "chat_templates/default.jinja")
        ):
            raise ValueError("FLUX.3 processor is missing its chat template")
        files, weight_map = cls._weight_files(path)
        shapes, weight_elements = {}, 0
        for weight in files:
            with safe_open(str(weight), framework="pt") as tensors:
                for key in tensors.keys():
                    if key in shapes:
                        raise ValueError(f"FLUX.3 文本编码器分片包含重复权重：{key}")
                    if weight_map is not None and (
                        key not in weight_map or (path / weight_map[key]).resolve() != weight.resolve()
                    ):
                        raise ValueError(f"FLUX.3 文本编码器索引与权重分片不符：{key}")
                    tensor = tensors.get_slice(key)
                    if tensor.get_dtype() not in {"F32", "F16", "BF16", "F64"}:
                        raise ValueError("FLUX.3 requires unquantized Qwen3-VL text encoder weights")
                    if path.is_file() and tensor.get_dtype() != "BF16":
                        raise ValueError("FLUX.3 单文件文本编码器需要 BF16 权重。")
                    shapes[key] = tuple(tensor.get_shape())
                    weight_elements += math.prod(shapes[key])
        if weight_map is not None and set(weight_map) != set(shapes):
            missing = sorted(set(weight_map) - set(shapes))
            raise ValueError(f"FLUX.3 文本编码器分片缺少索引中的权重：{missing[:3]}")
        try:
            with LIBRARY_IMPORTS:
                from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise ValueError("当前 Transformers 缺少 Qwen3-VL 模型支持，无法读取 FLUX.3 文本编码器。") from error
        try:
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                expected_model = Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(config))
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"当前 Transformers 无法解析 FLUX.3 文本编码器配置：{error}") from error
        expected, present = expected_model.state_dict(), set(shapes)
        # Shared embeddings may be stored once; use the loader's actual tied parameters.
        aliases = {}
        for name, parameter in expected_model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), set()).add(name)
        shared_missing = set().union(*(names - present for names in aliases.values() if names & present))
        missing = sorted(set(expected) - present - shared_missing)
        unexpected = sorted(present - set(expected))
        mismatched = [key for key in shapes.keys() & expected.keys() if shapes[key] != tuple(expected[key].shape)]
        if missing or unexpected or mismatched:
            raise ValueError(
                "FLUX.3 text encoder weights do not match config: "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}, shapes={sorted(mismatched)[:3]}"
            )
        return config, weight_elements

    @staticmethod
    def _weight_files(path):
        if path.is_file():
            if path.suffix.lower() != ".safetensors":
                raise ValueError("FLUX.3 单文件文本编码器需要 safetensors 格式。")
            return [path], None
        index = path / "model.safetensors.index.json"
        weight_map = None
        if (path / "model.safetensors").is_file():
            files = [path / "model.safetensors"]
        elif index.is_file():
            parsed = json.loads(index.read_text(encoding="utf-8"))
            weight_map = parsed.get("weight_map") if isinstance(parsed, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("FLUX.3 text encoder has an invalid safetensors index")
            if any(not isinstance(name, str) for name in weight_map.values()):
                raise ValueError("FLUX.3 text encoder has an invalid weight shard name")
            names = set(weight_map.values())
            files = [path / name for name in sorted(names)]
        else:
            files = [path / "model.safetensors"]
        if any(not file.is_file() or file.suffix != ".safetensors" or not file.resolve().is_relative_to(path.resolve()) for file in files):
            raise ValueError("FLUX.3 text encoder directory is missing local safetensors weights")
        return files, weight_map

    def _ensure(self):
        if self.processor is None:
            with LIBRARY_IMPORTS:
                from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(str(self.processor_path), local_files_only=True)
            if processor.tokenizer.pad_token_id is None:
                raise ValueError("FLUX.3 text tokenizer has no padding token")
            if not processor.chat_template:
                raise ValueError("FLUX.3 processor is missing its chat template")
            processor.tokenizer.padding_side = "right"
            self.processor = processor
        if self.model is None:
            with LIBRARY_IMPORTS:
                from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

            kwargs = {}
            source = str(self.path)
            if self.path.is_file():
                from safetensors.torch import load_file

                source = None
                kwargs = {"config": Qwen3VLConfig.from_dict(self.config), "state_dict": load_file(str(self.path))}
            model, info = Qwen3VLForConditionalGeneration.from_pretrained(
                source, local_files_only=True, use_safetensors=True, dtype=self.dtype,
                low_cpu_mem_usage=True, output_loading_info=True, **kwargs,
            )
            if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"FLUX.3 text encoder weights do not match config: {info}")
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
    def _encode_caption(self, caption: str) -> Tensor:
        self._ensure()
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": caption}], tokenize=False, add_generation_prompt=True,
        )
        tokenizer = self.processor.tokenizer
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_len)["input_ids"].shape[1]
        length = min(math.ceil(real / self.pad_multiple) * self.pad_multiple, self.max_len)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"].to(self.device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        with torch.autocast(self.device.type, enabled=False):
            output = self.model(
                input_ids=inputs["input_ids"].to(self.device), attention_mask=mask,
                position_ids=positions[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False,
            )
        stacked = torch.stack([output.hidden_states[layer] for layer in self.layers], dim=1)
        return stacked.permute(0, 2, 1, 3).reshape(length, self.hidden_size).to(self.dtype)

    @torch.no_grad()
    def encode(self, captions, device):
        return self.cond_from_cache([{"embeds": self._encode_caption(caption)} for caption in captions], device)

    @torch.no_grad()
    def encode_for_cache(self, captions):
        return [{"embeds": self._encode_caption(caption).cpu().contiguous()} for caption in captions]

    def cond_from_cache(self, entries, device):
        if not entries:
            raise ValueError("FLUX.3 text conditioning requires at least one caption")
        rows = [entry.get("embeds") for entry in entries]
        if any(
            not isinstance(row, Tensor) or row.ndim != 2 or row.shape[1] != self.hidden_size
            or not row.is_floating_point() or not 0 < row.shape[0] <= self.max_len
            or (row.shape[0] != self.max_len and row.shape[0] % self.pad_multiple)
            for row in rows
        ):
            raise ValueError("FLUX.3 text cache has incompatible sequence or hidden dimensions")
        lengths = torch.tensor([row.shape[0] for row in rows], device=device, dtype=torch.long)
        embeds = torch.zeros(len(rows), max(row.shape[0] for row in rows), self.hidden_size, device=device, dtype=self.dtype)
        for index, row in enumerate(rows):
            embeds[index, : row.shape[0]] = row.to(device=device, dtype=self.dtype)
        return TextCond({"embeds": embeds, "lengths": lengths})
