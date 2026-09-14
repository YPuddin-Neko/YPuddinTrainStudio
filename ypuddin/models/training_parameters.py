"""Count the exact selected text architecture on meta, never by loading its weights."""

from pathlib import Path

import torch


def text_parameter_count(family, cfg) -> int:
    name = family.spec.name
    # These libraries lazily initialize Dynamo during class import. Performing
    # that import on meta can poison subsequent model imports in the service.
    # Only configurations are prepared here; parameters are constructed below.
    with torch.device("cpu"):
        if name == "toy":
            from .toy import DIM, MAX_LEN, VOCAB

            return (VOCAB + MAX_LEN) * DIM
        if name == "anima":
            from transformers import AutoConfig, Qwen3Model

            from .anima.text import ASSETS

            path = Path(cfg.text_encoder_path or "")
            root = path if path.is_dir() else path.parent
            config = AutoConfig.from_pretrained(
                str(root if (root / "config.json").is_file() else ASSETS / "qwen3_06b"), local_files_only=True
            )
            constructors = [(Qwen3Model, config)]
        elif name == "krea2":
            from transformers import Qwen3VLTextModel

            from .krea2.text import text_config_for

            constructors = [(Qwen3VLTextModel, text_config_for(cfg.text_encoder_path or ""))]
        elif name == "sdxl":
            from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

            from .sdxl.loading import component_config, component_path

            root = cfg.dit_path or "."
            constructors = [
                (cls, CLIPTextConfig(**component_config(component_path(root, component, override), component)))
                for component, override, cls in (
                    ("text_encoder", cfg.text_encoder_path, CLIPTextModel),
                    ("text_encoder_2", cfg.text_encoder_2_path, CLIPTextModelWithProjection),
                )
            ]
        elif name == "flux2":
            from transformers import Qwen3Config, Qwen3ForCausalLM

            path = family._paths(cfg)[2]
            constructors = [(Qwen3ForCausalLM, Qwen3Config.from_pretrained(str(path), local_files_only=True))]
        else:
            raise ValueError(f"{name} has no text-encoder training architecture")
    with torch.device("meta"):
        return sum(p.numel() for cls, config in constructors for p in cls(config).parameters())
