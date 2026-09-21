"""Offline Qwen3 single-file metadata and strict state loading."""

from pathlib import Path

import torch

from .loading import read_json, shapes

ASSETS = Path(__file__).parent / "assets"


def text_config(path: Path) -> dict:
    sidecar = (path if path.is_dir() else path.parent) / "config.json"
    if sidecar.is_file():
        config = read_json(sidecar)
        if config.get("model_type") != "qwen3":
            raise ValueError("Klein requires Qwen3, not Qwen3-VL or another text encoder")
        return config
    if path.is_dir():
        raise ValueError("Klein text encoder directory is missing config.json")
    return infer_text_config(shapes(path))


def infer_text_config(tensors: dict) -> dict:
    tensors = {key: tuple(value) for key, value in tensors.items()}
    embed = tensors.get("model.embed_tokens.weight")
    size = {(151936, 2560): "4b", (151936, 4096): "8b"}.get(embed)
    if size is None:
        raise ValueError("Unrecognized Klein Qwen3 single-file embedding geometry")
    config = read_json(ASSETS / f"qwen3-{size}.json")
    width = config["hidden_size"]
    intermediate = config["intermediate_size"]
    if tensors.get("model.layers.0.mlp.gate_proj.weight") != (intermediate, width):
        raise ValueError("Klein Qwen3 MLP geometry mismatch; Qwen3-VL is not interchangeable")
    if not all(
        f"model.layers.{i}.self_attn.q_norm.weight" in tensors for i in range(config["num_hidden_layers"])
    ):
        raise ValueError("Incomplete Qwen3 single-file checkpoint")
    return config


def default_tokenizer(path: Path) -> Path:
    directory = path if path.is_dir() else path.parent
    return directory if (directory / "tokenizer_config.json").is_file() else ASSETS / "tokenizer"


def load_single_text(path: Path, config: dict, dtype):
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    state = load_file(str(path))
    if config.get("tie_word_embeddings") and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config(**config))
    model.load_state_dict(
        {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}, strict=True, assign=True
    )
    model.tie_weights()
    with torch.device("cpu"):
        model.model.rotary_emb = type(model.model.rotary_emb)(config=model.config)
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Qwen3 checkpoint left uninitialized parameters or buffers")
    return model
