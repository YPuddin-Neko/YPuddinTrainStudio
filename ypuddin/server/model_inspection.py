"""Bounded local metadata inspection; never deserialize pickle or load model tensors."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any

HEADER_LIMIT = 32 * 1024 * 1024
CONFIG_LIMIT = 2 * 1024 * 1024
TOKENIZER_LIMIT = 32 * 1024 * 1024
MAX_FILES = 1024
FLUX1_RETIRED_REASON = "当前 FLUX 训练仅保留 FLUX.2 Klein base 4B/9B；不再接入 FLUX.1，已有记录和文件仍保留。"
FLUX2_DEV_UNSUPPORTED_REASON = (
    "检测到 FLUX.2 dev；当前仅支持 FLUX.2 Klein base 4B/9B，dev 及其 Mistral 编码器不能登记为 Klein。"
)


def training_rejection(detected: dict[str, Any], family: str | None = None) -> str | None:
    """Shared admission gate; diagnostic family identities remain unchanged."""
    if family == "flux" or detected.get("family") == "flux":
        return FLUX1_RETIRED_REASON
    return detected.get("unsupported_reason")


def _json(path: Path, limit: int) -> dict[str, Any]:
    if path.stat().st_size > limit:
        raise ValueError(f"{path.name}: metadata exceeds the inspection limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: metadata must be a JSON object")
    return value


def _header(path: Path) -> dict[str, dict]:
    # Read only the bounded JSON header; safe_open validates offsets and shapes
    # through its mmap without materializing any tensor or allocating GPU memory.
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 2 <= length <= HEADER_LIMIT or length + 8 > path.stat().st_size:
            raise ValueError(f"{path.name}: invalid or oversized safetensors header")
    from safetensors import SafetensorError, safe_open

    tensors = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                item = handle.get_slice(key)
                tensors[key] = {"shape": item.get_shape(), "dtype": item.get_dtype()}
    except SafetensorError as error:
        raise ValueError(f"{path.name}: invalid safetensors metadata") from error
    return tensors


def _key(key: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "net.", "first_stage_model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def sdxl_component(shapes: dict[str, list[int]]) -> str | None:
    """Recognize SDXL geometry before a bundled checkpoint's VAE/CLIP keys.

    Names alone are insufficient: SD1/SD2 share the original UNet prefix.
    SDXL base combines 4 latent channels with 2816-dimensional added conditioning.
    """
    shapes = {_key(key): shape for key, shape in shapes.items()}
    conv = shapes.get("input_blocks.0.0.weight", shapes.get("conv_in.weight", []))
    added = shapes.get("label_emb.0.0.weight", shapes.get("add_embedding.linear_1.weight", []))
    if len(conv) == 4 and conv[1] == 4 and len(added) == 2 and added[1] == 2816:
        return "dit"
    clip_kinds = set()
    for prefix in ("", "conditioner.embedders.0.transformer.", "conditioner.embedders.1.model."):
        keys = {key.removeprefix(prefix): value for key, value in shapes.items() if key.startswith(prefix)}
        embed = next(
            (
                keys[name]
                for name in (
                    "text_model.embeddings.token_embedding.weight",
                    "embeddings.token_embedding.weight",
                    "token_embedding.weight",
                )
                if name in keys
            ),
            [],
        )
        projection = keys.get("text_projection.weight", keys.get("text_projection", []))
        if embed == [49408, 1280] and projection == [1280, 1280]:
            clip_kinds.add("text_encoder_2")
        elif embed == [49408, 768]:
            clip_kinds.add("text_encoder")
    if len(clip_kinds) == 1:
        return clip_kinds.pop()
    if (
        any(key.startswith("encoder.") for key in shapes)
        and any(key.startswith("decoder.") for key in shapes)
        and shapes.get("quant_conv.weight") == [8, 8, 1, 1]
        and shapes.get("post_quant_conv.weight") == [4, 4, 1, 1]
    ):
        return "vae"
    return None


def _contained(root: Path, path: Path, allowed, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or allowed and not allowed(resolved):
        raise ValueError(f"{label} is outside the selected directory or allowed storage roots")
    return resolved


def flux_component(shapes: dict[str, list[int]], config: dict | None = None) -> dict[str, Any] | None:
    """Architecture evidence shared by inspection and single-file download validation.

    FLUX.1/2 use different packed latent widths. Klein base/distilled share tensor
    geometry; neither filenames nor that geometry establishes the training variant.
    """
    shapes = {_key(key): shape for key, shape in shapes.items()}
    config = config or {}

    def result(family, kind, candidates, evidence, warnings=(), unsupported_reason=None):
        return dict(
            family=family,
            kind=kind,
            candidates=candidates,
            evidence=[evidence],
            warnings=list(warnings),
            unsupported_reason=unsupported_reason,
        )

    image = shapes.get("img_in.weight", shapes.get("x_embedder.weight", []))
    text = shapes.get("txt_in.weight", shapes.get("context_embedder.weight", []))
    double = any(key.startswith(("double_blocks.0.", "transformer_blocks.0.")) for key in shapes)
    single = any(key.startswith(("single_blocks.0.", "single_transformer_blocks.0.")) for key in shapes)
    if image == [3072, 64] and text == [3072, 4096] and double and single:
        guided = any(key.startswith(("guidance_in.", "time_text_embed.guidance_embedder.")) for key in shapes)
        return result(
            "flux",
            "dit",
            ["flux"],
            f"FLUX.1 64-channel packed input, T5 width 4096 and dual/single-stream blocks; guidance={guided}",
            ["仅凭 FLUX.1 主干形状不能排除同形状的 Kontext 等衍生模型；具体支持范围由加载器校验。"],
            unsupported_reason=FLUX1_RETIRED_REASON,
        )
    widths = {6144: (15360, "dev"), 3072: (7680, "Klein 4B"), 4096: (12288, "Klein 9B")}
    if len(image) == 2 and image[1] == 128 and image[0] in widths and double and single:
        text_width, variant = widths[image[0]]
        modulation = any(
            key in shapes
            for key in ("single_stream_modulation.lin.weight", "single_stream_modulation.linear.weight")
        )
        if text == [image[0], text_width] and modulation:
            warnings = (
                []
                if variant == "dev"
                else [
                    f"识别到 FLUX.2 {variant} 架构候选；base 与 distilled 权重形状相同，需通过 flux2_variant 或可信模型配置确认。"
                ]
            )
            return result(
                "flux2",
                "dit",
                ["flux2"],
                f"FLUX.2 {variant}: 128-channel packed input and shared stream modulation",
                warnings,
                unsupported_reason=FLUX2_DEV_UNSUPPORTED_REASON if variant == "dev" else None,
            )
    encoder = shapes.get("encoder.conv_out.weight", [])
    decoder = shapes.get("decoder.conv_in.weight", [])
    if len(encoder) == len(decoder) == 4 and any(key.startswith("encoder.") for key in shapes):
        if (
            encoder[0] == 64
            and decoder[1] == 32
            and shapes.get("bn.running_mean") == [128]
            and shapes.get("bn.running_var") == [128]
        ):
            return result(
                "flux2",
                "vae",
                ["flux2"],
                "FLUX.2 32-channel spatial VAE with 128-channel patchified batchnorm statistics",
            )
        if (
            encoder[0] == 32
            and decoder[1] == 16
            and "quant_conv.weight" not in shapes
            and "post_quant_conv.weight" not in shapes
        ):
            return result(
                None,
                "vae",
                ["flux"],
                "16-channel spatial Autoencoder geometry compatible with FLUX.1; normalization comes from its config",
            )
    t5 = shapes.get("shared.weight", shapes.get("encoder.embed_tokens.weight", []))
    if t5 == [32128, 4096] and any(
        key.startswith("encoder.block.") and ".SelfAttention.q.weight" in key for key in shapes
    ):
        return result(
            None,
            "text_encoder_2",
            ["flux"],
            "T5-XXL encoder embedding and attention geometry; FLUX.1's second text encoder",
        )
    embed = next(
        (shape for key, shape in shapes.items() if key.endswith("embed_tokens.weight") and len(shape) == 2),
        [],
    )
    decoder_attention = any(".self_attn.q_proj.weight" in key for key in shapes)
    text_config = config.get("text_config", {})
    text_config = text_config if isinstance(text_config, dict) else {}
    types = {
        value for value in (config.get("model_type"), text_config.get("model_type")) if isinstance(value, str)
    }
    if decoder_attention and embed == [131072, 5120]:
        family = "flux2" if types & {"mistral3", "mistral"} else None
        return result(
            family,
            "text_encoder",
            ["flux2"],
            "Mistral 5120-wide text decoder compatible with FLUX.2 dev; complete local processor/model directory required",
            unsupported_reason=FLUX2_DEV_UNSUPPORTED_REASON,
        )
    if (
        decoder_attention
        and embed in ([151936, 2560], [151936, 4096])
        and any("q_norm.weight" in key for key in shapes)
    ):
        if types & {"qwen3_vl", "qwen3_vl_text"}:
            return None  # Preserve the explicit Qwen3-VL/Krea2 classification below.
        native_qwen3 = shapes.get("model.layers.0.mlp.gate_proj.weight") == (
            [9728, 2560] if embed[1] == 2560 else [12288, 4096]
        ) and all(f"model.layers.{i}.self_attn.q_norm.weight" in shapes for i in range(36))
        if "qwen3" in types or native_qwen3:
            return result(
                "flux2",
                "text_encoder",
                ["flux2"],
                "Qwen3 config and 4B/8B decoder geometry compatible with FLUX.2 Klein",
            )
        candidates = ["krea2", "flux2"] if embed[1] == 2560 else ["flux2"]
        return result(
            None,
            "text_encoder",
            candidates,
            "Qwen decoder geometry; bare tensors do not distinguish Qwen3 from Qwen3-VL",
            ["请保留文本编码器完整 HF 配置；不能由共享的 Qwen 权重形状唯一确定模型系列。"],
        )
    return None


def _local_references(config: dict, root: Path, allowed, label: str) -> None:
    # `_name_or_path`/`name_or_path` record model provenance, not runtime file references.
    for key in (
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "sp_model_file",
        "sentencepiece_model_file",
        "chat_template_file",
    ):
        value = config.get(key)
        if value is None:
            continue
        if (
            not isinstance(value, str)
            or Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or "\\" in value
        ):
            raise ValueError(f"{label} {key} must reference a local relative asset")
        file = _contained(root, root / value, allowed, f"{label} {key}")
        if not file.is_file():
            raise ValueError(f"{label} {key} references a missing local asset")


def _tokenizer_assets(folder: Path, allowed, label: str, *, chat=False, limit=TOKENIZER_LIMIT) -> set[str]:
    config_path = _contained(folder, folder / "tokenizer_config.json", allowed, f"{label} config")
    if not config_path.is_file():
        raise ValueError(f"{label}: tokenizer_config.json is required")
    config = _json(config_path, CONFIG_LIMIT)
    classes = {
        value for key in ("tokenizer_class", "processor_class") if isinstance(value := config.get(key), str)
    }
    _local_references(config, folder, allowed, label)
    if (folder / "tokenizer.json").is_file():
        vocabulary = ("tokenizer.json",)
    elif (folder / "vocab.json").is_file():
        vocabulary = ("vocab.json", "merges.txt")
    else:
        vocabulary = tuple(name for name in ("spiece.model", "tokenizer.model") if (folder / name).is_file())
    if not vocabulary:
        raise ValueError(f"{label}: local tokenizer vocabulary is required")
    for name in vocabulary:
        file = _contained(folder, folder / name, allowed, f"{label}/{name}")
        if not file.is_file() or not 0 < file.stat().st_size <= limit:
            raise ValueError(f"{label}/{name}: missing, empty or oversized tokenizer asset")
        if name.endswith(".json") and not _json(file, limit):
            raise ValueError(f"{label}/{name}: tokenizer asset must not be empty")
    templates = []
    for name in (
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
    ):
        path = folder / name
        if path.exists() or path.is_symlink():
            file = _contained(folder, path, allowed, f"{label}/{name}")
            if not file.is_file() or not 0 < file.stat().st_size <= CONFIG_LIMIT:
                raise ValueError(f"{label}/{name}: invalid processor/template asset")
            if name.endswith(".json"):
                value = _json(file, CONFIG_LIMIT)
                _local_references(value, folder, allowed, label)
                if isinstance(value.get("processor_class"), str):
                    classes.add(value["processor_class"])
            if name.startswith("chat_template"):
                templates.append(file)
    template_dir = folder / "chat_templates"
    if template_dir.exists() or template_dir.is_symlink():
        directory = _contained(folder, template_dir, allowed, f"{label}/chat_templates")
        for path in directory.glob("*.jinja"):
            file = _contained(folder, path, allowed, f"{label} chat template")
            if not file.is_file() or not 0 < file.stat().st_size <= CONFIG_LIMIT:
                raise ValueError(f"{label}: invalid chat template")
            templates.append(file)
            if len(templates) > MAX_FILES:
                raise ValueError(f"{label}: too many chat templates")
    if chat and not config.get("chat_template") and not templates:
        raise ValueError(f"{label}: FLUX.2 requires its local chat template")
    return classes


def _flux_directory(root: Path, allowed, budget: dict[str, int]) -> dict[str, Any] | None:
    path = root / "model_index.json"
    if not path.is_file():
        return None
    index = _json(_contained(root, path, allowed, "model index"), CONFIG_LIMIT)
    cls = index.get("_class_name", "")
    if not isinstance(cls, str):
        raise ValueError("model index _class_name must be a string")
    if cls not in {"FluxPipeline", "Flux2Pipeline", "Flux2KleinPipeline"}:
        if isinstance(cls, str) and cls.startswith("Flux"):
            raise ValueError(f"Unsupported FLUX pipeline class for local training: {cls}")
        return None
    family = "flux" if cls == "FluxPipeline" else "flux2"
    klein = cls == "Flux2KleinPipeline"
    expected = {
        "transformer": (
            "diffusers",
            "FluxTransformer2DModel" if family == "flux" else "Flux2Transformer2DModel",
            "dit",
        ),
        "text_encoder": (
            "transformers",
            "CLIPTextModel"
            if family == "flux"
            else "Qwen3ForCausalLM"
            if klein
            else "Mistral3ForConditionalGeneration",
            "text_encoder",
        ),
        "vae": ("diffusers", "AutoencoderKL" if family == "flux" else "AutoencoderKLFlux2", "vae"),
    }
    if family == "flux":
        expected["text_encoder_2"] = ("transformers", "T5EncoderModel", "text_encoder_2")
    components, configs = [], {}
    for name, (library, model_class, kind) in expected.items():
        if index.get(name) != [library, model_class]:
            raise ValueError(f"{family} model index must declare local {name}: {model_class}")
        folder = _contained(root, root / name, allowed, f"{family} {name}")
        file = _contained(folder, folder / "config.json", allowed, f"{family} {name} config")
        if not file.is_file():
            raise ValueError(f"{family} directory is incomplete: {name}/config.json is required")
        config = _json(file, CONFIG_LIMIT)
        _local_references(config, folder, allowed, f"{family} {name} config")
        if config.get("_class_name", model_class) != model_class:
            raise ValueError(f"{family} {name}/config.json declares an incompatible component class")
        if family == "flux2" and name == "text_encoder":
            if config.get("model_type") != ("qwen3" if klein else "mistral3"):
                raise ValueError(f"{family} {name}/config.json declares an incompatible text encoder")
        configs[name] = config
        result = inspect_model(folder, allowed=allowed, _budget=budget)
        if (
            result["kind"] != kind
            or family not in result["family_candidates"]
            or not result["files_inspected"]
        ):
            raise ValueError(f"{family} {name} needs compatible complete local safetensors weights")
        components.append(result)
    transformer = configs["transformer"]
    text = configs["text_encoder"]
    text = text.get("text_config", text)
    text = text if isinstance(text, dict) else {}
    if family == "flux":
        geometry = (
            transformer.get("in_channels") == 64
            and transformer.get("joint_attention_dim") == 4096
            and transformer.get("pooled_projection_dim") == 768
        )
        geometry &= (
            configs["text_encoder"].get("hidden_size") == 768
            and configs["text_encoder_2"].get("d_model") == 4096
            and configs["vae"].get("latent_channels") == 16
        )
        tokenizers = {
            "tokenizer": {"CLIPTokenizer", "CLIPTokenizerFast"},
            "tokenizer_2": {"T5Tokenizer", "T5TokenizerFast"},
        }
    else:
        joint = transformer.get("joint_attention_dim")
        geometry = (
            transformer.get("in_channels") == 128
            and isinstance(joint, int)
            and joint in ({7680, 12288} if klein else {15360})
        )
        geometry &= (
            text.get("hidden_size") == (joint // 3 if isinstance(joint, int) else None)
            and configs["vae"].get("latent_channels") == 32
        )
        tokenizers = {
            "tokenizer": {"Qwen2Tokenizer", "Qwen2TokenizerFast", "AutoTokenizer"}
            if klein
            else {"PixtralProcessor", "AutoProcessor"}
        }
    if not geometry:
        raise ValueError(f"{family} pipeline component configurations have incompatible geometry")
    for name, classes in tokenizers.items():
        declaration = index.get(name)
        if (
            not isinstance(declaration, list)
            or len(declaration) != 2
            or declaration[0] != "transformers"
            or not isinstance(declaration[1], str)
            or declaration[1] not in classes
        ):
            raise ValueError(f"{family} model index must declare local {name}: {', '.join(sorted(classes))}")
        folder = _contained(root, root / name, allowed, f"{family} {name}")
        _tokenizer_assets(folder, allowed, f"{family} {name}", chat=family == "flux2")
    dtypes = Counter()
    for component in components:
        dtypes.update(component["dtypes"])
    precisions = {component["dtype"] for component in components if component["dtype"] is not None}
    dtype = next(iter(precisions)) if len(precisions) == 1 else "mixed" if precisions else None
    warnings = ["矩阵权重包含多种精度，已标记为 mixed。"] if dtype == "mixed" else []
    if klein:
        warnings.append(
            "Klein base/distilled 不能由形状区分；加载器将结合 is_distilled 配置与 flux2_variant 校验。"
        )
        if index.get("is_distilled") is True:
            warnings.append("模型配置声明 is_distilled=true；当前训练加载器不支持 distilled/KV。")
    unsupported_reason = (
        FLUX1_RETIRED_REASON if family == "flux" else FLUX2_DEV_UNSUPPORTED_REASON if not klein else None
    )
    if unsupported_reason:
        warnings.append(unsupported_reason)
    return dict(
        path=str(root),
        family=family,
        family_candidates=[family],
        kind="dit",
        dtype=dtype,
        dtypes=dict(dtypes),
        confidence="high",
        evidence=[f"Local {cls} directory: transformer, required text encoders, VAE and tokenizer assets"],
        warnings=warnings,
        files_inspected=sum(item["files_inspected"] for item in components),
        unsupported_reason=unsupported_reason,
    )


def _sdxl_directory(root: Path, allowed, budget: dict[str, int]) -> dict[str, Any] | None:
    index_path = root / "model_index.json"
    if not index_path.is_file():
        return None
    index = _json(_contained(root, index_path, allowed, "model index"), CONFIG_LIMIT)
    if not str(index.get("_class_name", "")).startswith("StableDiffusionXL"):
        return None
    expected = {
        "unet": ("diffusers", "UNet2DConditionModel", "dit"),
        "text_encoder": ("transformers", "CLIPTextModel", "text_encoder"),
        "text_encoder_2": ("transformers", "CLIPTextModelWithProjection", "text_encoder_2"),
        "vae": ("diffusers", "AutoencoderKL", "vae"),
    }
    components = []
    for name, (library, model_class, kind) in expected.items():
        if index.get(name) != [library, model_class]:
            raise ValueError(f"SDXL model_index.json must declare local {name}: {model_class}")
        folder = _contained(root, root / name, allowed, f"SDXL {name}")
        if not folder.is_dir() or not (folder / "config.json").is_file():
            raise ValueError(f"SDXL directory is incomplete: {name}/config.json is required")
        config = _json(
            _contained(folder, folder / "config.json", allowed, f"SDXL {name} config"), CONFIG_LIMIT
        )
        geometry = {
            "unet": {
                "in_channels": 4,
                "out_channels": 4,
                "cross_attention_dim": 2048,
                "addition_embed_type": "text_time",
                "projection_class_embeddings_input_dim": 2816,
            },
            "text_encoder": {"hidden_size": 768, "vocab_size": 49408},
            "text_encoder_2": {"hidden_size": 1280, "vocab_size": 49408, "projection_dim": 1280},
            "vae": {"latent_channels": 4},
        }[name]
        if any(config.get(key) != value for key, value in geometry.items()):
            raise ValueError(f"SDXL {name}/config.json has incompatible base-model geometry")
        result = inspect_model(folder, allowed=allowed, _budget=budget)
        compatible = result["family"] == "sdxl" or "sdxl" in result["family_candidates"]
        if result["kind"] != kind or not compatible or not result["files_inspected"]:
            raise ValueError(f"SDXL {name} needs complete compatible safetensors weights")
        components.append(result)
    for name in ("tokenizer", "tokenizer_2"):
        if index.get(name) not in (["transformers", "CLIPTokenizer"], ["transformers", "CLIPTokenizerFast"]):
            raise ValueError(f"SDXL model_index.json must declare {name}: CLIPTokenizer")
        folder = _contained(root, root / name, allowed, f"SDXL {name}")
        vocabulary = (
            ("tokenizer.json",) if (folder / "tokenizer.json").is_file() else ("vocab.json", "merges.txt")
        )
        for asset in ("tokenizer_config.json", *vocabulary):
            file = _contained(folder, folder / asset, allowed, f"SDXL {name}/{asset}")
            if not file.is_file() or not file.stat().st_size:
                raise ValueError(f"SDXL directory is incomplete: {name}/{asset} is required")
            if file.stat().st_size > CONFIG_LIMIT:
                raise ValueError(f"SDXL {name}/{asset} exceeds the inspection limit")
            if asset.endswith(".json") and not _json(file, CONFIG_LIMIT):
                raise ValueError(f"SDXL {name}/{asset} must not be empty")
    dtypes = Counter()
    for component in components:
        dtypes.update(component["dtypes"])
    precisions = {component["dtype"] for component in components if component["dtype"] is not None}
    dtype = next(iter(precisions)) if len(precisions) == 1 else "mixed" if precisions else None
    return {
        "path": str(root),
        "family": "sdxl",
        "family_candidates": ["sdxl"],
        "kind": "dit",
        "dtype": dtype,
        "dtypes": dict(dtypes),
        "confidence": "high",
        "evidence": [
            "Complete local SDXL Diffusers directory: UNet, CLIP-L, CLIP-G, VAE and both tokenizers"
        ],
        "warnings": ["矩阵权重包含多种精度，已标记为 mixed。"] if dtype == "mixed" else [],
        "files_inspected": sum(component["files_inspected"] for component in components),
    }


def inspect_model(path: Path, *, allowed=None, _budget: dict[str, int] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if allowed and not allowed(path):
        raise ValueError("model path is outside allowed storage roots")
    if not path.exists():
        raise ValueError("model path does not exist")
    budget = _budget if _budget is not None else {"files": 0, "headers": 0}
    if path.is_dir():
        directory = _sdxl_directory(path, allowed, budget)
        if directory is not None:
            return directory
        directory = _flux_directory(path, allowed, budget)
        if directory is not None:
            return directory
    root = path if path.is_dir() else path.parent
    config_path = root / "config.json"
    config = (
        _json(_contained(root, config_path, allowed, "model config"), CONFIG_LIMIT)
        if config_path.is_file()
        else {}
    )
    _local_references(config, root, allowed, "model config")
    files = []
    mapping = {}
    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            raise ValueError(
                "Only safetensors files and local model/tokenizer directories can be inspected; pickle is never executed"
            )
        files = [path]
    else:
        indexes = [
            root / name
            for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
            if (root / name).is_file()
        ]
        if len(indexes) > 1:
            raise ValueError("multiple safetensors indexes in the selected directory; select one component")
        if indexes:
            index = indexes[0]
            if not index.resolve().is_relative_to(root) or allowed and not allowed(index.resolve()):
                raise ValueError("shard index is outside the selected directory")
            mapping = _json(index, CONFIG_LIMIT).get("weight_map", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("safetensors shard index has no weight_map")
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
                raise ValueError("invalid safetensors shard mapping")
            names = set(mapping.values())
            if len(names) > MAX_FILES:
                raise ValueError("too many model shards")
            for name in sorted(names):
                file = (root / name).resolve()
                if (
                    Path(name).is_absolute()
                    or PureWindowsPath(name).is_absolute()
                    or not file.is_relative_to(root)
                    or file.suffix.lower() != ".safetensors"
                    or not file.is_file()
                ):
                    raise ValueError("model shard is missing or outside the selected directory")
                files.append(file)
        else:
            files = sorted(root.glob("*.safetensors"))
            if len(files) > MAX_FILES:
                raise ValueError("too many weight files in the selected directory")
    shapes = {}
    dtypes = Counter()
    weight_dtypes = Counter()
    budget["files"] += len(files)
    if budget["files"] > MAX_FILES:
        raise ValueError("too many weight files in the selected directory")
    for file in files:
        if path.is_dir() and not file.resolve().is_relative_to(root):
            raise ValueError("model shard is outside the selected directory")
        if allowed and not allowed(file.resolve()):
            raise ValueError("model shard is outside allowed storage roots")
        with file.open("rb") as stream:
            budget["headers"] += int.from_bytes(stream.read(8), "little")
        if budget["headers"] > 64 * 1024 * 1024:
            raise ValueError("combined model headers exceed the inspection limit")
        for key, item in _header(file).items():
            if mapping and (key not in mapping or (root / mapping[key]).resolve() != file.resolve()):
                raise ValueError("model shard index does not match tensor locations")
            key = _key(key)
            if key in shapes:
                raise ValueError(
                    "multiple checkpoints contain duplicate tensor keys; select one file or a complete sharded model directory"
                )
            shapes[key] = item["shape"]
            count = math.prod(item["shape"])
            dtypes[item["dtype"]] += count
            if key.endswith("weight") and len(item["shape"]) >= 2:
                weight_dtypes[item["dtype"]] += count
    if mapping and not {_key(key) for key in mapping} <= shapes.keys():
        raise ValueError("model shard index references missing tensors")
    dtype_names = {
        "F32": "fp32",
        "F16": "fp16",
        "BF16": "bf16",
        "F8_E4M3": "fp8",
        "F8_E4M3FN": "fp8",
        "F8_E5M2": "fp8",
    }
    floating = {
        dtype_names[k] for k, count in (weight_dtypes or dtypes).items() if count and k in dtype_names
    }
    dtype = next(iter(floating)) if len(floating) == 1 else "mixed" if floating else None
    if "fp8" in floating and any(
        key.endswith((".scale_weight", ".weight_scale")) or key == "scaled_fp8" for key in shapes
    ):
        dtype = "fp8"
    family, kind, evidence, candidates = None, None, [], []
    warnings = []
    unsupported_reason = None
    sdxl_kind = sdxl_component(shapes)
    flux = flux_component(shapes, config)
    if sdxl_kind == "dit":
        family, kind = "sdxl", "dit"
        evidence.append("SDXL base UNet: four latent channels and 2816-wide size/pooled conditioning")
        if any(key.startswith("conditioner.embedders.") for key in shapes):
            evidence.append("Checkpoint includes SDXL text encoder components")
    elif flux is not None:
        family, kind, candidates = flux["family"], flux["kind"], flux["candidates"]
        evidence.extend(flux["evidence"])
        warnings.extend(flux["warnings"])
        unsupported_reason = flux["unsupported_reason"]
    elif sdxl_kind in ("text_encoder", "text_encoder_2"):
        kind = sdxl_kind
        candidates = ["sdxl", "flux"] if kind == "text_encoder" else ["sdxl"]
        evidence.append(
            "CLIP-L geometry" if kind == "text_encoder" else "CLIP-G with text projection geometry"
        )
    elif (
        "x_embedder.proj.1.weight" in shapes
        and len(shapes["x_embedder.proj.1.weight"]) == 2
        and any(key.startswith("llm_adapter.") for key in shapes)
        and any(key.startswith("blocks.0.") for key in shapes)
    ):
        family, kind = "anima", "dit"
        evidence.append("Anima patch embedding + LLM adapter + transformer block keys")
    elif {"first.weight", "txtfusion.projector.weight", "blocks.0.attn.wq.weight"} <= shapes.keys() and len(
        shapes["first.weight"]
    ) == 2:
        family, kind = "krea2", "dit"
        evidence.append("Krea 2 single-stream projection + layerwise text-fusion keys")
    elif (
        shapes.get("conv1.weight") == [32, 32, 1, 1, 1]
        and shapes.get("conv2.weight") == [16, 16, 1, 1, 1]
        and shapes.get("encoder.conv1.weight") == [96, 3, 3, 3, 3]
        and shapes.get("decoder.conv1.weight") == [384, 16, 3, 3, 3]
        and shapes.get("encoder.head.2.weight") == [32, 384, 3, 3, 3]
    ):
        kind = "vae"
        candidates = ["anima", "krea2"]
        evidence.append("Native Qwen-Image causal VAE convolution geometry; shared by Anima and Krea 2")
    elif (
        any(key.startswith("encoder.") for key in shapes)
        and any(key.startswith("decoder.") for key in shapes)
        and "quant_conv.weight" in shapes
    ):
        kind = "vae"
        if shapes.get("post_quant_conv.weight", [])[:2] == [16, 16] and len(shapes["quant_conv.weight"]) == 5:
            candidates = ["anima", "krea2"]
            evidence.append("Qwen-Image causal VAE geometry; shared by Anima and Krea 2")
        elif sdxl_kind == "vae":
            candidates = ["sdxl"]
            evidence.append(
                "Four-channel spatial VAE geometry compatible with SDXL; also shared by older SD models"
            )
        else:
            evidence.append("Encoder / decoder / quantization convolution keys; family compatibility unknown")
    else:
        model_type = str(config.get("model_type", ""))
        text_cfg = config.get("text_config", {})
        text_cfg = text_cfg if isinstance(text_cfg, dict) else {}
        embed = next(
            (
                shape
                for key, shape in shapes.items()
                if key.endswith("embed_tokens.weight") and len(shape) == 2
            ),
            None,
        )
        decoder = any(".self_attn.q_proj.weight" in key for key in shapes)
        if embed and decoder:
            kind = "text_encoder"
            if model_type in ("qwen3_vl", "qwen3_vl_text") or text_cfg.get("model_type") == "qwen3_vl_text":
                family = "krea2"
                evidence.append("Qwen3-VL config and decoder tensor keys")
            elif embed == [151936, 2560] and any("q_norm.weight" in key for key in shapes):
                family = "krea2"
                evidence.append("Qwen3-VL-4B decoder embedding and query-normalization geometry")
            elif embed == [151936, 1024] and any("q_norm.weight" in key for key in shapes):
                family = "anima"
                evidence.append("Qwen3-0.6B decoder embedding and query-normalization geometry")
            else:
                evidence.append("Text decoder keys found; training family compatibility unknown")
        elif (
            not files
            and (root / "tokenizer_config.json").is_file()
            and any(
                (root / name).is_file()
                for name in ("tokenizer.json", "vocab.json", "spiece.model", "tokenizer.model")
            )
        ):
            classes = _tokenizer_assets(root, allowed, "tokenizer")
            kind = "tokenizer"
            evidence.append("Local tokenizer configuration and vocabulary assets")
            if "PixtralProcessor" in classes:
                family = "flux2"
                unsupported_reason = FLUX2_DEV_UNSUPPORTED_REASON
                evidence.append(
                    "Pixtral processor assets belong to the dev Mistral conditioning path, not Klein Qwen3"
                )
    if family:
        candidates = [family]
    if not family:
        warnings.append(
            "模型系列尚不能唯一确定，请确认兼容系列。"
            if candidates
            else "未识别模型系列；请根据模型来源确认，不按文件名猜测。"
        )
    if not kind:
        warnings.append("未识别组件类型，请手动确认。")
    if dtype == "mixed":
        warnings.append("矩阵权重包含多种精度，已标记为 mixed。")
    if not dtype and kind != "tokenizer":
        warnings.append("未识别浮点权重精度，保留未知。")
    if unsupported_reason:
        warnings.append(unsupported_reason)
    variant = None
    purpose = None
    if family == "krea2" and kind == "dit":
        from ypuddin.models.krea2.variants import verified_variant

        variant = verified_variant(files[0] if len(files) == 1 else path)
        purpose = ("inference" if variant == "turbo" else "training") if variant else None
        if not variant:
            warnings.append("Raw 与 Turbo 权重形状相同；请按模型发布说明确认用途，不能根据文件名自动判断。")
    return {
        "path": str(
            files[0]
            if path.is_dir()
            and kind in ("dit", "vae")
            and len(files) == 1
            and not {"sdxl", "flux", "flux2"}.intersection(candidates)
            else path
        ),
        "family": family,
        "family_candidates": candidates,
        "kind": kind,
        "dtype": dtype,
        "dtypes": dict(dtypes),
        "confidence": "high" if family and kind else "partial" if kind else "unknown",
        "evidence": evidence,
        "warnings": warnings,
        "files_inspected": len(files),
        "unsupported_reason": unsupported_reason,
        "variant": variant,
        "purpose": purpose,
    }
