"""Full component training, independent of LyCORIS linear-layer delta adapters.

A full-model artifact contains native component weights and a portable training-state
bundle. Frozen assets (for example the VAE) remain explicit references in config.toml.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config import TrainConfig, write_config


class FullTrainingSet:
    """The trainer's weight/optimizer surface for real backbone and encoder parameters."""

    layers: dict = {}  # old adapter-only compatibility checks never interpret full weights as adapters

    def __init__(self, modules: dict[str, nn.Module]):
        if not modules:
            raise ValueError("full training requires at least one selected component")
        self.modules = modules
        for module in modules.values():
            if any(isinstance(layer, FrozenLinear) for layer in module.modules()):
                raise ValueError(
                    "Full fine-tuning requires unquantized base weights; this checkpoint contains frozen FP8 layers"
                )
            module.to(dtype=torch.float32).requires_grad_(True)
        self.rebind_parameters()
        if not self._names:
            raise ValueError("selected full-training components have no parameters")

    def rebind_parameters(self) -> None:
        """Refresh references after distributed placement replaces native parameters."""
        self._names = {
            f"{component}.{name}": parameter
            for component, module in self.modules.items()
            for name, parameter in module.named_parameters()
        }

    def parameters(self) -> list[nn.Parameter]:
        return list({id(parameter): parameter for parameter in self._names.values()}.values())

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def param_groups(self, base_lr: float, weight_decay: float, group_lr=None) -> list[dict[str, Any]]:
        buckets = {}
        seen = set()
        for name, parameter in self._names.items():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            component = name.split(".", 1)[0]
            lr = next((value for key, value in (group_lr or {}).items() if key in name), base_lr)
            buckets.setdefault((component, lr), []).append(parameter)
        return [
            {"name": component, "params": parameters, "lr": lr, "weight_decay": weight_decay}
            for (component, lr), parameters in sorted(buckets.items())
        ]

    def train(self, mode=True):
        for module in self.modules.values():
            module.train(mode)

    @staticmethod
    def _native_key(key: str) -> str:
        return ".".join(part for part in key.split(".") if part != "_orig_mod")

    def _state_keys(self):
        # torch.compile adds wrappers to the live tree. Native exports and resumes
        # retain the uncompiled module names regardless of the execution path.
        keys = {}
        for component, module in self.modules.items():
            for key in module.state_dict():
                native = f"{component}.{self._native_key(key)}"
                if native in keys:
                    raise ValueError(f"duplicate native training state key: {native}")
                keys[native] = (component, key)
        return keys

    def training_state_dict(self) -> dict[str, Tensor]:
        states = {component: module.state_dict() for component, module in self.modules.items()}
        # Snapshot directly to host memory; cloning a full DiT on CUDA would
        # temporarily double VRAM at every save/evaluation boundary.
        return {
            native: states[component][key].detach().to(device="cpu", copy=True)
            for native, (component, key) in self._state_keys().items()
        }

    def export_state(self):
        return self.training_state_dict(), {"components": sorted(self.modules), "format": "full-model-v1"}

    def load_training_state(self, tensors):
        expected = self._state_keys()
        if tensors.keys() != expected.keys():
            raise ValueError(
                "checkpoint full-training components or parameter structure differ from the current selection"
            )
        for component, module in self.modules.items():
            module.load_state_dict(
                {key: tensors[native] for native, (owner, key) in expected.items() if owner == component},
                strict=True,
            )

    def load_state(self, tensors, *, strict=True):
        self.load_training_state(tensors)
        return []

    def summary(self):
        return {
            "layers": 0,
            "trainable_params": self.num_params(),
            "by_algo": {"full-model": len(self.modules)},
            "components": sorted(self.modules),
        }

    def load_weights(self, path: str, family: str):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "ypuddin-full-model-v1" or manifest.get("family") != family:
            raise ValueError("full-model artifact format or model family differs from this run")
        if set(manifest.get("components", {})) != set(self.modules):
            raise ValueError("full-model artifact component selection differs from this run")
        tensors = {}
        for component in self.modules:
            relative = Path(manifest["components"][component])
            source = (root / relative).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("full-model artifact contains an invalid component path")
            tensors.update({f"{component}.{key}": value for key, value in load_file(str(source)).items()})
        self.load_training_state(tensors)


def _cpu_tensors(tensors, dtype):
    return {
        key: value.detach()
        .to(device="cpu", dtype=dtype if value.is_floating_point() else value.dtype)
        .contiguous()
        for key, value in tensors.items()
    }


def save_model_artifact(
    path: Path, training: FullTrainingSet, cfg: TrainConfig, loaded, *, tensors=None
) -> Path:
    """Atomically export actual model components plus an explicit configuration for reuse."""
    state = training.training_state_dict() if tensors is None else tensors
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.checkpoint.save_dtype]
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        config = cfg.model_copy(deep=True)
        if cfg.training.train_backbone or cfg.memory.base_precision == "fp32":
            config.model.dtype = "fp32"
        config.training.resume_weights = None
        config.checkpoint.resume = None
        component_outputs = {}
        family = cfg.model.family
        for component, module in training.modules.items():
            prefix = component + "."
            values = _cpu_tensors(
                {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}, dtype
            )
            target = temporary / component
            target.mkdir()
            module_config = getattr(module, "config", None)
            if component == "backbone":
                if family in {"sdxl", "flux2"}:
                    module.save_config(target)
                    save_file(values, str(target / "diffusion_pytorch_model.safetensors"))
                    config.model.dit_path = str(path / component)
                else:
                    save_file(values, str(target / "model.safetensors"))
                    config.model.dit_path = str(path / component / "model.safetensors")
            else:
                if hasattr(module_config, "save_pretrained"):
                    module_config.save_pretrained(target)
                save_file(values, str(target / "model.safetensors"))
                # Qwen decoder exports use the supported single-file path, preserving the
                # original tokenizer reference. Klein needs the original full HF wrapper.
                is_decoder = family in {"anima", "krea2"}
                setattr(
                    config.model,
                    "text_encoder_2_path" if component == "text_encoder_2" else "text_encoder_path",
                    str(path / component / "model.safetensors") if is_decoder else str(path / component),
                )
            component_outputs[component] = f"{component}/" + (
                "diffusion_pytorch_model.safetensors"
                if component == "backbone" and family in {"sdxl", "flux2"}
                else "model.safetensors"
            )
        # Resolve components embedded in the original SDXL checkpoint before replacing UNet.
        if family == "sdxl":
            for i, source in enumerate(loaded.text.paths):
                key = "text_encoder_path" if i == 0 else "text_encoder_2_path"
                if ("text_encoder" if i == 0 else "text_encoder_2") not in training.modules:
                    setattr(config.model, key, str(source))
            config.model.vae_path = str(loaded.latent.path)
            tokenizers = loaded.text.tokenizer_paths
            if tokenizers[0].parent == tokenizers[1].parent:
                config.model.tokenizer_path = str(tokenizers[0].parent)
        elif family == "flux2":
            config.model.flux2_variant = loaded.extra["variant"]
            config.model.vae_path = str(loaded.latent.path)
            if "text_encoder" not in training.modules:
                config.model.text_encoder_path = str(loaded.text.path)
            config.model.tokenizer_path = str(loaded.text.tokenizer_path)
        if family in {"anima", "krea2"} and "text_encoder" in training.modules:
            # tokenizer_path means old T5 in Anima, so retain that setting; the Qwen
            # tokenizer sits next to decoder weights and is copied as small text assets.
            tokenizer = loaded.text.tokenizer
            tokenizer.save_pretrained(temporary / "text_encoder")
        write_config(config, temporary / "config.toml")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "ypuddin-full-model-v1",
                    "family": family,
                    "components": component_outputs,
                    "config": "config.toml",
                    "frozen_assets": "referenced_by_config",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if path.exists():
            raise FileExistsError(f"model artifact already exists: {path}")
        temporary.rename(path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return path
