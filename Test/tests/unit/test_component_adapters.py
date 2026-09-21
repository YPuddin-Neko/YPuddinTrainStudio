"""Multi-component adapter files preserve algorithms, identities and load boundaries."""

import copy
import json

import pytest
import torch
from torch import nn

from ypuddin.adapters import inject, save_adapter_file
from ypuddin.adapters.components import TEXT_ADAPTER_PRESET, ComponentAdapterSet, inject_text_adapters
from ypuddin.config import AdapterConfig, TrainConfig
from ypuddin.config.training_rules import training_capabilities, training_errors
from ypuddin.server.xyz_worker import bind_checkpoint


class Encoder:
    def __init__(self, models):
        self.models = models

    def trainable_modules(self):
        return self.models


@pytest.mark.parametrize("algo", ["lora", "lokr", "loha", "full"])
@pytest.mark.parametrize("dora", [False, True])
def test_component_algorithms_train_export_load_and_scale(tmp_path, algo, dora):
    torch.manual_seed(4)
    originals = {k: nn.Sequential(nn.Linear(8, 8)) for k in ("backbone", "text_encoder", "text_encoder_2")}
    models = copy.deepcopy(originals)
    cfg = AdapterConfig(algo=algo, rank=2, alpha=2, dora=dora)
    for model in models.values():
        model.requires_grad_(False)
    adapters = ComponentAdapterSet(
        {
            "backbone": inject(models["backbone"], cfg, TEXT_ADAPTER_PRESET),
            **inject_text_adapters({k: v for k, v in models.items() if k != "backbone"}, cfg),
        }
    )
    initial = adapters.training_state_dict()
    optimizer = torch.optim.AdamW(adapters.param_groups(0.01, 0.001, {"text_encoder_2.": 0.02}))
    assert {g["lr"] for g in optimizer.param_groups if g["name"].startswith("text_encoder_2.")} == {0.02}
    x = torch.randn(3, 8)
    sum(model(x).square().mean() for model in models.values()).backward()
    optimizer.step()
    final = adapters.training_state_dict()
    for component in models:
        assert any(
            not torch.equal(value, final[key])
            for key, value in initial.items()
            if key.startswith(component + ".")
        )
    adapters.train(False)
    tensors, targets = adapters.export_state()
    assert all(key.startswith(("lora_unet_", "lora_te1_", "lora_te2_")) for key in targets)
    artifact = save_adapter_file(
        tmp_path / "adapter.safetensors",
        tensors,
        {"ypuddin.family": "sdxl", "ypuddin.targets": json.dumps(targets)},
        dtype="fp32",
    )
    restored = copy.deepcopy(originals)
    bindings = bind_checkpoint(
        restored["backbone"],
        artifact,
        "sdxl",
        "lora_unet",
        text=Encoder({k: v for k, v in restored.items() if k != "backbone"}),
    )
    for name in models:
        torch.testing.assert_close(restored[name](x), models[name](x), rtol=1e-5, atol=1e-6)
    for *_, wrapper in bindings:
        wrapper.multiplier = 0
    for name in models:
        torch.testing.assert_close(restored[name](x), originals[name](x), rtol=0, atol=0)
    warm_models = copy.deepcopy(originals)
    for model in warm_models.values():
        model.requires_grad_(False)
    warm = ComponentAdapterSet(
        {
            "backbone": inject(warm_models["backbone"], cfg, TEXT_ADAPTER_PRESET),
            **inject_text_adapters({k: v for k, v in warm_models.items() if k != "backbone"}, cfg),
        }
    )
    warm.load_state(tensors)
    warm.train(False)
    for name in models:
        torch.testing.assert_close(warm_models[name](x), models[name](x), rtol=1e-5, atol=1e-6)
    adapters.load_training_state(initial)
    assert all(torch.equal(v, initial[k]) for k, v in adapters.training_state_dict().items())


def test_bad_component_training_state_rejected_before_any_mutation():
    cfg = AdapterConfig(algo="lora", rank=2)
    models = {
        k: nn.Sequential(nn.Linear(4, 4)).requires_grad_(False) for k in ("text_encoder", "text_encoder_2")
    }
    adapters = ComponentAdapterSet(inject_text_adapters(models, cfg))
    initial = adapters.training_state_dict()
    malformed = {k: v + 1 for k, v in initial.items()}
    malformed[list(malformed)[-1]] = torch.ones(99)
    with pytest.raises(ValueError, match="component selection"):
        adapters.load_training_state(malformed)
    assert all(torch.equal(v, initial[k]) for k, v in adapters.training_state_dict().items())


def test_bad_text_target_does_not_partially_install_backbone(tmp_path):
    cfg = AdapterConfig(algo="lora", rank=2)
    backbone = nn.Sequential(nn.Linear(4, 4))
    adapter = inject(copy.deepcopy(backbone), cfg, TEXT_ADAPTER_PRESET)
    tensors, _ = adapter.export_state()
    tensors.update(
        {k.replace("lora_unet_0", "lora_te2_missing"): v.clone() for k, v in tensors.copy().items()}
    )
    path = save_adapter_file(tmp_path / "invalid.safetensors", tensors, {}, dtype="fp32")
    with pytest.raises(ValueError, match="target is absent"):
        bind_checkpoint(
            backbone,
            path,
            "sdxl",
            "lora_unet",
            text=Encoder({"text_encoder_2": nn.Sequential(nn.Linear(4, 4))}),
        )
    assert isinstance(backbone[0], nn.Linear)


@pytest.mark.parametrize("family", ["anima", "krea2", "sdxl", "flux2"])
def test_capability_and_memory_boundaries(family):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": family},
            "training": {"train_backbone": False, "train_text_encoder": True},
            "memory": {"offload_text_encoder": False},
        }
    )
    assert training_capabilities(family)["adapter_text_encoder"] is True
    assert not training_errors(cfg)
    cfg.dataset.text_encoding = "cached"
    assert any(e["loc"] == "dataset.text_encoding" for e in training_errors(cfg))
    cfg.dataset.text_encoding = "online"
    cfg.memory.offload_text_encoder = True
    assert any(e["loc"] == "memory.offload_text_encoder" for e in training_errors(cfg))
    cfg.memory.offload_text_encoder = False
    cfg.loop.gpu_count = 2
    cfg.loop.distributed_strategy = "fsdp"
    assert any(e["loc"] == "training.train_text_encoder" for e in training_errors(cfg))


@pytest.mark.parametrize("family", ["anima", "sdxl", "toy"])
def test_default_clip_length_preserves_legacy_model_identity(family, monkeypatch):
    from types import SimpleNamespace

    from ypuddin.config import ModelConfig
    from ypuddin.train import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(model=ModelConfig(family=family))
    trainer.family = SimpleNamespace(
        spec=SimpleNamespace(
            name=family, objective="ddpm" if family == "sdxl" else "flow", architecture=family
        )
    )
    trainer.loaded = SimpleNamespace(
        backbone=nn.Linear(2, 2),
        dtype=torch.float32,
        extra={},
        latent=SimpleNamespace(fingerprint="latent"),
        text=SimpleNamespace(fingerprint="text"),
    )
    current = trainer._model_identity()
    model_dump = ModelConfig.model_dump
    with monkeypatch.context() as patch:

        def historical_dump(self, *args, **kwargs):
            result = model_dump(self, *args, **kwargs)
            result.pop("sdxl_max_token_length", None)
            return result

        patch.setattr(ModelConfig, "model_dump", historical_dump)
        assert trainer._model_identity() == current
    trainer.cfg.model.sdxl_max_token_length = 150
    assert (trainer._model_identity() != current) == (family == "sdxl")
