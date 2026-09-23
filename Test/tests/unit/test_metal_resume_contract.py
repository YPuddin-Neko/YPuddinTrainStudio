"""An Apple attention extension cannot silently alter an exact-resume recipe."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import resolve_training_compute_config
from ypuddin.train.metal_compute import resolve_metal_attention_runtime, validate_metal_attention_resume
from ypuddin.train.trainer import Trainer

RUNTIME = {
    "implementation": "mtlattn-fp32-v1",
    "torch": "2.13.0",
    "mtlattn": "0.4.1",
    "platform": "Darwin",
    "machine": "arm64",
    "macos": "15.7.9",
    "python": "3.12.12",
}


@pytest.mark.parametrize("device,profile", [("cuda", "linux-dtk"), ("mps", "macos-mps"), ("cpu", "macos-cpu")])
def test_apple_choice_is_not_rewritten_into_a_dtk_sdpa_policy(device, profile):
    cfg = TrainConfig.model_validate({
        "model": {"family": "anima", "attention": "metal_flash", "dtype": "bf16"},
        "training": {"mode": "full"},
        "loop": {"deterministic": True, "mixed_precision": "fp16"},
    })
    effective, policy = resolve_training_compute_config(cfg, device, profile)
    assert effective.to_dict() == cfg.to_dict()
    assert effective.model.attention == "metal_flash" and policy is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_wrong_device_is_rejected_before_importing_the_extension(device):
    with pytest.raises(ValueError, match="Apple MPS"):
        resolve_metal_attention_runtime("metal_flash", device)
    assert resolve_metal_attention_runtime("sdpa", device) is None


@pytest.mark.parametrize("field", list(RUNTIME))
def test_resume_rejects_each_numerical_environment_change(field):
    changed = RUNTIME | {field: "changed"}
    with pytest.raises(ValueError, match="不能严格续训"):
        validate_metal_attention_resume(changed, RUNTIME)
    validate_metal_attention_resume(dict(RUNTIME), RUNTIME)


@pytest.mark.parametrize("current,saved", [(RUNTIME, None), (None, RUNTIME), (RUNTIME, {}), (RUNTIME, "invalid")])
def test_resume_rejects_switching_backend_or_missing_identity(current, saved):
    with pytest.raises(ValueError, match="不能严格续训"):
        validate_metal_attention_resume(current, saved)
    validate_metal_attention_resume(None, None)


def test_trainer_rejects_metal_change_before_loading_models_or_reading_tensor_states(tmp_path, monkeypatch):
    from ypuddin.models import metal_attention

    monkeypatch.setattr(metal_attention, "require_metal_flash", lambda device: dict(RUNTIME))
    checkpoint = tmp_path / "state-2"
    checkpoint.mkdir()
    metadata = {"progress": {"extra": {"deterministic": True, "metal_attention_runtime": RUNTIME | {"mtlattn": "old"}}}}
    (checkpoint / "state.json").write_text(json.dumps(metadata))
    cfg = TrainConfig.model_validate({
        "model": {"family": "anima", "attention": "metal_flash"},
        "checkpoint": {"output_dir": str(tmp_path / "run"), "resume": str(checkpoint)},
        "loop": {"deterministic": True},
    })
    trainer = Trainer(cfg, device="mps")
    with pytest.raises(ValueError, match="不能严格续训"):
        trainer._read_resume_compute_metadata()
    assert not (tmp_path / "run" / "config.toml").exists()
    assert json.loads((checkpoint / "state.json").read_text()) == metadata


def test_model_identity_adds_runtime_only_for_metal_checkpoints():
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig.model_validate({"model": {"family": "anima", "attention": "sdpa"}})
    trainer.family = SimpleNamespace(spec=SimpleNamespace(name="anima", objective="flow", architecture="anima"))
    trainer.loaded = SimpleNamespace(
        backbone=torch.nn.Linear(2, 2), dtype=torch.float32, extra={},
        latent=SimpleNamespace(fingerprint="latent"), text=SimpleNamespace(fingerprint="text"),
    )
    trainer.metal_attention_runtime = None
    native = trainer._model_identity()
    trainer.metal_attention_runtime = deepcopy(RUNTIME)
    assert trainer._model_identity() == native
    trainer.cfg.model.attention = "metal_flash"
    metal = trainer._model_identity()
    assert metal != native
    trainer.metal_attention_runtime["macos"] = "changed"
    assert trainer._model_identity() != metal


def test_unconnected_family_rejects_metal_before_loading(tmp_path):
    from ypuddin.models import get_family
    from ypuddin.train.plan import plan

    cfg = TrainConfig.model_validate({
        "model": {"family": "toy", "attention": "metal_flash"},
        "checkpoint": {"output_dir": str(tmp_path / "run")},
    })
    trainer = Trainer(cfg, device="mps")
    trainer.family = get_family("toy")
    with pytest.raises(ValueError, match="not supported by this model family"):
        trainer._check_capabilities()
    result = plan(cfg, device=None)
    assert any(error["loc"] == "model.attention" for error in result["errors"])
