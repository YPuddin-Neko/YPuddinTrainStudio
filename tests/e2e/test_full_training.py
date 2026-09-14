"""Actual tiny training: component changes, samples, native export and exact resume."""

import json
from pathlib import Path

import pytest
import torch

from tests.e2e.test_anima_pipeline import tiny_models as anima_models  # noqa: F401
from tests.e2e.test_anima_pipeline import tiny_vae_loader as tiny_qwen_vae  # noqa: F401
from tests.e2e.test_krea2_pipeline import tiny_models as krea_models  # noqa: F401
from ypuddin.config import TrainConfig
from ypuddin.train import Trainer


@pytest.mark.parametrize("backbone,text", [(True, False), (False, True), (True, True)])
def test_full_components_export_and_resume(image_dataset, tmp_path, backbone, text):
    cfg = TrainConfig.model_validate(
        {
            "training": {"mode": "full", "train_backbone": backbone, "train_text_encoder": text},
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolutions": [64],
                "bucket_step": 16,
                "num_workers": 0,
                "batch_size": 2,
                "text_encoding": "online",
            },
            "memory": {"offload_text_encoder": False},
            "loop": {"max_steps": 4, "mixed_precision": "no", "seed": 7},
            "optimizer": {"lr": 0.001},
            "checkpoint": {
                "output_dir": str(tmp_path / "reference"),
                "name": "full",
                "save_every_epochs": None,
                "save_state_every_steps": 2,
                "save_dtype": "fp32",
            },
            "sampling": {
                "enabled": True,
                "every_steps": 4,
                "width": 64,
                "height": 64,
                "steps": 2,
                "prompts": [{"prompt": "a test", "width": 64, "height": 64, "steps": 2}],
            },
        }
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    assert set(trainer.adapters.modules) == ({"backbone"} if backbone else set()) | (
        {"text_encoder"} if text else set()
    )
    events = [json.loads(line) for line in (tmp_path / "reference/events.jsonl").read_text().splitlines()]
    assert any(e["type"] == "sample.saved" for e in events)
    artifact = tmp_path / "reference/full-final.model"
    assert json.loads((artifact / "manifest.json").read_text())["format"] == "ypuddin-full-model-v1"
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed_cfg.checkpoint.resume = str(tmp_path / "reference/state-2")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    expected = trainer.adapters.training_state_dict()
    actual = resumed.adapters.training_state_dict()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.usefixtures("tiny_qwen_vae")
@pytest.mark.parametrize("family", ["anima", "krea2"])
@pytest.mark.parametrize("backbone,text", [(True, False), (False, True), (True, True)])
def test_native_qwen_full_training_reload_resume(request, image_dataset, tmp_path, family, backbone, text):
    from ypuddin.config import load_config

    assets = request.getfixturevalue("anima_models" if family == "anima" else "krea_models")
    cfg = TrainConfig.model_validate(
        {
            "training": {"mode": "full", "train_backbone": backbone, "train_text_encoder": text},
            "model": {
                "family": family,
                "dtype": "fp32",
                "dit_path": str(assets["dit"]),
                "text_encoder_path": str(assets["qwen"] if family == "anima" else assets["qwen_dir"]),
                "vae_path": str(assets["vae"]),
            },
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolutions": [64],
                "bucket_step": 16,
                "batch_size": 1,
                "num_workers": 0,
                "text_encoding": "online" if text else "cached",
            },
            "memory": {"activation_checkpointing": "block"},
            "loop": {"max_steps": 2, "mixed_precision": "no", "seed": 7},
            "optimizer": {"lr": 0.001},
            "checkpoint": {
                "output_dir": str(tmp_path / "reference"),
                "name": "full",
                "save_dtype": "fp32",
                "save_every_epochs": None,
                "save_state_every_steps": 1,
            },
            "sampling": {
                "enabled": True,
                "every_epochs": None,
                "every_steps": 2,
                "width": 64,
                "height": 64,
                "prompts": [{"prompt": "test image", "width": 64, "height": 64, "steps": 2}],
            },
        }
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    original_backbone = {k: v.detach().clone() for k, v in trainer.loaded.backbone.state_dict().items()}
    from ypuddin.train.plan import plan

    planned = plan(cfg, device="cpu")
    assert planned["ok"], planned["errors"]
    assert planned["params"]["trainable"] == trainer.adapters.num_params()
    original = trainer.adapters.training_state_dict()
    assert trainer.run() == "finished"
    final = trainer.adapters.training_state_dict()
    for component in trainer.adapters.modules:
        assert any(not torch.equal(original[k], final[k]) for k in original if k.startswith(component + "."))
    if not backbone:
        for k, v in trainer.loaded.backbone.state_dict().items():
            torch.testing.assert_close(v, original_backbone[k], rtol=0, atol=0)
    artifact = tmp_path / "reference/full-final.model"
    reloaded_cfg = load_config(artifact / "config.toml")
    reloaded_cfg.checkpoint.output_dir = str(tmp_path / "reloaded")
    reloaded_cfg.loop.max_steps = 1
    reloaded = Trainer(reloaded_cfg, device="cpu")
    reloaded.prepare()
    for k, v in reloaded.adapters.training_state_dict().items():
        torch.testing.assert_close(v, final[k], rtol=0, atol=0)
    assert reloaded.run() == "finished"
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed_cfg.checkpoint.resume = str(tmp_path / "reference/state-1")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    for k, v in resumed.adapters.training_state_dict().items():
        torch.testing.assert_close(v, final[k], rtol=0, atol=0)


def test_full_schedulefree_ema_warm_start_and_mode_boundaries(image_dataset, tmp_path):
    import shutil

    from ypuddin.config import load_config
    from ypuddin.train.state import load_checkpoint

    pytest.importorskip("prodigyplus")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "training": {"mode": "full", "train_backbone": True, "train_text_encoder": True},
            "dataset": {"sources": [{"path": str(image_dataset)}], "resolutions": [32], "num_workers": 0},
            "loop": {"max_steps": 3, "mixed_precision": "no", "ema": True},
            "optimizer": {"type": "prodigy_plus_sf"},
            "checkpoint": {
                "output_dir": str(tmp_path / "source"),
                "name": "full",
                "save_every_epochs": None,
                "save_state_every_steps": 1,
                "save_dtype": "fp32",
            },
            "sampling": {
                "enabled": True,
                "every_epochs": None,
                "every_steps": 2,
                "width": 32,
                "height": 32,
                "prompts": [{"prompt": "test", "width": 32, "height": 32, "steps": 2}],
            },
        }
    )
    source = Trainer(cfg, device="cpu")
    assert source.run() == "finished"
    state = source.run_dir / "state-1"
    assert load_checkpoint(state)["training_kind"] == "full-model"
    assert (state / "model.safetensors").is_file()
    assert not (state / "training.safetensors").exists()  # one raw full model, no redundant second copy
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.resume = str(state)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    for key, tensor in source.adapters.training_state_dict().items():
        torch.testing.assert_close(resumed.adapters.training_state_dict()[key], tensor, rtol=0, atol=0)
    for key, tensor in source.ema.items():
        torch.testing.assert_close(resumed.ema[key], tensor, rtol=0, atol=0)
    artifact = source.run_dir / "full-final.model"
    relocated = tmp_path / "moved.model"
    shutil.copytree(artifact, relocated)
    moved_cfg = load_config(relocated / "config.toml")
    assert Path(moved_cfg.model.dit_path).is_relative_to(relocated)
    assert Path(moved_cfg.model.text_encoder_path).is_relative_to(relocated)
    warm_cfg = cfg.model_copy(deep=True)
    warm_cfg.training.resume_weights = str(relocated)
    warm_cfg.checkpoint.output_dir = str(tmp_path / "warm")
    warm = Trainer(warm_cfg, device="cpu")
    warm.prepare()
    assert warm.progress.step == 0
    assert not warm.optimizer.state  # warm start does not reuse optimizer momentum
    warm._close_logs()
    bad_cfg = cfg.model_copy(deep=True)
    bad_cfg.training.mode = "adapter"
    bad_cfg.training.train_text_encoder = False
    bad_cfg.checkpoint.resume = str(state)
    bad_cfg.checkpoint.output_dir = str(tmp_path / "bad-mode")
    with pytest.raises(ValueError, match="training mode differs"):
        Trainer(bad_cfg, device="cpu").prepare()
