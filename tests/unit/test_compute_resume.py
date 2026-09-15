"""Versioned numeric policy is checked before overwriting or loading state."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from ypuddin.config import TrainConfig
from ypuddin.train import Trainer
from ypuddin.train.reproducibility import capture_compute_runtime, validate_compute_runtime


def _trainer(tmp_path, monkeypatch, *, resume=True):
    import ypuddin.train.trainer as module

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "anima", "attention": "xformers"},
            "training": {"mode": "full", "train_backbone": True},
            "loop": {"deterministic": True, "mixed_precision": "bf16"},
            "memory": {"allow_tf32": True},
            "checkpoint": {
                "output_dir": str(tmp_path),
                "resume": str(tmp_path / "state-4") if resume else None,
            },
        }
    )
    trainer = Trainer(cfg, device="cuda:0")
    return cfg, trainer


def _write_metadata(tmp_path, extra):
    checkpoint = tmp_path / "state-4"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "state.json").write_text(json.dumps({"progress": {"extra": extra}}))


def test_trainer_resolves_policy_before_hash_without_mutating_requested_config(tmp_path, monkeypatch):
    from ypuddin.config import config_hash

    requested, trainer = _trainer(tmp_path, monkeypatch, resume=False)
    assert requested.loop.mixed_precision == "bf16"
    assert requested.memory.allow_tf32
    assert requested.model.attention == "xformers"
    assert trainer.cfg.loop.mixed_precision == "no"
    assert not trainer.cfg.memory.allow_tf32
    assert trainer.cfg.model.attention == "sdpa"
    assert trainer.compute_dtype == torch.float32
    assert trainer.config_hash == config_hash(trainer.cfg)
    assert trainer.compute_policy["id"] == "dtk-full-fp32-math-v1"


@pytest.mark.parametrize("changed", [None, "id", "mixed_precision", "allow_tf32", "attention"])
def test_old_or_different_policy_rejected_before_model_load_and_config_write(tmp_path, monkeypatch, changed):
    import ypuddin.train.trainer as module

    _, trainer = _trainer(tmp_path, monkeypatch)
    saved = copy.deepcopy(trainer.compute_policy) if changed is not None else None
    if changed is not None:
        saved[changed] = True if changed == "allow_tf32" else "old"
    _write_metadata(tmp_path, {"deterministic": True, "compute_policy": saved})
    original = b"original recipe must survive\n"
    (tmp_path / "config.toml").write_bytes(original)
    monkeypatch.setattr(module, "get_family", lambda _: SimpleNamespace())
    monkeypatch.setattr(trainer, "_check_capabilities", lambda: None)
    monkeypatch.setattr(trainer, "_seed_all", lambda: pytest.fail("GPU setup began before preflight"))
    with pytest.raises(ValueError, match="计算|旧"):
        trainer.prepare_data()
    assert (tmp_path / "config.toml").read_bytes() == original


def test_unchanged_compute_metadata_preflight_is_read_only(tmp_path, monkeypatch):
    _, trainer = _trainer(tmp_path, monkeypatch)
    extra = {
        "deterministic": True,
        "compute_policy": trainer.compute_policy,
        "compute_runtime": {"torch": "v"},
    }
    _write_metadata(tmp_path, extra)
    before = (tmp_path / "state-4" / "state.json").read_bytes()
    assert trainer._read_resume_compute_metadata() == extra
    assert (tmp_path / "state-4" / "state.json").read_bytes() == before


@pytest.mark.parametrize("saved", [None, {"torch": "previous"}])
def test_runtime_change_rejected_before_output_config_overwrite(tmp_path, monkeypatch, saved):
    import ypuddin.train.trainer as module

    _, trainer = _trainer(tmp_path, monkeypatch)
    _write_metadata(
        tmp_path, {"deterministic": True, "compute_policy": trainer.compute_policy, "compute_runtime": saved}
    )
    (tmp_path / "config.toml").write_text("original")
    monkeypatch.setattr(module, "get_family", lambda _: SimpleNamespace())
    monkeypatch.setattr(trainer, "_check_capabilities", lambda: None)
    monkeypatch.setattr(trainer, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(trainer, "_seed_all", lambda: None)
    monkeypatch.setattr(
        module, "TrainingLogs", lambda *_: pytest.fail("training logs started before validation")
    )
    monkeypatch.setattr(module, "capture_compute_runtime", lambda _: {"torch": "current"})
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        with pytest.raises(ValueError, match="计算环境"):
            trainer.prepare_data()
        assert (tmp_path / "config.toml").read_text() == "original"
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def test_runtime_fingerprint_allows_another_logical_gpu_but_rejects_version_change():
    current = capture_compute_runtime(torch.device("cuda:0"))
    assert current == capture_compute_runtime(torch.device("cuda:1"))
    validate_compute_runtime(current, copy.deepcopy(current))
    saved = {**current, "torch": "another-version"}
    with pytest.raises(ValueError, match="计算环境"):
        validate_compute_runtime(current, saved)
    with pytest.raises(ValueError, match="计算环境"):
        validate_compute_runtime(None, None)


@pytest.mark.parametrize(
    "key",
    [
        "HIPBLASLT_WORKSPACE_SIZE",
        "CUBLASLT_WORKSPACE_SIZE",
        "TORCH_BLAS_PREFER_HIPBLASLT",
    ],
)
def test_blas_runtime_environment_change_is_part_of_resume_contract(monkeypatch, key):
    monkeypatch.delenv(key, raising=False)
    saved = capture_compute_runtime(torch.device("cuda"))
    monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError, match="计算环境"):
        validate_compute_runtime(capture_compute_runtime(torch.device("cuda")), saved)


def test_active_policy_drift_fails_before_autocast(tmp_path, monkeypatch):
    import ypuddin.train.trainer as module

    _, trainer = _trainer(tmp_path, monkeypatch, resume=False)
    trainer.compute_runtime = {"torch": "current"}
    monkeypatch.setattr(module, "capture_compute_runtime", lambda _: trainer.compute_runtime)
    trainer.cfg.loop.mixed_precision = "bf16"
    with pytest.raises(ValueError, match="运行中改变"):
        trainer._autocast()
