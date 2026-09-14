"""Determinism settings preserve legacy checkpoints without hiding real changes."""

import hashlib
import json

import pytest
import torch

from ypuddin.config import TrainConfig, config_hash
from ypuddin.train import Trainer
from ypuddin.train.reproducibility import configure_reproducibility, validate_resume_reproducibility


@pytest.fixture(autouse=True)
def restore_torch_policy():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn
    torch.backends.cudnn.benchmark = benchmark


def test_deterministic_kernels_are_explicitly_enabled(tmp_path):
    assert TrainConfig().loop.deterministic is False
    cfg = TrainConfig.model_validate({
        "checkpoint": {"output_dir": str(tmp_path)}, "loop": {"deterministic": True},
    })
    trainer = Trainer(cfg, device="cpu")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    trainer._seed_all()
    expected = torch.rand(5)
    trainer._seed_all()
    assert torch.equal(torch.rand(5), expected)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_explicit_disable_clears_previous_process_policy():
    configure_reproducibility(True, torch.device("cpu"))
    configure_reproducibility(False, torch.device("cpu"))
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic


def test_dtk_strict_sdpa_uses_math_and_restores_previous_backend_choices(monkeypatch):
    from ypuddin.train import reproducibility

    monkeypatch.setattr(reproducibility, "current_profile", lambda: "linux-dtk")
    monkeypatch.setattr(torch.version, "hip", "5.7")
    cuda = torch.backends.cuda
    original = {name: getattr(cuda, f"{name}_sdp_enabled")()
                for name in ("flash", "mem_efficient", "math", "cudnn")}
    try:
        cuda.enable_flash_sdp(False)  # explicit caller choice must survive both transitions
        cuda.enable_math_sdp(False)
        before = {name: getattr(cuda, f"{name}_sdp_enabled")() for name in original}
        configure_reproducibility(True, torch.device("cuda"))
        configure_reproducibility(True, torch.device("cuda"))  # preparation may be repeated
        assert cuda.math_sdp_enabled()
        assert not cuda.flash_sdp_enabled()
        assert not cuda.mem_efficient_sdp_enabled()
        assert not cuda.cudnn_sdp_enabled()
        configure_reproducibility(False, torch.device("cuda"))
        assert {name: getattr(cuda, f"{name}_sdp_enabled")() for name in original} == before
    finally:
        reproducibility._configure_dtk_sdpa(False)
        for name, enabled in original.items():
            getattr(cuda, f"enable_{name}_sdp")(enabled)


def test_non_dtk_determinism_does_not_change_sdpa_selection(monkeypatch):
    from ypuddin.train import reproducibility

    monkeypatch.setattr(reproducibility, "current_profile", lambda: "linux-cuda")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
    configure_reproducibility(True, torch.device("cuda"))
    assert (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled()) == before


@pytest.mark.parametrize("hip,explicit,expected", [
    (None, None, ":4096:8"), (None, ":16:8", ":16:8"), ("5.7", None, None),
])
def test_cuda_workspace_policy_preserves_explicit_setting(monkeypatch, hip, explicit, expected):
    import os

    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    if explicit is not None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", explicit)
    configure_reproducibility(True, torch.device("cuda"))
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == expected


def test_legacy_false_hash_preserves_old_bytes_but_not_changed_parameters():
    cfg = TrainConfig.model_validate({"loop": {"deterministic": False}})
    old = cfg.to_dict()
    del old["loop"]["deterministic"]
    blob = json.dumps(old, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    old_hash = hashlib.blake2b(blob, digest_size=8).hexdigest()
    assert config_hash(cfg) == config_hash(old) == old_hash
    assert cfg.loop.deterministic is False  # fingerprinting never mutates the config
    changed = cfg.model_copy(deep=True)
    changed.optimizer.lr *= 2
    assert config_hash(changed) != old_hash
    changed = cfg.model_copy(deep=True)
    changed.loop.deterministic = True
    assert config_hash(changed) != old_hash


@pytest.mark.parametrize("enabled,saved", [(True, True), (False, False), (False, None)])
def test_resume_accepts_unchanged_policy_and_legacy_false(enabled, saved):
    validate_resume_reproducibility(enabled, saved)


@pytest.mark.parametrize("enabled,saved", [(False, True), (True, False), (True, None)])
def test_resume_rejects_changed_or_unknown_deterministic_policy(enabled, saved):
    with pytest.raises(ValueError, match="可复现训练"):
        validate_resume_reproducibility(enabled, saved)
