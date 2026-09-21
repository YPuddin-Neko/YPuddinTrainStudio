"""Versioned numeric policy is checked before overwriting or loading state."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from ypuddin.config import TrainConfig
from ypuddin.train import Trainer
from ypuddin.train.reproducibility import capture_compute_runtime, validate_compute_runtime


def _trainer(tmp_path, monkeypatch, *, resume=True, precision="fp16"):
    import ypuddin.train.trainer as module

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "anima", "attention": "xformers"},
            "training": {"mode": "full", "train_backbone": True},
            "loop": {"deterministic": True, "mixed_precision": precision},
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
    from ypuddin.train.scheduler_contract import scheduler_recipe

    checkpoint = tmp_path / "state-4"
    checkpoint.mkdir(exist_ok=True)
    cfg = TrainConfig.model_validate({"training": {"mode": "full"}})
    extra = {**extra, "scheduler_contract": scheduler_recipe(cfg, 8)}
    (checkpoint / "state.json").write_text(
        json.dumps(
            {"format": 3, "training_kind": "full-model", "progress": {"total_steps": 8, "extra": extra}}
        )
    )
    return extra


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_trainer_resolves_policy_before_hash_without_mutating_requested_config(
    tmp_path, monkeypatch, precision
):
    from ypuddin.config import config_hash

    requested, trainer = _trainer(tmp_path, monkeypatch, resume=False, precision=precision)
    assert requested.loop.mixed_precision == precision
    assert requested.memory.allow_tf32
    assert requested.model.attention == "xformers"
    assert trainer.cfg.loop.mixed_precision == ("bf16" if precision == "bf16" else "no")
    assert not trainer.cfg.memory.allow_tf32
    assert trainer.cfg.model.attention == "sdpa"
    assert trainer.compute_dtype == (torch.bfloat16 if precision == "bf16" else torch.float32)
    assert trainer.config_hash == config_hash(trainer.cfg)
    assert trainer.compute_policy["id"] == (
        "dtk-anima-bf16-linear-fp32-compute-v1" if precision == "bf16" else "dtk-full-fp32-math-v1"
    )


@pytest.mark.parametrize("changed", [None, "id", "mixed_precision", "allow_tf32", "attention"])
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_old_or_different_policy_rejected_before_model_load_and_config_write(
    tmp_path, monkeypatch, changed, precision
):
    import ypuddin.train.trainer as module

    _, trainer = _trainer(tmp_path, monkeypatch, precision=precision)
    saved = copy.deepcopy(trainer.compute_policy) if changed is not None else None
    if changed is not None:
        saved[changed] = True if changed == "allow_tf32" else "old"
    _write_metadata(tmp_path, {"deterministic": True, "compute_policy": saved})
    original = b"original recipe must survive\n"
    (tmp_path / "config.toml").write_bytes(original)
    monkeypatch.setattr(module, "get_family", lambda _: SimpleNamespace())
    monkeypatch.setattr(trainer, "_check_capabilities", lambda: None)
    monkeypatch.setattr(trainer, "_seed_all", lambda: pytest.fail("GPU setup began before preflight"))
    monkeypatch.setattr(
        module,
        "read_resume_scheduler_contract",
        lambda *_, **__: pytest.fail("scheduler checks ran before rejecting the compute policy"),
    )
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
    extra = _write_metadata(tmp_path, extra)
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


@pytest.mark.parametrize(
    "mode,gpu_count,strategy",
    [
        ("full", 1, "ddp"),
        ("adapter", 1, "ddp"),
        ("adapter", 2, "ddp"),
        ("full", 2, "fsdp"),
    ],
)
def test_anima_product_installs_new_linear_recipe_and_checks_real_modules(
    tmp_path, monkeypatch, mode, gpu_count, strategy
):
    from torch import nn

    from ypuddin.adapters.frozen import FrozenLinear
    from ypuddin.adapters.linear import AdaptedLinear
    from ypuddin.adapters.lokr import LoKr
    from ypuddin.train.linear_backward import validate_linear_backward_installation

    cfg, _ = _trainer(tmp_path, monkeypatch, resume=False, precision="bf16")
    cfg.training.mode = mode
    cfg.loop.gpu_count, cfg.loop.distributed_strategy = gpu_count, strategy
    trainer = Trainer(cfg, device="cuda:0")  # Device metadata only.
    layer = nn.Linear(4, 4)
    if mode == "adapter":
        layer = AdaptedLinear(FrozenLinear.from_linear(layer), LoKr(4, 4, rank=2), mode="auto")
    trainer.loaded = SimpleNamespace(backbone=nn.Sequential(layer))
    with pytest.raises(ValueError, match="未完整安装"):
        trainer._validate_training_compute_policy()
    if strategy == "fsdp":
        with pytest.raises(ValueError, match="FSDP"):
            trainer._place_training_model()
        trainer._install_anima_compute_operators()
    else:
        monkeypatch.setattr(trainer.loaded.backbone, "to", lambda _: trainer.loaded.backbone)
        trainer._place_training_model()
    try:
        trainer._validate_training_compute_policy()
        with pytest.raises(ValueError, match="重复安装"):
            trainer._install_anima_compute_operators()
        # The old native-forward marker must never validate this new recipe.
        with pytest.raises(ValueError, match="forward was replaced"):
            validate_linear_backward_installation(trainer.loaded.backbone, trainer._linear_backward_counts)
        if mode == "adapter":
            layer.mode = "merged"
            with pytest.raises(ValueError, match="LoKr"):
                trainer._validate_training_compute_policy()
        else:
            trainer.cfg.memory.activation_checkpointing = "unsloth"
            with pytest.raises(ValueError, match="Unsloth"):
                trainer._validate_training_compute_policy()
    finally:
        trainer._linear_backward_restore()


@pytest.mark.parametrize("mode,strategy", [("full", "fsdp"), ("adapter", "ddp")])
@pytest.mark.parametrize(
    "saved_id", ["dtk-anima-bf16-linear-fp32-compute-v1", "diagnostic-anima-dual", "dtk-full-fp32-math-v1"]
)
def test_anima_distributed_old_policy_rejected_before_loading_or_writing(
    tmp_path, monkeypatch, mode, strategy, saved_id
):
    import ypuddin.train.trainer as module

    cfg, _ = _trainer(tmp_path, monkeypatch, precision="bf16")
    cfg.training.mode = mode
    cfg.loop.gpu_count, cfg.loop.distributed_strategy = 2, strategy
    trainer = Trainer(cfg, device="cuda")
    _write_metadata(
        tmp_path, {"deterministic": True, "compute_policy": trainer.compute_policy | {"id": saved_id}}
    )
    original = b"preserve original config\n"
    (tmp_path / "config.toml").write_bytes(original)
    monkeypatch.setattr(module, "get_family", lambda _: SimpleNamespace())
    monkeypatch.setattr(trainer, "_check_capabilities", lambda: None)
    monkeypatch.setattr(
        trainer, "_seed_all", lambda: pytest.fail("device setup started before policy rejection")
    )
    with pytest.raises(ValueError, match="计算"):
        trainer.prepare_data()
    assert (tmp_path / "config.toml").read_bytes() == original
