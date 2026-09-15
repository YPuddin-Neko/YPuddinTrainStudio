"""Per-device FSDP capacity, optimizer layout and admission-facing estimates."""

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from ypuddin.config import TrainConfig
from ypuddin.models import get_family
from ypuddin.train.plan import _fsdp_memory, plan


def _config(path=None, *, optimizer="adafactor", args=None, strategy="fsdp", count=2):
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "training": {"mode": "full"},
            "dataset": {
                "sources": [{"path": str(path)}] if path else [],
                "resolutions": [64],
                "batch_size": 1,
                "text_encoding": "cached",
            },
            "loop": {"gpu_count": count, "distributed_strategy": strategy, "max_steps": 8},
            "memory": {"activation_checkpointing": "block" if strategy == "fsdp" else "none"},
            "optimizer": {"type": optimizer, "args": args or {}},
        }
    )


def _shapes():
    model = nn.Module()
    with torch.device("meta"):
        model.matrix = nn.Parameter(torch.empty(7, 5))
        model.projector = nn.Parameter(torch.empty(1, 12))
        model.vector = nn.Parameter(torch.empty(7))
        model.scalar = nn.Parameter(torch.empty(()))
        model.register_buffer("frozen_buffer", torch.ones(8, dtype=torch.bfloat16))
    return model


def test_factor_states_follow_actual_odd_and_column_shards_and_replicated_scalar():
    model = _shapes()
    result = _fsdp_memory(model, [], _config())
    # 7x5 => row shards 4/3, full 5-column variance; 1x12 => row
    # variance replicated and 6-column shards; vector variance is unfactored.
    assert result["local_parameter_bytes_by_rank"] == [124, 100]
    assert result["optimizer_state_bytes_by_rank"] == [100, 92]
    assert result["replicated_parameter_count"] == 1
    assert result["replicated_buffer_bytes"] == 16
    assert result["global_parameter_bytes"] == 220
    assert result["groups"][0]["parameter_count"] == 54
    with_momentum = _fsdp_memory(model, [], _config(args={"beta1": 0.9}))
    assert with_momentum["optimizer_state_bytes_by_rank"] == [224, 192]
    assert with_momentum["optimizer_state_layout"] == "factored_with_first_moment"


@pytest.mark.parametrize(
    "optimizer,args,multiplier",
    [("adamw", {}, 2), ("adamw", {"amsgrad": True}, 3), ("sgd", {}, 0), ("sgd", {"momentum": 0.9}, 1)],
)
def test_optimizer_states_are_not_all_assumed_to_be_factored(optimizer, args, multiplier):
    result = _fsdp_memory(_shapes(), [], _config(optimizer=optimizer, args=args))
    assert result["optimizer_state_bytes_by_rank"] == [124 * multiplier, 100 * multiplier]


def test_nested_wrap_groups_and_shared_parameters_are_counted_once():
    with torch.device("meta"):
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))])
        model.shared = model.blocks[0][0]
    result = _fsdp_memory(model, [model.blocks[0][0], model.blocks[0]], _config())
    assert sum(group["parameter_count"] for group in result["groups"]) == 40
    assert result["global_parameter_bytes"] == 160
    assert result["local_parameter_bytes_by_rank"] == [80, 80]


class _CapacityBackbone(nn.Module):
    """12.8B meta parameters, no allocation or dependency on model weights."""

    dim = 64

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(20_000, 20_000, bias=False, device="meta") for _ in range(32)])


@pytest.fixture
def capacity_family(monkeypatch):
    family = get_family("toy")
    monkeypatch.setattr(family, "meta_backbone", lambda cfg: _CapacityBackbone())
    monkeypatch.setattr(family, "memory_layout_meta", lambda model: SimpleNamespace(blocks=model.blocks))
    monkeypatch.setattr(family, "cache_memory_estimate", lambda cfg, dtype: {})
    return family


def test_capacity_estimate_uses_shards_without_shrinking_activations(image_dataset, capacity_family):
    fsdp = plan(_config(image_dataset), device="cuda", gpu_total_mb=64 * 1024)
    ddp = plan(_config(image_dataset, strategy="ddp"), device="cuda", gpu_total_mb=64 * 1024)
    assert fsdp["ok"], fsdp["errors"]
    assert ddp["ok"], ddp["errors"]
    assert fsdp["params"]["trainable"] == ddp["params"]["trainable"] == 12_800_000_000
    assert fsdp["distributed"]["strategy"] == "fsdp"
    assert fsdp["distributed"]["optimizer_storage"] == "sharded"
    assert ddp["distributed"]["parameter_storage"] == "replicated"
    assert fsdp["memory"]["adapter_mb"] == round(12_800_000_000 * 4 / 2 / 2**20, 1)
    # This is exactly the threshold the supervisor uses against each GPU.
    assert fsdp["memory"]["peak_mb_estimate"] < 64 * 1024 * 0.95
    assert ddp["memory"]["peak_mb_estimate"] > 64 * 1024
    assert fsdp["memory"]["communication_mb_estimate"] > 0
    assert fsdp["memory"]["initialization_peak_mb_estimate"] > fsdp["memory"]["adapter_mb"]
    assert fsdp["memory"]["estimate_scope"] == "per_device"
    # For the same per-device batch, adding cards cannot divide activation memory.
    four = plan(_config(image_dataset, count=4), device="cuda")
    assert four["ok"], four["errors"]
    assert four["memory"]["activations_mb_by_bucket"] == fsdp["memory"]["activations_mb_by_bucket"]
    assert any("不保证" in note for note in fsdp["memory"]["estimate_notes"])


@pytest.mark.parametrize("optimizer,args", [("adamw", {}), ("adafactor", {"beta1": 0.9})])
def test_full_moment_states_do_not_falsely_fit_64gib(image_dataset, capacity_family, optimizer, args):
    result = plan(
        _config(image_dataset, optimizer=optimizer, args=args), device="cuda", gpu_total_mb=64 * 1024
    )
    assert result["ok"], result["errors"]
    assert result["memory"]["known_training_residency_mb"] > 64 * 1024
    assert result["memory"]["peak_mb_estimate"] > 64 * 1024
    assert any(warning["code"] == "vram.tight" for warning in result["warnings"])
    assert all("8bit" not in suggestion for suggestion in result["memory"]["suggestions"])


def test_unsharded_cache_phase_remains_an_admission_constraint(image_dataset, capacity_family, monkeypatch):
    monkeypatch.setattr(
        capacity_family, "cache_memory_estimate", lambda cfg, dtype: {"text_encoding": 80 * 1024}
    )
    result = plan(_config(image_dataset), device="cuda", gpu_total_mb=64 * 1024)
    assert result["ok"], result["errors"]
    assert result["memory"]["peak_mb_estimate"] == 80 * 1024
    assert result["memory"]["cache_phase_peak_mb_estimates"]["text_encoding"] == 80 * 1024


def test_fsdp_does_not_admit_cpu_execution(image_dataset):
    result = plan(_config(image_dataset), device="cpu")
    assert not result["ok"]
    assert any(error["loc"] == "loop.distributed_strategy" for error in result["errors"])


def test_job_api_persists_per_card_peak_and_queue_uses_it(
    image_dataset, capacity_family, tmp_path, monkeypatch
):
    from ypuddin.server import create_app
    from ypuddin.server import supervisor as module

    cards = [
        {
            "device": f"cuda:{index}",
            "name": f"test GPU {index}",
            "mem_free_mb": 64 * 1024,
            "mem_total_mb": 64 * 1024,
        }
        for index in range(2)
    ]
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "current_profile", lambda: "linux-cuda")
    monkeypatch.setattr(module, "gpu_info", lambda: cards)
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: cards)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    # Do not start the service lifespan/supervisor: this checks real admission
    # methods with a supplied inventory, and must never launch training workers.
    client = TestClient(app)
    ctx = app.state.ctx
    try:
        for optimizer, expected in [("adafactor", ("cuda:0", "cuda:1")), ("adamw", None)]:
            cfg = _config(image_dataset, optimizer=optimizer)
            response = client.post("/api/jobs", json={"name": "capacity", "config": cfg.to_dict()})
            assert response.status_code == 201, response.text
            row = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
            saved_peak = json.loads(row["progress_json"])["estimated_peak_mb"]
            assert saved_peak == plan(cfg, device="cuda")["memory"]["peak_mb_estimate"]
            assert ctx.supervisor._choose_device(row) == expected
    finally:
        client.close()
        ctx.db.close()


def test_disabling_mixed_precision_accounts_for_fp32_gathers(image_dataset, capacity_family):
    raw = _config(image_dataset).to_dict()
    raw["dataset"]["resolutions"] = [512]
    bf16 = plan(raw, device="cuda")
    raw["loop"]["mixed_precision"] = "no"
    fp32 = plan(raw, device="cuda")
    assert fp32["ok"], fp32["errors"]
    assert fp32["memory"]["communication_mb_estimate"] > bf16["memory"]["communication_mb_estimate"]
    assert bf16["memory"]["activations_mb_by_bucket"][0]["mb"] > 0
    assert (
        fp32["memory"]["activations_mb_by_bucket"][0]["mb"]
        == 2 * bf16["memory"]["activations_mb_by_bucket"][0]["mb"]
    )
