"""Per-device FSDP capacity, optimizer layout and admission-facing estimates."""

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from ypuddin.config import TrainConfig
from ypuddin.models import get_family
from ypuddin.train.plan import _fsdp_memory, _online_latent_memory, plan


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


def test_online_vae_uses_actual_image_loader_geometry_without_loading_weights(monkeypatch):
    from ypuddin.models.anima.vendor import qwen_image_vae_2d as qwen

    # The native loader's JSON geometry is independent of planner assumptions.
    # Keep all checkpoint tensors on meta so this executes no large allocation.
    with torch.device("meta"):
        state = qwen.AutoencoderKLQwenImage2D().state_dict()
    monkeypatch.setattr(qwen, "load_safetensors", lambda *_args, **_kwargs: state)
    with torch.device("meta"):
        loaded = qwen.load_vae("no-weight-file.safetensors", device="meta")
    actual_bytes = sum(t.numel() * t.element_size() for t in loaded.state_dict().values())
    monkeypatch.setattr(qwen, "load_safetensors", lambda *_a, **_k: pytest.fail("plan loaded VAE weights"))
    for family in ("anima", "krea2"):
        raw = _config().to_dict()
        raw["model"]["family"] = family
        cfg = TrainConfig.model_validate(raw)
        weights, workspace = _online_latent_memory(cfg, torch.float32, 512 * 512)
        assert weights * 2**20 == actual_bytes
        cfg.dataset.batch_size = 2
        cfg.loop.gpu_count = 4
        more_weights, more_workspace = _online_latent_memory(cfg, torch.float32, 512 * 768)
        assert more_weights == weights  # The VAE is replicated, not sharded.
        assert more_workspace == workspace * 3  # Per-card batch and actual pixel area.


@pytest.mark.parametrize("family", ["sdxl", "flux2"])
def test_online_vae_honors_local_component_geometry_and_fp32(tmp_path, family):
    from diffusers import AutoencoderKL, AutoencoderKLFlux2

    model_type = AutoencoderKL if family == "sdxl" else AutoencoderKLFlux2
    vae = model_type(
        block_out_channels=(32, 32),
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        layers_per_block=1,
        latent_channels=4 if family == "sdxl" else 32,
    )
    root = tmp_path / "local-model"
    component = root / "vae"
    vae.save_config(component)
    raw = _config().to_dict()
    raw["model"].update(family=family, dit_path=str(root))
    cfg = TrainConfig.model_validate(raw)
    weights, _ = _online_latent_memory(cfg, torch.bfloat16, 64 * 64)
    assert weights * 2**20 == sum(t.numel() * t.element_size() for t in vae.state_dict().values())
    assert weights > 0  # These families encode at FP32, even with BF16 backbone compute.


def test_online_vae_overlaps_training_and_changes_per_card_queue_admission(
    image_dataset, capacity_family, tmp_path, monkeypatch
):
    from ypuddin.server import create_app
    from ypuddin.server import supervisor as module

    # Standalone VAE encoding fits comfortably; overlapping it with the 12.8B
    # full-training shards must nevertheless stop admission on these cards.
    weights_mb, workspace_mb = 8 * 1024, 2 * 1024
    monkeypatch.setattr(
        "ypuddin.train.plan._online_latent_memory",
        lambda *_args: (weights_mb, workspace_mb),
    )
    monkeypatch.setattr(
        capacity_family, "cache_memory_estimate", lambda *_args: {"latent_cache": weights_mb + workspace_mb}
    )
    cards = [
        {"device": f"cuda:{i}", "name": "test GPU", "mem_free_mb": 64 * 1024, "mem_total_mb": 64 * 1024}
        for i in range(2)
    ]
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "current_profile", lambda: "linux-cuda")
    monkeypatch.setattr(module, "gpu_info", lambda: cards)
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: cards)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    cfg = _config(image_dataset)
    cached = plan(cfg, device="cuda")["memory"]
    cfg.dataset.cache_latents = False
    online = plan(cfg, device="cuda")["memory"]
    assert (
        online["training_peak_mb_estimate"] == cached["training_peak_mb_estimate"] + weights_mb + workspace_mb
    )
    assert online["known_training_residency_mb"] == cached["known_training_residency_mb"] + weights_mb
    assert online["latent_encoder_mb"] == weights_mb
    assert online["latent_encoding_workspace_mb_estimate"] == workspace_mb
    cfg.loop.gpu_count = 4
    four_cards = plan(cfg, device="cuda")["memory"]
    assert four_cards["latent_encoder_mb"] == weights_mb
    assert four_cards["latent_encoding_workspace_mb_estimate"] == workspace_mb
    cfg.loop.gpu_count = 2
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)  # No lifespan: this cannot launch a GPU worker.
    try:
        for cache_latents, devices in [(True, ("cuda:0", "cuda:1")), (False, None)]:
            cfg.dataset.cache_latents = cache_latents
            response = client.post("/api/jobs", json={"name": "online VAE", "config": cfg.to_dict()})
            assert response.status_code == 201, response.text
            row = app.state.ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
            expected = cached if cache_latents else online
            assert json.loads(row["progress_json"])["estimated_peak_mb"] == expected["peak_mb_estimate"]
            assert app.state.ctx.supervisor._choose_device(row) == devices
    finally:
        client.close()
        app.state.ctx.db.close()


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
