"""Header-only admission estimates agree with real reduced Krea loading/storage."""

import json
import math

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from tests.unit.test_krea2_family import checkpoints  # noqa: F401
from ypuddin.adapters import inject
from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config import AdapterConfig, ModelConfig, TrainConfig
from ypuddin.models.krea2.family import Krea2Family, load_dit
from ypuddin.models.krea2.vendor.krea2_mmdit import KREA2_CONFIG, SingleStreamDiT
from ypuddin.server import create_app
from ypuddin.train.plan import _frozen_storage_bytes, plan


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("filename", ["bare.safetensors", "comfy.safetensors", "fp8.safetensors"])
@pytest.mark.parametrize("precision", ["auto", "bf16", "fp8_e4m3"])
def test_header_plan_matches_actual_loaded_and_injected_storage(checkpoints, filename, precision):  # noqa: F811
    family = Krea2Family()
    cfg = ModelConfig(family="krea2", dit_path=str(checkpoints / filename), dtype="bf16")
    actual, _ = load_dit(cfg.dit_path, device="cpu", dtype=torch.bfloat16)
    with torch.device("meta"):
        expected = family.meta_backbone(cfg)
    family.prepare_backbone_for_plan(expected, cfg, torch.bfloat16)
    adapter = AdapterConfig(algo="lora", preset="attn-only", rank=4, alpha=4)
    for model in (actual, expected):
        inject(
            model,
            adapter,
            family.presets()[adapter.preset],
            base_precision="keep" if precision == "auto" else precision,
        )
    actual_tensors = dict(actual.named_parameters()) | dict(actual.named_buffers())
    expected_tensors = dict(expected.named_parameters()) | dict(expected.named_buffers())
    assert set(actual_tensors) == set(expected_tensors)
    for name, tensor in expected_tensors.items():
        reference = actual_tensors[name]
        assert tensor.is_meta
        assert tensor.shape == reference.shape and tensor.dtype == reference.dtype
        assert tensor.requires_grad == reference.requires_grad
    assert _frozen_storage_bytes(actual) == _frozen_storage_bytes(expected)
    assert [_frozen_storage_bytes(b) for b in actual.blocks] == [
        _frozen_storage_bytes(b) for b in expected.blocks
    ]
    if filename == "fp8.safetensors":
        # Explicit BF16 does not replace already-frozen native FP8 in inject().
        assert any(isinstance(m, FrozenLinear) and m.is_fp8 for m in expected.modules())


def _sparse_checkpoint(path, state):
    header, offset = {}, 0
    for name, tensor in state.items():
        dtype = (
            "F8_E4M3"
            if tensor.dtype == torch.float8_e4m3fn
            else "F32"
            if tensor.dtype == torch.float32
            else "BF16"
        )
        end = offset + tensor.numel() * tensor.element_size()
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, end]}
        offset = end
    payload = json.dumps(header).encode()
    payload += b" " * (-len(payload) % 8)
    with path.open("wb") as stream:
        stream.write(len(payload).to_bytes(8, "little"))
        stream.write(payload)
        stream.truncate(8 + len(payload) + offset)
    return path


@pytest.fixture
def official_geometry(tmp_path):
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    from ypuddin.models.krea2.text import QWEN3_VL_4B_TEXT_CONFIG

    # Full architecture, sparse zero payloads: this verifies planning/admission,
    # never inference or model quality, and allocates no large tensor storage.
    with torch.device("meta"):
        model = SingleStreamDiT(KREA2_CONFIG).to(torch.bfloat16)
        state = dict(model.state_dict())
        bf16 = _sparse_checkpoint(tmp_path / "bf16.safetensors", state)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.startswith(("blocks.", "txtfusion.")):
                state[name + ".weight"] = torch.empty_like(module.weight, dtype=torch.float8_e4m3fn)
                state[name + ".scale_weight"] = torch.empty((), dtype=torch.float32)
        fp8 = _sparse_checkpoint(tmp_path / "no-precision-in-filename.safetensors", state)
        decoder = Qwen3VLTextModel(Qwen3VLTextConfig(**QWEN3_VL_4B_TEXT_CONFIG)).to(torch.bfloat16)
        text = _sparse_checkpoint(tmp_path / "decoder.safetensors", decoder.state_dict())
    # VAE loading is not executed by this planning-only test; use a small
    # safetensors header to keep that independent phase below the text budget.
    vae = tmp_path / "vae.safetensors"
    save_file({"encoder.conv_in.weight": torch.zeros(96, 3, 3, 3)}, str(vae))
    return {"fp8": fp8, "bf16": bf16, "text": text, "vae": vae}


def _config(assets, image_dataset, kind="fp8"):
    return TrainConfig.model_validate(
        {
            "model": {
                "family": "krea2",
                "dit_path": str(assets[kind]),
                "text_encoder_path": str(assets["text"]),
                "vae_path": str(assets["vae"]),
                "dtype": "bf16",
                "attention": "sdpa",
                "krea2_variant": "raw",
            },
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolutions": [256],
                "aspect_ratio_limit": 1,
                "batch_size": 1,
                "text_encoding": "cached",
            },
            "adapter": {"algo": "lora", "preset": "attn-only", "rank": 4, "alpha": 4},
            "memory": {"base_precision": "auto", "blocks_to_swap": 8, "activation_checkpointing": "block"},
            "loop": {"max_steps": 2, "mixed_precision": "bf16"},
            "sampling": {"enabled": False},
            "checkpoint": {"output_dir": str(image_dataset.parent / "output")},
        }
    )


def test_full_geometry_auto_preserves_fp8_and_last_blocks_without_payload_reads(
    official_geometry, image_dataset, monkeypatch
):
    monkeypatch.setattr("safetensors.torch.load_file", lambda *a, **k: pytest.fail("plan read payload"))
    cfg = _config(official_geometry, image_dataset)
    result = plan(cfg, device="cuda", gpu_total_mb=15996)
    assert result["ok"], result["errors"]
    mem = result["memory"]
    family = Krea2Family()
    with torch.device("meta"):
        model = family.meta_backbone(cfg.model)
    family.prepare_backbone_for_plan(model, cfg.model, torch.bfloat16)
    expected = _frozen_storage_bytes(model) / 2**20
    swapped = sum(_frozen_storage_bytes(b) for b in model.blocks[-8:]) / 2**20
    assert mem["weights_mb"] == round(expected)
    assert mem["swapped_mb"] == round(swapped)
    assert not math.isclose(swapped, expected * 8 / 28, rel_tol=0.01)
    assert mem["swap_staging_mb"] > 0 and mem["dequant_mb"] > 0
    assert mem["gradients_mb"] == mem["adapter_mb"]
    assert mem["cache_phase_peak_mb_estimates"]["text_cache"] > 7672 + 2090
    assert mem["peak_mb_estimate"] == max(
        mem["training_peak_mb_estimate"], *mem["cache_phase_peak_mb_estimates"].values()
    )
    assert 10_000 < mem["peak_mb_estimate"] < 15996 * 0.95
    cfg.model.dit_path = str(official_geometry["bf16"])
    bf16 = plan(cfg, device="cuda")["memory"]
    assert bf16["weights_mb"] > mem["weights_mb"] * 1.8
    assert bf16["peak_mb_estimate"] > 15996 * 0.95


def test_api_native_fp8_job_admits_on_16gb_but_preserves_low_memory_queue_gate(
    official_geometry, image_dataset, monkeypatch, tmp_path
):
    gpu = {"device": "cuda:0", "mem_total_mb": 15996, "mem_free_mb": 15000}
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: [gpu])
    monkeypatch.setattr("ypuddin.server.supervisor.gpu_info", lambda: [gpu])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)  # no lifespan: never starts a training worker
    try:
        cfg = _config(official_geometry, image_dataset)
        response = client.post(
            "/api/jobs", json={"type": "train", "name": "header admission", "config": cfg.to_dict()}
        )
        assert response.status_code == 201, response.text
        ctx = app.state.ctx
        job = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
        estimate = json.loads(job["progress_json"])["estimated_peak_mb"]
        assert 10_000 < estimate < gpu["mem_free_mb"] * 0.95
        assert ctx.supervisor._choose_device(job) == "cuda:0"
        gpu["mem_free_mb"] = estimate / 0.95 - 1
        assert ctx.supervisor._choose_device(job) is None
        assert ctx.db.fetchone("SELECT status FROM jobs WHERE id=?", (job["id"],))["status"] == "queued"
    finally:
        client.close()
        app.state.ctx.db.close()


def test_full_krea_cache_admits_using_cache_phases_while_training_keeps_its_peak(
    official_geometry, image_dataset, monkeypatch, tmp_path
):
    # Official model dimensions in sparse headers; no payload loads or GPU
    # workers. Admission uses the real model planner and a 24 GiB inventory.
    gpu = {"device": "cuda:0", "mem_total_mb": 24576, "mem_free_mb": 24576}
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: [gpu])
    monkeypatch.setattr("ypuddin.server.supervisor.gpu_info", lambda: [gpu])
    monkeypatch.setattr("ypuddin.server.supervisor.current_profile", lambda: "linux-cuda")
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)  # No lifespan: do not launch actual workers.
    try:
        config = _config(official_geometry, image_dataset, "bf16").to_dict()
        config["training"]["mode"] = "full"
        config["memory"]["blocks_to_swap"] = 0
        config["loop"].update(gpu_count=2, distributed_strategy="fsdp")
        response = client.post(
            "/api/jobs", json={"type": "cache", "name": "full model cache", "config": config}
        )
        assert response.status_code == 201, response.text
        ctx = app.state.ctx
        cached = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
        normalized = json.loads(cached["config_json"])
        assert normalized["loop"]["gpu_count"] == 1
        assert normalized["loop"]["distributed_strategy"] == "ddp"
        estimate = plan(normalized, device="cuda")["memory"]
        cache_peak = json.loads(cached["progress_json"])["estimated_peak_mb"]
        assert cache_peak == max(estimate["cache_phase_peak_mb_estimates"].values())
        assert 0 < cache_peak < gpu["mem_free_mb"] * 0.95
        assert estimate["peak_mb_estimate"] > gpu["mem_free_mb"]
        assert ctx.supervisor._choose_device(cached) == "cuda:0"
        # The complete training task keeps the complete estimate and cannot
        # bypass its VRAM gate merely because cache preparation fits.
        response = client.post(
            "/api/jobs", json={"type": "train", "name": "full model train", "config": normalized}
        )
        assert response.status_code == 201, response.text
        training = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
        assert json.loads(training["progress_json"])["estimated_peak_mb"] == estimate["peak_mb_estimate"]
        assert ctx.supervisor._choose_device(training) is None
    finally:
        client.close()
        app.state.ctx.db.close()
