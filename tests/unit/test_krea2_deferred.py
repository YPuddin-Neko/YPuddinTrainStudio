"""Real Krea cache/train/Turbo sampling lifecycles keep DiT out of encoder phases."""

import json
import os
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from safetensors.torch import load_file, save_file

from tests.unit.test_krea2_family import (  # noqa: F401
    _cond,
    _fp8_scaled,
    _tiny_state_dict,
    checkpoints,
    tiny_qwen3vl,
)
from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config import MemoryConfig, ModelConfig, TrainConfig
from ypuddin.models.anima.vendor import qwen_image_vae_2d
from ypuddin.models.base import LoadedModel
from ypuddin.models.krea2 import family as krea_module
from ypuddin.models.krea2.family import Krea2Family, load_dit
from ypuddin.server import create_app
from ypuddin.server.db import now
from ypuddin.server.xyz_worker import generate
from ypuddin.train import Trainer


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize(
    "device,requested,blocks,free_delta,expected,reason",
    [
        ("cuda:2", "cpu", 8, 0, "cuda:2", "cuda_staging_before_host_masters"),
        ("cuda:2", "cpu", 8, 1, "cuda:2", "cuda_staging_before_host_masters"),
        ("cuda:2", "cpu", 8, -1, "cpu", "insufficient_cuda_headroom"),
        ("cuda:2", "cpu", 8, None, "cpu", "cuda_memory_unavailable"),
        ("cpu", "cpu", 8, None, "cpu", "requested_placement"),
        ("mps", "cpu", 8, None, "cpu", "requested_placement"),
        ("cuda:2", "cpu", 0, None, "cpu", "requested_placement"),
        ("cuda:2", "cuda:2", 8, None, "cuda:2", "requested_placement"),
        ("cuda:2", "mps", 8, None, "mps", "requested_placement"),
    ],
)
def test_cuda_staging_requires_real_header_budget_and_two_gib_headroom(
    monkeypatch, device, requested, blocks, free_delta, expected, reason
):
    weight_bytes, calls = 123456, []
    loaded = LoadedModel(
        None,
        None,
        None,
        torch.device(device),
        torch.bfloat16,
        extra={
            "backbone_device": requested,
            "blocks_to_swap": blocks,
            "dit_path": Path("explicit.safetensors"),
        },
    )

    def budget(path, dtype):
        assert path == loaded.extra["dit_path"] and dtype == loaded.dtype
        calls.append("header")
        return weight_bytes

    def available(target):
        assert target == torch.device(device)
        calls.append("cuda")
        if free_delta is None:
            raise RuntimeError("device query unavailable")
        free = weight_bytes + krea_module.CUDA_STAGING_MARGIN + free_delta
        return free, free * 2

    monkeypatch.setattr(krea_module, "_checkpoint_storage_bytes", budget)
    monkeypatch.setattr(torch.cuda, "mem_get_info", available)
    assert krea_module._materialization_device(loaded) == torch.device(expected)
    assert loaded.extra["materialization_placement"]["reason"] == reason
    assert loaded.extra["backbone_device"] == requested and loaded.extra["blocks_to_swap"] == blocks
    assert calls == ([] if reason == "requested_placement" else ["header", "cuda"])


def test_materialize_uses_admitted_device_after_encoders_unload(assets, monkeypatch):
    family = Krea2Family()
    loaded = family.load(
        assets, MemoryConfig(blocks_to_swap=8), device="cpu", dtype=torch.float32, backbone_device="cpu"
    )
    # Only placement selection is emulated; the loader still executes the
    # actual reduced FP8 checkpoint on CPU in this GPU-free integration test.
    loaded.device = torch.device("cuda:1")
    size = krea_module._checkpoint_storage_bytes(Path(assets.dit_path), loaded.dtype)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (size + krea_module.CUDA_STAGING_MARGIN, 16 * 1024**3)
    )
    original = krea_module.load_dit
    observed = []

    def load(path, *, device, dtype):
        assert loaded.text.encoder is None and loaded.latent.vae is None
        observed.append(str(device))
        return original(path, device="cpu", dtype=dtype)

    monkeypatch.setattr(krea_module, "load_dit", load)
    family.materialize_backbone(loaded)
    family.materialize_backbone(loaded)
    assert observed == ["cuda:1"]
    assert loaded.extra["materialization_placement"]["device"] == "cuda:1"
    assert loaded.extra["blocks_to_swap"] == 8 and loaded.extra["materialized"] is True
    assert sum(isinstance(layer, FrozenLinear) and layer.is_fp8 for layer in loaded.backbone.modules()) == 16


@pytest.fixture
def assets(checkpoints, tiny_qwen3vl, tmp_path, monkeypatch):  # noqa: F811
    path = tmp_path / "vae.safetensors"
    vae = qwen_image_vae_2d.AutoencoderKLQwenImage2D(base_dim=16, z_dim=16, num_res_blocks=1)
    save_file(vae.state_dict(), str(path))

    def tiny_vae(source, device="cpu", **kwargs):
        # Only the VAE constructor dimensions change; encoding and decoding use
        # the real Qwen-Image architecture and real saved weights.
        assert Path(source) == path
        model = qwen_image_vae_2d.AutoencoderKLQwenImage2D(base_dim=16, z_dim=16, num_res_blocks=1)
        model.load_state_dict(load_file(str(source)), strict=True, assign=True)
        return model.to(device).eval().requires_grad_(False)

    monkeypatch.setattr(qwen_image_vae_2d, "load_vae", tiny_vae)
    return ModelConfig(
        family="krea2",
        dit_path=str(checkpoints / "fp8.safetensors"),
        text_encoder_path=str(tiny_qwen3vl["single"]),
        tokenizer_path=str(tiny_qwen3vl["hf"]),
        vae_path=str(path),
        krea2_variant="raw",
        dtype="fp32",
        attention="sdpa",
    )


def configuration(assets, tmp_path, *, algo="lora"):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    Image.new("RGB", (64, 64), (120, 60, 30)).save(data / "cat.png")
    (data / "cat.txt").write_text("a cat", encoding="utf-8")
    return TrainConfig.model_validate(
        {
            "model": assets.model_dump(mode="json"),
            "dataset": {
                "sources": [{"path": str(data)}],
                "resolutions": [64],
                "batch_size": 1,
                "num_workers": 0,
                "cache_latents": True,
                "text_encoding": "cached",
                "caption": {"shuffle": False},
            },
            "adapter": {
                "algo": algo,
                "rank": "full" if algo == "lokr" else 4,
                "alpha": 4,
                "factor": 2,
                "preset": "attn-only",
            },
            "optimizer": {"type": "adamw", "lr": 0.001},
            "memory": {"activation_checkpointing": "block", "blocks_to_swap": 1},
            "loop": {"max_steps": 2, "mixed_precision": "no"},
            "sampling": {
                "enabled": True,
                "at_start": False,
                "every_steps": 2,
                "every_epochs": None,
                "width": 64,
                "height": 64,
                "steps": 2,
                "cfg": 1,
                "sampler": "euler",
                "scheduler": "uniform",
                "prompts": [{"prompt": "a cat", "seed": 42}],
            },
            "checkpoint": {
                "output_dir": str(tmp_path / "train"),
                "save_on_finish": True,
                "save_dtype": "fp32",
            },
            "logging": {"tensorboard": False},
        }
    )


def test_cache_only_never_loads_dit_tensors(assets, tmp_path, monkeypatch):
    import safetensors.torch

    original = safetensors.torch.load_file
    seen = []

    def check(path, *args, **kwargs):
        assert Path(path).resolve() != Path(assets.dit_path).resolve(), "cache loaded the DiT payload"
        seen.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(safetensors.torch, "load_file", check)
    trainer = Trainer(configuration(assets, tmp_path), device="cpu")
    try:
        trainer.prepare_data()
        assert trainer.bundle.plan.images == 1
        assert all(parameter.is_meta for parameter in trainer.loaded.backbone.parameters())
        assert trainer.loaded.extra["materialized"] is False
        assert trainer.loaded.text.encoder is None and trainer.loaded.latent.vae is None
        assert Path(assets.text_encoder_path) in seen  # real text encoding actually happened
    finally:
        trainer._close_logs()
        trainer.emitter.close()


@pytest.mark.parametrize("variant", ["raw", "turbo"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_materialize_unloads_encoders_and_preserves_fp8(assets, monkeypatch, variant, dtype):
    family = Krea2Family()
    loaded = family.load(
        assets.model_copy(update={"krea2_variant": variant}),
        MemoryConfig(activation_checkpointing="block"),
        device="cpu",
        dtype=dtype,
        backbone_device="cpu",
    )
    assert all(parameter.is_meta for parameter in loaded.backbone.parameters())
    loaded.text.encode_for_cache(["a cat"])
    loaded.latent.encode(torch.zeros(1, 3, 64, 64))
    assert loaded.text.encoder is not None and loaded.latent.vae is not None
    original = krea_module.load_dit
    reads = []

    def observed(*args, **kwargs):
        assert loaded.text.encoder is None and loaded.latent.vae is None
        reads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(krea_module, "load_dit", observed)
    family.materialize_backbone(loaded)
    first = loaded.backbone
    family.materialize_backbone(loaded)
    assert loaded.backbone is first and len(reads) == 1
    assert loaded.extra["materialized"] is True and loaded.extra["variant"] == variant
    assert first.gradient_checkpointing and first.attn_mode == "torch"
    frozen = [m for m in first.modules() if isinstance(m, FrozenLinear)]
    assert len(frozen) == 16 and all(m.is_fp8 for m in frozen)
    reference, _ = load_dit(assets.dit_path, device="cpu", dtype=dtype)
    expected = type(loaded)(reference, None, None, loaded.device, loaded.dtype)
    x, t, cond = torch.randn(2, 16, 8, 8).to(dtype), torch.rand(2), _cond()
    with torch.no_grad():
        torch.testing.assert_close(
            family.forward(loaded, x, t, cond), family.forward(expected, x, t, cond), rtol=0, atol=0
        )


def test_inference_requires_materialization_and_changed_source_is_rejected(assets, tmp_path):
    copy = tmp_path / "dit.safetensors"
    copy.write_bytes(Path(assets.dit_path).read_bytes())
    family = Krea2Family()
    loaded = family.load(
        assets.model_copy(update={"dit_path": str(copy)}), MemoryConfig(), device="cpu", dtype=torch.float32
    )
    with pytest.raises(RuntimeError, match="materialize_backbone"):
        family.forward(loaded, torch.zeros(1, 16, 8, 8), torch.zeros(1), _cond(1, valid=(7,)))
    stat = copy.stat()
    os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        family.materialize_backbone(loaded)
    assert loaded.extra["materialized"] is False
    assert all(parameter.is_meta for parameter in loaded.backbone.parameters())


@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_real_training_checkpoint_and_turbo_xyz(assets, tmp_path, monkeypatch, algo):
    cfg = configuration(assets, tmp_path, algo=algo)
    trainer = Trainer(cfg, device="cpu")
    original = krea_module.load_dit
    order = []

    def observed(*args, **kwargs):
        assert trainer.loaded.text.encoder is None and trainer.loaded.latent.vae is None
        order.append("dit")
        return original(*args, **kwargs)

    monkeypatch.setattr(krea_module, "load_dit", observed)
    trainer.prepare_data()
    assert not order
    trainer._prepare_training()
    trainer._preparing = False
    assert order == ["dit"]
    before = {k: v.clone() for k, v in trainer.adapters.training_state_dict().items()}
    assert trainer.run() == "finished" and trainer.progress.step == 2
    assert any(not torch.equal(before[k], v) for k, v in trainer.adapters.training_state_dict().items())
    checkpoint = next((tmp_path / "train").glob("*final.safetensors"))
    exported, _ = trainer.adapters.export_state()
    restored = load_file(str(checkpoint))
    assert restored.keys() == exported.keys()
    for key, value in restored.items():
        torch.testing.assert_close(value, exported[key], rtol=0, atol=0)
    assert list((tmp_path / "train/samples").glob("*.png"))
    monkeypatch.setattr(krea_module, "load_dit", original)

    app = create_app(tmp_path / "studio")
    context = app.state.ctx
    context.db.set_kv("queue.settings", {"held": True, "max_concurrent": 1})
    context.db.insert(
        "jobs",
        {
            "id": "source",
            "type": "train",
            "name": "Raw",
            "status": "completed",
            "created_at": now(),
            "run_dir": str(checkpoint.parent),
            "config_json": json.dumps(cfg.to_dict()),
        },
    )
    context.db.insert(
        "artifacts",
        {
            "id": "adapter",
            "job_id": "source",
            "kind": "weights",
            "name": checkpoint.name,
            "path": str(checkpoint),
            "step": 2,
            "created_at": now(),
        },
    )
    turbo_path = tmp_path / "turbo.safetensors"
    save_file(_fp8_scaled(_tiny_state_dict(seed=99)), str(turbo_path))
    with TestClient(app) as client:
        registered = client.post(
            "/api/models",
            json={"family": "krea2", "kind": "dit", "path": str(turbo_path), "variant": "turbo"},
        )
        assert registered.status_code == 200, registered.text
        assert registered.json()["purpose"] == "inference"
        response = client.post(
            "/api/jobs/source/xyz",
            json={
                "prompt": "a cat",
                "width": 64,
                "height": 64,
                "steps": 2,
                "cfg": 0,
                "checkpoint_id": "adapter",
                "sampling_model_id": registered.json()["id"],
                "x": {"key": "adapter_scale", "values": [0, 1]},
            },
        )
        assert response.status_code == 202, response.text
        row = context.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
        payload = json.loads(row["config_json"])
        assert payload["model"]["krea2_variant"] == "turbo"
        assert Path(payload["model"]["dit_path"]) == turbo_path
        payload.update(device="cpu", fingerprint_cache=str(tmp_path / "fingerprints"))
        generate(payload, tmp_path / "xyz", lambda *args, **kwargs: None, lambda: False)
    manifest = json.loads((tmp_path / "xyz/manifest.json").read_text())
    assert manifest["complete"] and len(manifest["cells"]) == 2
    for cell in manifest["cells"]:
        assert cell["cfg"] == 0
        with Image.open(tmp_path / "xyz" / cell["file"]) as image:
            assert image.size == (64, 64)
