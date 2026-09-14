"""Full XYZ must sample selected native components, including a changed text encoder."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from safetensors.torch import load_file

from tests.e2e.test_anima_pipeline import tiny_models as anima_models  # noqa: F401
from tests.e2e.test_anima_pipeline import tiny_vae_loader as tiny_qwen_vae  # noqa: F401
from tests.e2e.test_krea2_pipeline import tiny_models as krea_models  # noqa: F401
from tests.unit.test_flux2_family import tiny_root as klein_models  # noqa: F401
from tests.unit.test_sdxl_family import tiny_pipeline as sdxl_models  # noqa: F401
from ypuddin.config import TrainConfig
from ypuddin.models import get_family
from ypuddin.server import create_app
from ypuddin.server.db import now
from ypuddin.server.xyz_worker import generate
from ypuddin.train import Trainer


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def train_full(root, model, selection="both"):
    data = root / "data"
    data.mkdir(parents=True)
    Image.new("RGB", (64, 64), (160, 70, 20)).save(data / "cat.png")
    (data / "cat.txt").write_text("a red cat")
    cfg = TrainConfig.model_validate(
        {
            "model": dict(model, dtype="fp32"),
            "training": {
                "mode": "full",
                "train_backbone": selection != "text",
                "train_text_encoder": selection != "main",
            },
            "dataset": {
                "sources": [{"path": str(data)}],
                "resolutions": [64],
                "batch_size": 1,
                "num_workers": 0,
            },
            "loop": {"max_steps": 2, "mixed_precision": "no", "seed": 7},
            "objective": {"timestep_sampling": "uniform"},
            "optimizer": {"lr": 0.01},
            "sampling": {"enabled": False},
            "checkpoint": {
                "output_dir": str(root / "train"),
                "name": "full",
                "save_every_epochs": None,
                "save_every_steps": 1,
                "save_dtype": "fp32",
            },
        }
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    paths = [root / "train/full-step000001.model", root / "train/full-final.model"]
    assert all(path.is_dir() for path in paths)
    return cfg, paths


@contextmanager
def api_for(tmp_path, cfg, paths):
    app = create_app(tmp_path / "studio", poll_interval=0.02)
    ctx = app.state.ctx
    ctx.db.insert(
        "jobs",
        {
            "id": "source",
            "type": "train",
            "name": "full",
            "status": "completed",
            "project_id": None,
            "created_at": now(),
            "run_dir": str(paths[0].parent),
            "config_json": json.dumps(cfg.to_dict()),
        },
    )
    for i, path in enumerate(paths, 1):
        ctx.db.insert(
            "artifacts",
            {
                "id": f"model-{i}",
                "job_id": "source",
                "name": path.name,
                "path": str(path),
                "kind": "model",
                "step": i,
                "created_at": now(),
            },
        )
    ctx.db.set_kv("queue.settings", {"held": True, "max_concurrent": 1})
    with TestClient(app) as client:
        yield client, ctx


def body(**patch):
    return dict(
        prompt="a red cat",
        width=64,
        height=64,
        steps=2,
        cfg=1,
        seed=7,
        checkpoint_id="model-2",
        x={"key": "checkpoint", "values": ["model-1", "model-2"]},
        **patch,
    )


def enqueue(client, ctx, data=None):
    response = client.post("/api/jobs/source/xyz", json=data or body())
    assert response.status_code == 202, response.text
    job = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
    payload = json.loads(job["config_json"])
    payload.update(device="cpu", fingerprint_cache=str(ctx.data_root / "fingerprints"))
    return job, payload


@pytest.mark.usefixtures("tiny_qwen_vae")
@pytest.mark.parametrize("family", ["toy", "anima", "krea2", "sdxl", "flux2"])
@pytest.mark.parametrize("selection", ["main", "text", "both"])
def test_selected_full_components_are_loaded_and_sampled(request, tmp_path, monkeypatch, family, selection):
    model = {"family": family}
    if family in {"anima", "krea2"}:
        assets = request.getfixturevalue("anima_models" if family == "anima" else "krea_models")
        model.update(
            dit_path=str(assets["dit"]),
            vae_path=str(assets["vae"]),
            text_encoder_path=str(assets["qwen"] if family == "anima" else assets["qwen_dir"]),
        )
    elif family in {"sdxl", "flux2"}:
        model["dit_path"] = str(
            request.getfixturevalue("sdxl_models" if family == "sdxl" else "klein_models")
        )
    cfg, paths = train_full(tmp_path, model, selection)
    expected = {}
    for i, path in enumerate(paths, 1):
        manifest = json.loads((path / "manifest.json").read_text())
        expected[f"model-{i}"] = {
            component: load_file(str(path / relative))
            for component, relative in manifest["components"].items()
        }
    assert any(
        not torch.equal(expected["model-1"][component][key], value)
        for component, tensors in expected["model-2"].items()
        for key, value in tensors.items()
    )
    with api_for(tmp_path, cfg, paths) as (client, ctx):
        options = client.get("/api/jobs/source/xyz/options").json()
        assert options["training_mode"] == "full"
        assert options["defaults"]["checkpoint_id"] == "model-2"
        assert options["sampling_models"] == []
        assert "adapter_scale" not in {axis["key"] for axis in options["axes"]}
        assert {row["kind"] for row in options["checkpoints"]} == {"model"}
        job, payload = enqueue(client, ctx)
        # Raw source model is deliberately unusable after enqueue: the worker must use each native snapshot.
        payload["model"]["dit_path"] = str(tmp_path / "MUST-NOT-LOAD-ORIGINAL.safetensors")
        observed_loads, observed_text, observed_forward = [], [], []
        implementation = get_family(family)
        original_load, original_forward = implementation.load, implementation.forward

        def checked_load(selected, *args, **kwargs):
            matching = [
                key
                for key, item in payload["xyz"]["checkpoints"].items()
                if item["model"] == selected.model_dump(mode="json")
            ]
            assert len(matching) == 1
            identity = matching[0]
            observed_loads.append(identity)
            loaded = original_load(selected, *args, **kwargs)
            encode = loaded.text.encode

            def checked_encode(*encode_args, **encode_kwargs):
                value = encode(*encode_args, **encode_kwargs)
                for component, module in loaded.text.trainable_modules().items():
                    if component in expected[identity]:
                        for key, tensor in module.state_dict().items():
                            torch.testing.assert_close(
                                tensor.cpu(), expected[identity][component][key], rtol=0, atol=0
                            )
                observed_text.append(identity)
                return value

            loaded.text.encode = checked_encode
            loaded.extra["test_checkpoint"] = identity
            return loaded

        def checked_forward(loaded, *args, **kwargs):
            identity = loaded.extra["test_checkpoint"]
            if "backbone" in expected[identity]:
                for key, tensor in loaded.backbone.state_dict().items():
                    torch.testing.assert_close(
                        tensor.cpu(), expected[identity]["backbone"][key], rtol=0, atol=0
                    )
            observed_forward.append(identity)
            return original_forward(loaded, *args, **kwargs)

        monkeypatch.setattr(implementation, "load", checked_load)
        monkeypatch.setattr(implementation, "forward", checked_forward)
        monkeypatch.setattr(
            "ypuddin.server.xyz_worker.bind_checkpoint",
            lambda *args: pytest.fail("Full model must not bind an adapter"),
        )
        output = Path(job["samples_dir"])
        generate(payload, output, lambda *args, **kwargs: None, lambda: False)
        result = json.loads((output / "manifest.json").read_text())
        assert result["complete"] and len(result["cells"]) == 2
        assert observed_loads == ["model-1", "model-2"]
        assert set(observed_text) == set(observed_forward) == {"model-1", "model-2"}
        assert [cell["checkpoint_id"] for cell in result["cells"]] == ["model-1", "model-2"]
        for cell in result["cells"]:
            with Image.open(output / cell["file"]) as image:
                assert image.size == (64, 64)


def test_full_xyz_rejects_base_fallback_and_mutated_models(tmp_path):
    cfg, paths = train_full(tmp_path, {"family": "toy"})
    with api_for(tmp_path, cfg, paths) as (client, ctx):
        for patch in (
            {"checkpoint_id": None, "x": {"key": "seed", "values": [1]}},
            {"checkpoint_id": "missing", "x": {"key": "seed", "values": [1]}},
            {"adapter_scale": 0},
            {"sampling_model_id": "original"},
            {"x": {"key": "adapter_scale", "values": [1]}},
        ):
            request = body() | patch
            response = client.post("/api/jobs/source/xyz", json=request)
            assert response.status_code == 422, response.text
        assert ctx.db.fetchall("SELECT id FROM jobs WHERE type='xyz'") == []
        job, payload = enqueue(client, ctx)
        assert client.delete("/api/artifacts/model-1?delete_file=true").status_code == 409
        missing = copy.deepcopy(payload)
        missing["xyz"]["request"].update(checkpoint_id=None, x={"key": "seed", "values": [1]})
        with pytest.raises(ValueError, match="base fallback"):
            generate(missing, tmp_path / "bad", lambda *args, **kwargs: None, lambda: False)
        legacy = copy.deepcopy(payload)
        del legacy["training"]
        ctx.db.update("jobs", job["id"], {"config_json": json.dumps(legacy)})
        with pytest.raises(ValueError, match="legacy XYZ"):
            ctx.supervisor._launch(
                ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (job["id"],)), device="cpu"
            )
        with (paths[0] / "manifest.json").open("a") as handle:
            handle.write(" ")
        with pytest.raises(ValueError, match="changed after"):
            generate(payload, tmp_path / "changed", lambda *args, **kwargs: None, lambda: False)
