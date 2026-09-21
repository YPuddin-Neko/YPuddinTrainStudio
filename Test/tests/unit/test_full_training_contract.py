"""Full-model selection, memory planning, compiled names and artifact HTTP operations."""

import io
import json
import zipfile

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from torch import nn

from ypuddin.config import TrainConfig
from ypuddin.server import create_app
from ypuddin.server.db import now
from ypuddin.train import Trainer
from ypuddin.train.plan import plan
from ypuddin.train.training_modes import FullTrainingSet


@pytest.mark.parametrize(
    "selection,patch",
    [
        ({"mode": "adapter", "train_text_encoder": True}, {}),
        ({"mode": "full", "train_backbone": False, "train_text_encoder": False}, {}),
        ({"mode": "full", "train_text_encoder": True}, {"dataset": {"text_encoding": "cached"}}),
        ({"mode": "full", "train_text_encoder": True}, {"memory": {"offload_text_encoder": True}}),
        ({"mode": "full"}, {"memory": {"blocks_to_swap": 1}}),
        ({"mode": "full"}, {"memory": {"base_precision": "fp8_e4m3"}}),
    ],
)
def test_unsupported_combinations_fail_before_loading(selection, patch):
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"model": {"family": "toy"}, "training": selection, **patch})


@pytest.mark.parametrize("backbone,text", [(True, False), (False, True), (True, True)])
def test_full_plan_counts_selected_real_parameters_and_no_false_peak(image_dataset, tmp_path, backbone, text):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "training": {"mode": "full", "train_backbone": backbone, "train_text_encoder": text},
            "dataset": {"sources": [{"path": str(image_dataset)}], "resolutions": [32], "num_workers": 0},
            "loop": {"max_steps": 1},
            "checkpoint": {"output_dir": str(tmp_path / "plan-run")},
        }
    )
    planned = plan(cfg, device="cpu", gpu_total_mb=1)
    assert planned["ok"], planned["errors"]
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    assert planned["params"]["trainable"] == trainer.adapters.num_params()
    expected_frozen = (
        0 if backbone else sum(t.numel() * t.element_size() for t in trainer.loaded.backbone.parameters())
    )
    assert planned["memory"]["weights_mb"] == round(expected_frozen / 2**20)
    assert planned["text_encoding"] == "online"
    assert (planned["memory"]["peak_mb_estimate"] is None) == text
    assert (planned["memory"]["training_peak_mb_estimate"] is None) == text
    assert all("blocks_to_swap" not in value for value in planned["memory"]["suggestions"])
    trainer._close_logs()


def test_compiled_full_model_keys_remain_native_and_restore():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    training = FullTrainingSet({"backbone": model})
    before = training.training_state_dict()

    class Compiled(nn.Module):
        def __init__(self, original):
            super().__init__()
            self._orig_mod = original

        def forward(self, x):
            return self._orig_mod(x)

    model[0] = Compiled(model[0])
    assert training.training_state_dict().keys() == before.keys()
    assert not any("_orig_mod" in key for key in training.export_state()[0])
    optimizer = torch.optim.SGD(training.parameters(), lr=0.1)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    training.load_training_state(before)
    for key, value in training.training_state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def _artifact_api(tmp_path):
    app = create_app(tmp_path / "studio", poll_interval=0.1)
    path = tmp_path / "full-final.model"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ypuddin-full-model-v1",
                "family": "sdxl",
                "components": {"backbone": "backbone/model.safetensors"},
            }
        )
    )
    (path / "backbone").mkdir()
    (path / "backbone/model.safetensors").write_bytes(b"test model weights")
    app.state.ctx.db.insert(
        "artifacts",
        {
            "id": "full_model",
            "name": path.name,
            "path": str(path),
            "kind": "model",
            "size": 100,
            "created_at": now(),
        },
    )
    return app, path


def test_full_model_artifact_list_download_convert_and_delete(tmp_path):
    app, path = _artifact_api(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/artifacts")
        assert response.status_code == 200, response.text
        assert response.json()[0]["kind"] == "model"
        assert response.json()[0]["family"] == "sdxl"
        assert response.json()[0]["metadata"]["components"] == ["backbone"]
        download = client.get("/api/artifacts/full_model/download")
        assert download.status_code == 200, download.text
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            assert archive.read("full-final.model/backbone/model.safetensors") == b"test model weights"
        converted = client.post("/api/artifacts/full_model/convert", json={"format": "kohya"})
        assert converted.status_code == 422
        removed = client.delete("/api/artifacts/full_model?delete_file=true")
        assert removed.status_code == 200, removed.text
        assert not path.exists()
        assert client.get("/api/artifacts").json() == []


def test_model_download_rejects_symlinks_and_delete_rejects_invalid_manifest(tmp_path):
    app, path = _artifact_api(tmp_path)
    external = tmp_path / "unrelated.txt"
    external.write_text("keep me")
    (path / "external.txt").symlink_to(external)
    with TestClient(app) as client:
        assert client.get("/api/artifacts/full_model/download").status_code == 409
        (path / "manifest.json").write_text("{}")
        assert client.delete("/api/artifacts/full_model?delete_file=true").status_code == 409
        assert external.read_text() == "keep me"
