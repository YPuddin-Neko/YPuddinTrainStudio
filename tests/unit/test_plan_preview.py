"""Incomplete training drafts still expose truthful, independently validated image layouts."""

from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import TrainConfig
from ypuddin.models import get_family
from ypuddin.server import create_app, routes_core, routes_work
from ypuddin.train.plan import plan


@pytest.fixture
def images(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for i, size in enumerate([(101, 79), (64, 96), (128, 48), (80, 80)] * 2):
        Image.new("RGB", size, (20 * i, 50, 160)).save(source / f"image_{i}.png")
        (source / f"image_{i}.txt").write_text(f"image {i}", encoding="utf-8")
    return source


def raw_config(images, *, native=False):
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(images), "repeats": 2}],
                "resolutions": [64, 80],
                "bucket_step": 16,
                "batch_size": 3,
                "resolution_mode": "native" if native else "bucket",
                "image_fit": "pad",
                "native_max_pixels": 4096,
                "native_max_side": 128,
                "bucket_no_upscale": True,
            },
            "loop": {"epochs": 3, "grad_accum": 2, "mixed_precision": "no"},
        }
    ).to_dict()


@pytest.mark.parametrize("native", [False, True])
def test_empty_sampling_prompts_preserve_exact_data_layout_but_never_plan_models(images, monkeypatch, native):
    raw = raw_config(images, native=native)
    expected = plan(raw)
    assert expected["ok"], expected["errors"]
    raw["sampling"].update(enabled=True, prompts=[], prompts_file=None)
    original = deepcopy(raw)

    def no_model(*args, **kwargs):
        raise AssertionError("An invalid training draft must not construct a backbone")

    monkeypatch.setattr(get_family("toy"), "meta_backbone", no_model)
    preview = plan(raw)
    assert preview["ok"] is False
    assert any("sampling.enabled requires sampling.prompts" in e["msg"] for e in preview["errors"])
    assert preview["images"] == 8 and preview["captioned"] == 8
    for key in (
        "images",
        "items",
        "captioned",
        "validation_images",
        "buckets",
        "image_fit",
        "steps_per_epoch",
        "total_steps",
        "epochs",
    ):
        assert preview[key] == expected[key], key
    if native:
        assert preview["native"] == expected["native"]
        assert preview["native"]["downscaled"] > 0
    assert "params" not in preview and "memory" not in preview
    assert raw == original


def test_optimizer_errors_keep_preview_and_preserve_validation_exclusion(images):
    raw = raw_config(images)
    raw["validation"].update(enabled=True, split_ratio=0.5)
    expected = plan(raw)
    assert expected["ok"] and 0 < expected["images"] < 8
    raw["optimizer"].update(lr=-1, type=123)
    preview = plan(raw)
    assert not preview["ok"]
    assert {e["loc"] for e in preview["errors"]} >= {"optimizer.lr", "optimizer.type"}
    for key in ("images", "items", "validation_images", "buckets", "image_fit"):
        assert preview[key] == expected[key]


@pytest.mark.parametrize("update", [{"batch_size": 0}, {"resolutions": []}, {"sources": None}])
def test_invalid_dataset_does_not_scan_or_substitute_defaults(images, monkeypatch, update):
    raw = raw_config(images)
    raw["dataset"].update(update)
    module = import_module("ypuddin.train.plan")
    monkeypatch.setattr(
        module, "prepare_data_layout", lambda *a, **k: pytest.fail("invalid data was scanned")
    )
    result = plan(raw)
    assert not result["ok"] and any(e["loc"].startswith("dataset.") for e in result["errors"])
    assert "buckets" not in result and "image_fit" not in result


def test_invalid_data_geometry_keeps_its_error_without_publishing_a_fake_preview(images):
    raw = raw_config(images)
    raw["sampling"]["enabled"] = True
    raw["dataset"]["bucket_step"] = 9
    result = plan(raw)
    assert not result["ok"]
    assert any(e["loc"] == "dataset.bucket_step" for e in result["errors"])
    assert "buckets" not in result and "image_fit" not in result


@pytest.mark.parametrize("family", ["unknown", "flux3", None])
def test_unknown_or_missing_family_never_guesses_alignment(images, monkeypatch, family):
    raw = raw_config(images)
    raw["sampling"]["enabled"] = True
    raw["model"]["family"] = family
    module = import_module("ypuddin.train.plan")
    monkeypatch.setattr(
        module, "prepare_data_layout", lambda *a, **k: pytest.fail("unknown family was scanned")
    )
    result = plan(raw)
    assert not result["ok"] and any(e["loc"] == "model.family" for e in result["errors"])
    assert "buckets" not in result


@pytest.mark.parametrize("validation", [{"enabled": True}, {"split_ratio": -1}, {"sources": None}])
def test_invalid_validation_partition_never_uses_all_images_as_training(images, validation):
    raw = raw_config(images)
    raw["validation"] = validation
    result = plan(raw)
    assert not result["ok"]
    assert "buckets" not in result and "image_fit" not in result


@pytest.mark.parametrize("loop", [{"grad_accum": 0}, {"epochs": None, "max_steps": None}])
def test_invalid_loop_does_not_invent_steps_but_retains_data_preview(images, loop):
    raw = raw_config(images, native=True)
    raw["loop"].update(loop)
    result = plan(raw)
    assert not result["ok"] and result["images"] == 8 and result["buckets"]
    assert result["native"]["forward_groups"] > 0  # Seed remains valid.
    assert "total_steps" not in result and "steps_per_epoch" not in result


def test_invalid_seed_preserves_native_shapes_but_marks_forward_counts_unknown(images):
    raw = raw_config(images, native=True)
    raw["loop"]["seed"] = "not an integer"
    result = plan(raw)
    assert not result["ok"] and result["images"] == 8
    assert result["native"]["forward_groups"] is None
    assert all(bucket["batches"] is None for bucket in result["buckets"])
    assert result["native"]["sizes"] == len(result["buckets"])
    assert result["image_fit"]["items"]
    assert "total_steps" not in result


def test_rest_plan_previews_uploaded_data_while_validate_and_enqueue_stay_blocked(
    tmp_path, images, monkeypatch
):
    monkeypatch.setattr(routes_core, "gpu_info", lambda: [])
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    try:
        project = client.post("/api/projects", json={"name": "Preview", "family": "toy"}).json()
        response = client.post(
            f"/api/projects/{project['id']}/datasets/upload",
            files=[("files", (path.name, path.read_bytes())) for path in sorted(images.iterdir())],
        )
        assert response.status_code == 200, response.text
        source = response.json()["source"]["path"]
        cfg = raw_config(Path(source))
        cfg["sampling"]["enabled"] = True
        body = {"config": cfg, "project_id": project["id"], "version_id": project["active_version_id"]}
        response = client.post("/api/plan", json=body)
        assert response.status_code == 200, response.text
        preview = response.json()
        assert preview["ok"] is False and preview["images"] == 8 and preview["buckets"]
        assert preview["image_fit"]["items"] and preview["total_steps"] > 0
        validation = client.post("/api/config/validate", json=body).json()
        assert not validation["ok"] and validation["errors"] == preview["errors"]
        jobs_before = client.get("/api/jobs").json()["total"]
        rejected = client.post("/api/jobs", json={**body, "name": "Cannot train"})
        assert rejected.status_code == 400 and rejected.json()["error"]["code"] == "config.invalid"
        assert client.get("/api/jobs").json()["total"] == jobs_before
        cfg["dataset"]["resolution_mode"] = "native"
        cfg["loop"]["seed"] = "bad"
        native = client.post("/api/plan", json={**body, "config": cfg})
        assert native.status_code == 200, native.text
        assert native.json()["native"]["forward_groups"] is None
        assert all(b["batches"] is None for b in native.json()["buckets"])
    finally:
        app.state.regularization.close()
        app.state.dataset_pipeline.close()
        app.state.ctx.versions.close()
        client.close()
        app.state.ctx.db.close()
