"""WD14 input conventions, label selection and truthful local runtime diagnostics."""

import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app
from ypuddin.server import dataset_tagging as tagging


def test_preprocessing_keeps_raw_float_bgr_white_square_and_alpha():
    pixels = tagging.prepare_image(Image.new("RGB", (2, 4), (255, 0, 0)), size=4)
    assert pixels.shape == (1, 4, 4, 3) and pixels.dtype == np.float32 and pixels.flags.c_contiguous
    np.testing.assert_array_equal(pixels[0, 0, 0], [255, 255, 255])
    np.testing.assert_array_equal(pixels[0, 0, 1], [0, 0, 255])
    alpha = tagging.prepare_image(Image.new("RGBA", (2, 2), (255, 0, 0, 128)), size=2)
    np.testing.assert_array_equal(alpha[0, 0, 0], [127, 127, 255])


def test_preprocessing_honors_exif_orientation():
    image = Image.new("RGB", (3, 1), "red")
    image.getexif()[274] = 6
    pixels = tagging.prepare_image(image, size=3)
    np.testing.assert_array_equal(pixels[0, :, 1], [[0, 0, 255]] * 3)
    np.testing.assert_array_equal(pixels[0, :, 0], [[255, 255, 255]] * 3)


def test_label_categories_thresholds_and_output_match():
    labels = (("general", 9), ("blue_eyes", 0), ("low_confidence", 0), ("character_name", 4), ("^_^", 0))
    result = tagging.select_tags(labels, np.array([0.99, 0.7, 0.2, 0.9, 0.5]), 0.35, 0.85)
    assert result == "character name, blue eyes, ^_^"
    with pytest.raises(ValueError, match="match"):
        tagging.select_tags(labels, np.zeros(2), 0.35, 0.85)
    with pytest.raises(ValueError, match="match"):
        tagging.select_tags(labels, np.array([0.9, 0.8, 0.1, float("nan"), 0.2]), 0.35, 0.85)


def test_csv_validation_and_modified_file_invalidates_cache(tmp_path):
    path = tmp_path / "selected_tags.csv"
    path.write_text("name,category\nblue_eyes,0\n")
    assert tagging.read_labels(path) == (("blue_eyes", 0),)
    path.write_text("name,category\nred_hair,0\ncharacter,4\n")
    assert tagging.read_labels(path)[0][0] == "red_hair"
    path.write_text("wrong,columns\na,b\n")
    with pytest.raises(ValueError, match="columns"):
        tagging.read_labels(path)


def test_status_lists_complete_local_models_and_reports_missing_provider(tmp_path, monkeypatch):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    root = app.state.ctx.data_root / "models" / "tagger" / "wd-fixture"
    root.mkdir(parents=True)
    (root / "model.onnx").write_bytes(b"fixture; shape checked only on inference")
    (root / "selected_tags.csv").write_text("name,category\nblue_eyes,0\n")
    monkeypatch.setattr(
        "ypuddin.server.routes_dataset_tagging.runtime_info",
        lambda: {
            "runtime_available": True,
            "runtime_version": "fixture",
            "runtime_providers": ["CPUExecutionProvider"],
            "providers": ["cpu"],
            "runtime_error": None,
        },
    )
    try:
        response = client.get("/api/dataset-tagging/status")
        assert response.status_code == 200, response.text
        status = response.json()
        assert (
            status["available"]
            and status["input_size"] == 448
            and status["models"][0]["model_path"] == str(root / "model.onnx")
        )
        assert "validated by the worker" in status["notes"][1]
        gpu = client.get("/api/dataset-tagging/status?provider=cuda").json()
        assert not gpu["available"] and "cuda" in gpu["errors"][0]
        (root / "selected_tags.csv").unlink()
        missing = client.get("/api/dataset-tagging/status").json()
        assert not missing["available"] and missing["models"] == []
    finally:
        app.state.dataset_pipeline.close()
        app.state.ctx.versions.close()
        client.close()
        app.state.ctx.db.close()


def test_requested_cuda_does_not_silently_fall_back_to_cpu(tmp_path, monkeypatch):
    tags = tmp_path / "selected_tags.csv"
    tags.write_text("name,category\nblue_eyes,0\n")

    class SessionOptions:
        pass

    fake = SimpleNamespace(
        disable_telemetry_events=lambda: None,
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        SessionOptions=SessionOptions,
        InferenceSession=lambda *args, **kwargs: SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"]
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    events = queue.Queue()
    tagging._worker([], {"model_path": "fixture.onnx", "tags_path": str(tags), "provider": "cuda"}, events)
    assert events.get()["type"] == "progress"
    result = events.get()
    assert result["type"] == "error" and "failed to initialize" in result["message"]


def test_runtime_probe_contains_native_session_files_in_temporary_directory(tmp_path, monkeypatch):
    directories = []

    def probe(command, *, cwd, **kwargs):
        directory = Path(cwd)
        directories.append(directory)
        (directory / ":memory:.ses").write_text("native runtime session")
        return SimpleNamespace(
            returncode=0, stdout='{"version":"fixture","providers":["CPUExecutionProvider"]}', stderr=""
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tagging.subprocess, "run", probe)
    tagging._runtime_probe.cache_clear()
    try:
        result = tagging._runtime_probe(("fixture",))
        assert result["providers"] == ["CPUExecutionProvider"]
        assert directories and not directories[0].exists()
        assert not (tmp_path / ":memory:.ses").exists()
    finally:
        tagging._runtime_probe.cache_clear()
