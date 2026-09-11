"""Opt-in real WD14 CPU test; never downloads weights or sends images to a network service.

YPUDDIN_TEST_WD14_MODEL points to a local folder containing the two official files.
"""

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ypuddin.server import create_app, routes_work

MODEL = os.environ.get("YPUDDIN_TEST_WD14_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL, reason="set YPUDDIN_TEST_WD14_MODEL to run real local WD14 inference"
)


def test_real_local_tagger_missing_append_overwrite_cancel_and_undo(tmp_path, monkeypatch):
    from ypuddin.server.dataset_tagging import runtime_info

    if not runtime_info()["runtime_available"]:
        pytest.skip("install ONNX Runtime to run the real tagging test")
    model = Path(MODEL)
    source = tmp_path / "originals"
    source.mkdir()
    image = Image.new("RGB", (320, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 50, 260, 280), fill=(242, 202, 175), outline=(20, 20, 30), width=3)
    draw.ellipse((55, 35, 265, 120), fill=(50, 35, 50))
    draw.ellipse((95, 125, 125, 160), fill=(40, 90, 160))
    draw.ellipse((195, 125, 225, 160), fill=(40, 90, 160))
    draw.arc((115, 145, 205, 210), 0, 180, fill=(110, 50, 60), width=3)
    draw.polygon([(60, 448), (100, 260), (220, 260), (260, 448)], fill=(80, 130, 210))
    image.save(source / "a.png")
    image.save(source / "b.png")
    (source / "a.txt").write_text("curated original caption")
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Real WD14 fixture"}).json()
    dataset = client.post(f"/api/projects/{project['id']}/datasets", json={"path": str(source)}).json()[
        "source"
    ]
    managed = Path(dataset["path"])
    url = f"/api/projects/{project['id']}/versions/{project['active_version_id']}/pipeline/operations"
    options = {
        "model_path": str(model / "model.onnx"),
        "tags_path": str(model / "selected_tags.csv"),
        "provider": "cpu",
        "trigger_word": "fixture_subject",
    }

    def await_terminal(response):
        assert response.status_code == 202, response.text
        oid = response.json()["id"]
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            result = client.get(f"/api/dataset-pipeline/operations/{oid}").json()
            if (
                result["status"] in ("completed", "failed", "cancelled")
                and not app.state.ctx.resolve_version(project["id"], project["active_version_id"])["busy"]
            ):
                return result
            time.sleep(0.025)
        pytest.fail("real tagger did not finish")

    def run(mode, names):
        response = client.post(
            url,
            json={
                "action": "tag",
                "images": [{"dataset_id": dataset["id"], "rel_path": name} for name in names],
                "tagging": {**options, "mode": mode},
            },
        )
        result = await_terminal(response)
        assert result["status"] == "completed", result
        return result

    def undo(op):
        result = await_terminal(
            client.post(url, json={"action": "restore", "restore_operation_id": op["id"]})
        )
        assert result["status"] == "completed", result

    try:
        status = client.get(
            "/api/dataset-tagging/status",
            params={"model_path": str(model / "model.onnx"), "tags_path": str(model / "selected_tags.csv")},
        ).json()
        assert status["available"], status
        missing = run("missing", ["a.png", "b.png"])
        assert (managed / "a.txt").read_text() == "curated original caption"
        generated = (managed / "b.txt").read_text()
        assert generated.startswith("fixture_subject, ") and len(generated.split(",")) > 2
        undo(missing)
        assert not (managed / "b.txt").exists()
        appended = run("append", ["a.png"])
        assert (
            "curated original caption" in (managed / "a.txt").read_text()
            and "fixture_subject" in (managed / "a.txt").read_text()
        )
        undo(appended)
        assert (managed / "a.txt").read_text() == "curated original caption"
        replaced = run("overwrite", ["a.png"])
        assert "curated original caption" not in (managed / "a.txt").read_text()
        undo(replaced)
        assert (managed / "a.txt").read_text() == "curated original caption"
        response = client.post(
            url,
            json={
                "action": "tag",
                "images": [{"dataset_id": dataset["id"], "rel_path": "b.png"}],
                "tagging": {**options, "mode": "missing"},
            },
        )
        oid = response.json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            op = client.get(f"/api/dataset-pipeline/operations/{oid}").json()
            if any("Loading local WD14 model" in log["message"] for log in op["logs"]):
                break
            time.sleep(0.01)
        else:
            pytest.fail("worker did not begin loading")
        started = time.monotonic()
        assert client.post(f"/api/dataset-pipeline/operations/{oid}/cancel").status_code == 200
        cancelled = await_terminal(response)
        assert cancelled["status"] == "cancelled" and time.monotonic() - started < 6, cancelled
        assert (
            not (managed / "b.txt").exists() and (managed / "a.txt").read_text() == "curated original caption"
        )
        assert not (source / "b.txt").exists()
    finally:
        app.state.dataset_pipeline.close()
        app.state.ctx.versions.close()
        client.close()
        app.state.ctx.db.close()
