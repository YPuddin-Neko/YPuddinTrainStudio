"""Actual streamed HTTP fixtures exercise download publication, cancellation and registry defaults."""

import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save

from ypuddin.server.app import create_app
from ypuddin.server.model_downloads import (
    ModelDownload,
    ModelDownloadRequest,
    ModelDownloads,
    _Redirect,
    resolve_source,
)


@pytest.fixture
def download_service(tmp_path):
    payload = save({"x_embedder.proj.1.weight": torch.zeros(320000)})
    control = SimpleNamespace(
        mode="ok", payload=payload, started=threading.Event(), release=threading.Event()
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html>not model weights</html>" if control.mode == "invalid" else control.payload
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) + (100 if control.mode == "short" else 0)))
            self.end_headers()
            control.started.set()
            if control.mode == "race":
                control.release.wait(5)
            try:
                for offset in range(0, len(body), 16384):
                    self.wfile.write(body[offset : offset + 16384])
                    self.wfile.flush()
                    if control.mode == "slow":
                        time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    app = create_app(tmp_path / "data")

    class LocalOpener:
        def open(self, _request, timeout):
            return urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/weights", timeout=timeout)

    app.state.model_downloads.opener = LocalOpener()
    with TestClient(app) as client:
        root = tmp_path / "selected-model-directory"
        client.put("/api/settings", json={"paths": {"models_dir": str(root)}}).raise_for_status()
        yield client, control, root, app
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def start(client, filename="model.safetensors", **extra):
    return client.post(
        "/api/models/downloads",
        json={
            "family": "anima",
            "kind": "dit",
            "repo_id": "example/model",
            "filename": filename,
            **extra,
        },
    )


def wait_for(client, id_, predicate=lambda row: row["status"] not in {"queued", "downloading"}):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        row = next(row for row in client.get("/api/models/downloads").json() if row["id"] == id_)
        if predicate(row):
            return row
        time.sleep(0.02)
    pytest.fail(f"download did not reach expected state: {row}")


def test_download_registers_complete_file_in_configured_directory_and_defaults(download_service):
    client, control, root, _app = download_service
    response = start(client)
    assert response.status_code == 202
    row = wait_for(client, response.json()["id"])
    assert row["status"] == "completed", row
    path = Path(row["target_path"])
    assert path.is_relative_to(root) and path.read_bytes() == control.payload
    assert row["downloaded_bytes"] == row["total_bytes"] == len(control.payload)
    assets = client.get("/api/models").json()
    assert len(assets) == 1 and assets[0]["is_default"] and assets[0]["exists"]
    assert assets[0]["id"] == row["model_id"]
    duplicate = start(client)
    assert duplicate.status_code == 409 and path.read_bytes() == control.payload
    second = wait_for(client, start(client, "another.safetensors").json()["id"])
    assets = client.get("/api/models").json()
    assert [m["id"] for m in assets if m["is_default"]] == [second["model_id"]]
    client.patch(f"/api/models/{row['model_id']}", json={"is_default": True}).raise_for_status()
    assert [m["id"] for m in client.get("/api/models").json() if m["is_default"]] == [row["model_id"]]
    assert not list((root / ".downloads").rglob("*.safetensors"))


@pytest.mark.parametrize("mode", ["invalid", "short"])
def test_failed_download_never_appears_as_a_model(download_service, mode):
    client, control, root, _app = download_service
    control.mode = mode
    row = wait_for(client, start(client).json()["id"])
    assert row["status"] == "failed" and row["error"]
    assert not Path(row["target_path"]).exists()
    assert client.get("/api/models").json() == []
    assert not list(root.rglob("*.safetensors"))


def test_progress_and_cancellation_do_not_register_partial_weights(download_service):
    client, control, root, _app = download_service
    control.mode = "slow"
    id_ = start(client).json()["id"]
    progress = wait_for(client, id_, lambda row: row["downloaded_bytes"] > 0)
    assert 0 < progress["downloaded_bytes"] < progress["total_bytes"]
    assert start(client).status_code == 409
    client.post(f"/api/models/downloads/{id_}/cancel", json={}).raise_for_status()
    row = wait_for(client, id_)
    assert row["status"] == "cancelled" and client.get("/api/models").json() == []
    assert not list(root.rglob("*.safetensors"))


def test_download_never_overwrites_a_file_created_during_transfer(download_service):
    client, control, _root, _app = download_service
    control.mode = "race"
    row = start(client).json()
    assert control.started.wait(2)
    target = Path(row["target_path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"owned by user")
    control.release.set()
    finished = wait_for(client, row["id"])
    assert finished["status"] == "failed" and target.read_bytes() == b"owned by user"
    assert client.get("/api/models").json() == []


def test_download_rejects_recognisable_wrong_component(download_service):
    client, _control, _root, _app = download_service
    row = wait_for(client, start(client, kind="text_encoder").json()["id"])
    assert row["status"] == "failed" and "looks like anima dit" in row["error"]
    assert client.get("/api/models").json() == []


@pytest.mark.parametrize(
    "filename",
    [
        "../model.safetensors",
        "/model.safetensors",
        "folder/../model.safetensors",
        "C:\\model.safetensors",
        "CON.safetensors",
        "model-00001-of-00002.safetensors",
        "model.py",
    ],
)
def test_download_path_validation(filename):
    with pytest.raises(ValueError):
        ModelDownloadRequest(family="anima", kind="dit", repo_id="owner/repo", filename=filename)


def test_hf_file_url_normalization_and_redirect_credentials():
    body = ModelDownloadRequest(
        family="anima",
        kind="dit",
        url="https://huggingface.co/owner/repo/blob/main/sub/model.safetensors?download=true",
    )
    assert resolve_source(body) == (
        "https://huggingface.co/owner/repo/resolve/main/sub/model.safetensors",
        "model.safetensors",
    )
    with pytest.raises(ValueError):
        ModelDownloadRequest(family="anima", kind="dit", url="http://127.0.0.1/file.safetensors")
    request = urllib.request.Request(resolve_source(body)[0], headers={"Authorization": "Bearer private"})
    redirected = _Redirect().redirect_request(
        request, None, 302, "redirect", {}, "https://cdn.example.com/signed"
    )
    assert redirected.get_header("Authorization") is None


def test_missing_model_cannot_be_default_and_duplicate_registration_is_idempotent(download_service):
    client, _control, root, _app = download_service
    missing = client.post(
        "/api/models",
        json={
            "family": "anima",
            "kind": "dit",
            "path": str(root / "missing.safetensors"),
            "is_default": True,
        },
    )
    assert missing.status_code == 404
    row = wait_for(client, start(client).json()["id"])
    again = client.post(
        "/api/models", json={"family": "anima", "kind": "dit", "path": row["target_path"]}
    ).json()
    assert again["id"] == row["model_id"] and again["is_default"]
    Path(row["target_path"]).unlink()
    assert client.patch(f"/api/models/{row['model_id']}", json={"is_default": True}).status_code == 404


def test_restart_marks_interrupted_download_failed_and_cleans_its_staging_only(tmp_path):
    app = create_app(tmp_path / "data")
    context = app.state.ctx
    app.state.model_downloads.close()
    root = Path(context.settings()["paths"]["models_dir"])
    id_ = "dl_0123456789ab"
    target = root / "anima" / "dit" / "asset" / "model.safetensors"
    staged = root / ".downloads" / id_ / target.name
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"partial")
    user_file = root / "keep.safetensors"
    user_file.write_bytes(b"user file")
    row = ModelDownload(
        id=id_,
        family="anima",
        kind="dit",
        source_url="https://huggingface.co/owner/repo/resolve/main/model.safetensors",
        filename=target.name,
        target_path=str(target),
        status="downloading",
        created_at=time.time(),
    ).model_dump()
    context.db.set_kv("model_downloads", [row])
    restarted = ModelDownloads(context)
    try:
        assert restarted.list()[0]["status"] == "failed"
        assert "restarted" in restarted.list()[0]["error"]
        assert not staged.exists() and user_file.read_bytes() == b"user file"
        assert context.db.fetchall("SELECT * FROM models") == []
    finally:
        restarted.close()
        context.db.close()
