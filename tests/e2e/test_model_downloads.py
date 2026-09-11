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
        mode="ok", payload=payload, started=threading.Event(), release=threading.Event(), requests=[]
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
            control.requests.append(_request)
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


def test_credentials_are_write_only_persistent_and_clear_overrides_environment(download_service, monkeypatch):
    client, control, _root, app = download_service
    from ypuddin.server.model_credentials import ModelCredentials

    monkeypatch.setenv("HF_TOKEN", "hf_environment_secret")
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "ms_environment_secret")
    assert client.get("/api/models/credentials").json() == {
        "huggingface": {"configured": True},
        "modelscope": {"configured": True},
    }
    for provider, token in [("huggingface", "hf_saved_secret"), ("modelscope", "ms_saved_secret")]:
        saved = client.put(f"/api/models/credentials/{provider}", json={"token": token})
        assert saved.status_code == 200 and saved.json() == {"configured": True}
        assert ModelCredentials(app.state.ctx.data_root).token(provider) == token
        row = wait_for(client, start(client, f"{provider}.safetensors", provider=provider).json()["id"])
        assert row["status"] == "completed", row
        request = control.requests[-1]
        if provider == "huggingface":
            assert request.get_header("Authorization") == f"Bearer {token}"
            assert request.get_header("Cookie") is None
        else:
            assert request.get_header("Cookie") == f"m_session_id={token}"
            assert request.get_header("Authorization") is None
            assert request.full_url.startswith("https://modelscope.cn/api/v1/models/")
            assert "Revision=master" in request.full_url
        for endpoint in ["/api/settings", "/api/models/credentials", "/api/models/downloads"]:
            assert token not in client.get(endpoint).text
        assert token not in str(app.state.ctx.db.get_kv("model_downloads", []))
        cleared = client.delete(f"/api/models/credentials/{provider}")
        assert cleared.json() == {"configured": False}
        assert ModelCredentials(app.state.ctx.data_root).token(provider) is None
    assert client.get("/api/models/credentials").json() == {
        "huggingface": {"configured": False},
        "modelscope": {"configured": False},
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"token": "hf_secret\nHeader:bad"},
        {"token": "secret;bad"},
        {"token": {"unexpected": "nested_secret"}},
        {"token": "valid_secret", "extra": "other_secret"},
    ],
)
def test_credential_validation_does_not_echo_any_input(download_service, payload):
    import json

    client, _control, _root, _app = download_service
    response = client.put("/api/models/credentials/huggingface", json=payload)
    assert response.status_code == 422
    assert "secret" not in response.text
    assert "input" not in response.json()["error"]["details"]["errors"][0]
    assert json.dumps(payload) not in response.text


def test_mirror_is_explicit_and_anonymous(download_service):
    client, control, _root, _app = download_service
    client.put(
        "/api/models/credentials/huggingface", json={"token": "hf_must_not_reach_mirror"}
    ).raise_for_status()
    row = wait_for(client, start(client, mirror="hf-mirror").json()["id"])
    assert row["status"] == "completed", row
    request = control.requests[-1]
    assert request.full_url.startswith("https://hf-mirror.com/")
    assert request.get_header("Authorization") is None and request.get_header("Cookie") is None
    assert start(client, provider="modelscope", mirror="hf-mirror").status_code == 422


def test_modelscope_url_and_cross_origin_redirect_strip_all_credentials():
    body = ModelDownloadRequest(
        family="anima",
        kind="vae",
        provider="modelscope",
        url="https://modelscope.cn/api/v1/models/owner/repo/repo?Revision=master&FilePath=sub%2Fmodel.safetensors",
    )
    url, filename = resolve_source(body)
    assert filename == "model.safetensors"
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query) == {
        "Revision": ["master"],
        "FilePath": ["sub/model.safetensors"],
    }
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer secret",
            "Cookie": "m_session_id=secret",
            "Proxy-Authorization": "secret",
            "User-Agent": "studio",
        },
    )
    redirected = _Redirect().redirect_request(
        request, None, 302, "redirect", {}, "https://cdn.example.com/signed"
    )
    assert {k.lower() for k in redirected.headers} == {"user-agent"}
    same = _Redirect().redirect_request(request, None, 302, "redirect", {}, "https://modelscope.cn/another")
    assert same.get_header("Cookie") == "m_session_id=secret"
    for bad in ["http://cdn.example.com/file", "https://secret@cdn.example.com/file"]:
        with pytest.raises(ValueError):
            _Redirect().redirect_request(request, None, 302, "redirect", {}, bad)
    with pytest.raises(ValueError):
        ModelDownloadRequest(family="anima", kind="vae", provider="modelscope", url=url + "&token=secret")


def test_retry_preserves_provider_and_uses_current_credentials(download_service):
    client, control, _root, _app = download_service
    control.mode = "invalid"
    failed = wait_for(client, start(client, provider="modelscope").json()["id"])
    assert failed["status"] == "failed"
    client.put("/api/models/credentials/modelscope", json={"token": "new_ms_secret"}).raise_for_status()
    control.mode = "ok"
    retried = client.post(f"/api/models/downloads/{failed['id']}/retry", json={})
    assert retried.status_code == 202 and retried.json()["id"] != failed["id"]
    row = wait_for(client, retried.json()["id"])
    assert row["status"] == "completed" and row["provider"] == "modelscope"
    assert control.requests[-1].get_header("Cookie") == "m_session_id=new_ms_secret"
    assert client.post(f"/api/models/downloads/{row['id']}/retry", json={}).status_code == 409


def test_download_errors_redact_saved_token_and_auth_failure_guides_user(download_service):
    client, _control, _root, app = download_service
    secret = "hf_never_echo_this"
    client.put("/api/models/credentials/huggingface", json={"token": secret}).raise_for_status()

    class ErrorOpener:
        def open(self, request, timeout):
            raise ValueError(f"transport failed with {secret}")

    app.state.model_downloads.opener = ErrorOpener()
    row = wait_for(client, start(client).json()["id"])
    assert secret not in str(row) and "[redacted]" in row["error"]

    class Unauthorized:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, secret, {}, None)

    app.state.model_downloads.opener = Unauthorized()
    row = wait_for(client, start(client, "gated.safetensors").json()["id"])
    assert secret not in str(row) and "Credentials" in row["error"] and "license" in row["error"]


def test_catalog_bundle_is_atomic_checks_both_hashes_and_never_training_default(
    download_service, monkeypatch
):
    import hashlib
    import io

    from ypuddin.server import model_downloads as module

    client, _control, _root, app = download_service
    payloads = {
        "model.onnx": b"controlled model fixture",
        "selected_tags.csv": b"tag_id,name,category\n1,fox,0\n",
    }
    monkeypatch.setattr(
        module,
        "TAGGER_FILES",
        {
            name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()
        },
    )

    class BundleOpener:
        corrupt = True

        def open(self, request, timeout):
            name = request.full_url.rsplit("/", 1)[-1]
            data = payloads[name]
            if self.corrupt and name.endswith("csv"):
                data = b"X" * len(data)
            stream = io.BytesIO(data)
            stream.status = 200
            stream.headers = {"Content-Length": str(len(data))}
            return stream

    opener = BundleOpener()
    app.state.model_downloads.opener = opener
    catalog = client.get("/api/models/catalog").json()[0]
    assert catalog["role"] == "tagger" and not catalog["ready"]
    response = client.post(f"/api/models/catalog/{catalog['id']}/download", json={})
    assert response.status_code == 202, response.text
    failed = wait_for(client, response.json()["id"])
    assert failed["status"] == "failed" and "selected_tags.csv" in failed["error"]
    assert not Path(catalog["path"]).exists() and client.get("/api/models").json() == []
    opener.corrupt = False
    row = wait_for(client, client.post(f"/api/models/downloads/{failed['id']}/retry", json={}).json()["id"])
    assert row["status"] == "completed", row
    catalog = client.get("/api/models/catalog").json()[0]
    assert catalog["ready"]
    assert all((Path(catalog["path"]) / name).read_bytes() == data for name, data in payloads.items())
    asset = client.get("/api/models").json()[0]
    assert asset["kind"] == asset["family"] == "tagger" and not asset["is_default"]
    assert client.patch(f"/api/models/{asset['id']}", json={"is_default": True}).status_code == 400
