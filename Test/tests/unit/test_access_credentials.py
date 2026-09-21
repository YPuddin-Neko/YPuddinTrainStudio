"""Central access keys use isolated fake secrets; no real key file or network is accessed."""

import base64
import json
import os
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from Test.tests.unit.test_regularization import Provider, finished, start
from ypuddin.server import create_app
from ypuddin.server.model_credentials import DanbooruCredentialUpdate, ModelCredentials


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Access keys"}).json()
    yield client, app, project["id"], project["active_version_id"]
    for manager in (
        app.state.regularization,
        app.state.dataset_pipeline,
        app.state.environment,
        app.state.model_downloads,
        app.state.ctx.versions,
    ):
        manager.close()
    client.close()
    app.state.ctx.db.close()


@pytest.fixture(autouse=True)
def isolate_cli_keys(monkeypatch):
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)


def test_central_model_aliases_and_site_states_preserve_unrelated_secrets(api):
    client, app, _, _ = api
    store = app.state.model_downloads.credentials
    store.path.write_text(json.dumps({"other_setting": {"keep": True}}))
    payloads = {
        "huggingface": {"token": "hf_fake_key"},
        "modelscope": {"token": "ms_fake_key"},
        "danbooru": {"username": "test_user", "api_key": "dan_fake_key"},
        "gelbooru": {"user_id": "1234", "api_key": "gel_fake_key"},
    }
    for provider, body in payloads.items():
        result = client.put(f"/api/credentials/{provider}", json=body)
        assert result.status_code == 200 and result.json() == {"configured": True}, result.text
    response = client.get("/api/credentials")
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {p: {"configured": True} for p in payloads}
    assert client.get("/api/models/credentials").json() == {
        p: {"configured": True} for p in ("huggingface", "modelscope")
    }
    assert app.state.regularization.credentials is store
    reopened = ModelCredentials(store.path.parent)
    assert reopened.site("danbooru") == ("test_user", "dan_fake_key")
    assert reopened.site("gelbooru") == ("1234", "gel_fake_key")
    assert reopened.token("huggingface") == "hf_fake_key"
    assert json.loads(store.path.read_text())["other_setting"] == {"keep": True}
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600
    for endpoint in ("/api/credentials", "/api/models/credentials", "/api/settings", "/api/models/downloads"):
        text = client.get(endpoint).text
        assert all(
            secret not in text
            for secret in ("hf_fake_key", "ms_fake_key", "dan_fake_key", "gel_fake_key", "test_user", "1234")
        )
    for provider in payloads:
        assert client.delete(f"/api/credentials/{provider}").json() == {"configured": False}
    assert client.get("/api/credentials").json() == {p: {"configured": False} for p in payloads}
    assert reopened.site("danbooru") == ("", "") and reopened.site("gelbooru") == ("", "")
    assert not any("fake_key" in str(value) for value in json.loads(store.path.read_text()).values())


@pytest.mark.parametrize("source", ["danbooru", "gelbooru"])
def test_regularization_uses_stored_site_snapshot_and_clear_affects_future_tasks(api, source):
    client, app, _, _ = api
    account = "username" if source == "danbooru" else "user_id"
    assert (
        client.put(
            f"/api/credentials/{source}", json={account: "1234", "api_key": "stored_fake_key"}
        ).status_code
        == 200
    )
    provider = Provider(source)
    provider.pause = True
    app.state.regularization.opener = provider
    response = start(api, source=source, prompt="dog")  # No credentials in task request.
    assert response.status_code == 202, response.text
    oid = response.json()["id"]
    assert provider.entered.wait(5)
    assert client.delete(f"/api/credentials/{source}").status_code == 200
    provider.release.set()
    task = finished(api, oid)
    assert task["status"] == "completed", task
    request = provider.calls[0]
    if source == "gelbooru":
        assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)["api_key"] == [
            "stored_fake_key"
        ]
    else:
        assert (
            request.get_header("Authorization")
            == "Basic " + base64.b64encode(b"1234:stored_fake_key").decode()
        )
    assert all(
        not r.get_header("Authorization") and "stored_fake_key" not in r.full_url for r in provider.calls[1:]
    )
    assert "stored_fake_key" not in json.dumps(app.state.regularization._row(oid))
    assert "stored_fake_key" not in client.get(f"/api/regularization/{oid}").text
    provider.colors = ("yellow", "green")
    provider.calls.clear()
    response = start(api, source=source, prompt="dog")
    if source == "gelbooru":
        assert response.status_code == 422 and "Settings" in response.text
        assert not provider.calls
    else:
        assert response.status_code == 202
        assert finished(api, response.json()["id"])["status"] == "completed"
        assert not provider.calls[0].get_header("Authorization")


@pytest.mark.parametrize(
    "provider,body",
    [
        ("danbooru", {"username": "user:secret", "api_key": "fake_secret"}),
        ("danbooru", {"username": "ok", "api_key": "fake_secret\nInjected"}),
        ("gelbooru", {"user_id": "fake_secret", "api_key": "valid_fake_secret"}),
        ("gelbooru", {"user_id": "1", "api_key": {"nested": "fake_secret"}}),
        ("huggingface", {"token": "fake_secret;bad"}),
        ("modelscope", {"token": "valid_fake_secret", "unexpected": "extra_fake_secret"}),
        ("modelscope", {"token": "valid", "fake_secret_property": "value"}),
    ],
)
def test_central_validation_never_echoes_secrets(api, provider, body):
    response = api[0].put(f"/api/credentials/{provider}", json=body)
    assert response.status_code == 422
    assert "secret" not in response.text
    assert all("input" not in e and "ctx" not in e for e in response.json()["error"]["details"]["errors"])
    assert not api[1].state.model_downloads.credentials.path.exists()


def test_write_failure_preserves_previous_keys_and_cleans_temporary_file(api, monkeypatch):
    client, app, _, _ = api
    store = app.state.model_downloads.credentials
    assert (
        client.put(
            "/api/credentials/danbooru", json={"username": "test_user", "api_key": "old_fake_key"}
        ).status_code
        == 200
    )

    def fail(*args):
        raise OSError("disk error with new_fake_key")

    monkeypatch.setattr("ypuddin.server.model_credentials.os.replace", fail)
    response = client.put(
        "/api/credentials/danbooru", json={"username": "new_user", "api_key": "new_fake_key"}
    )
    assert response.status_code == 503 and "new_fake_key" not in response.text
    assert store.site("danbooru") == ("test_user", "old_fake_key")
    assert not list(store.path.parent.glob(".secrets-*"))


def test_parallel_site_and_model_updates_do_not_lose_each_other(tmp_path):
    store = ModelCredentials(tmp_path)
    barrier = threading.Barrier(2)

    def site():
        barrier.wait()
        store.save_site("danbooru", DanbooruCredentialUpdate(username="test_user", api_key="fake_site"))

    def model():
        barrier.wait()
        store.save("huggingface", "fake_model")

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(site), pool.submit(model)]
        for future in futures:
            future.result()
    assert store.site("danbooru") == ("test_user", "fake_site")
    assert store.token("huggingface") == "fake_model"


def test_corrupt_site_entry_fails_closed_without_echoing_file_or_request(api):
    client, app, _, _ = api
    app.state.model_downloads.credentials.path.write_text(
        '{"site_sources":{"gelbooru":{"api_key":"corrupt_secret","user_id":null}}}'
    )
    response = client.get("/api/credentials")
    assert response.status_code == 503 and "corrupt_secret" not in response.text
    response = start(api, source="gelbooru", prompt="dog")
    assert response.status_code == 503 and "corrupt_secret" not in response.text
    assert app.state.ctx.db.fetchall("SELECT * FROM regularization_operations") == []
