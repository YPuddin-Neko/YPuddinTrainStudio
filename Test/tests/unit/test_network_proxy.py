"""Outbound policy contracts: hot reload, private credentials, real proxy traffic and child logs."""

import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ypuddin.server import create_app
from ypuddin.server.environment import Installer
from ypuddin.server.network import (
    PASSWORD_REVISION,
    PROXY_KEYS,
    ProxyCredentials,
    ProxyPolicy,
    validate_proxy_settings,
)


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    yield client, app.state
    client.close()
    app.state.ctx.db.close()


def test_settings_roundtrip_never_returns_or_publishes_password(api):
    client, state = api
    assert client.get("/api/settings").json()["network"]["proxy_mode"] == "system"
    patch = {"proxy_mode": "custom", "proxy_url": "http://127.0.0.1:7890/", "proxy_username": "alice", "proxy_password": "secret/@123"}
    result = client.put("/api/settings", json={"network": patch})
    assert result.status_code == 200, result.text
    assert "secret/@123" not in result.text
    assert result.json()["network"] == {
        "proxy_mode": "custom", "proxy_url": "http://127.0.0.1:7890", "proxy_username": "alice", "proxy_password_configured": True,
    }
    assert "secret/@123" not in state.ctx.settings_path.read_text()
    assert "secret/@123" not in client.get("/api/settings").text
    assert PASSWORD_REVISION not in client.get("/api/settings").json()
    stored = ProxyCredentials(state.ctx.data_root)
    assert stored.password() == "secret/@123"
    if os.name != "nt":
        assert stored.path.stat().st_mode & 0o077 == 0
    client.put("/api/settings", json={"network": {"proxy_mode": "direct"}})
    assert stored.password() == "secret/@123"
    client.put("/api/settings", json={"network": {"proxy_password": ""}})
    assert not client.get("/api/settings").json()["network"]["proxy_password_configured"]
    assert "secret/@123" not in stored.path.read_text()


@pytest.mark.parametrize("boundary", ["private", "public"])
def test_failed_proxy_save_keeps_address_and_password_consistent(api, monkeypatch, boundary):
    _, state = api
    c = state.ctx
    c.save_settings({"network": {"proxy_mode": "custom", "proxy_url": "http://old.invalid:8080", "proxy_password": "old-private"}})
    original = c.settings_path.read_bytes()
    credentials = ProxyCredentials(c.data_root)
    replace = Path.replace

    def fail(self, target):
        if Path(target) == (credentials.path if boundary == "private" else c.settings_path):
            raise OSError("simulated disk failure")
        return replace(self, target)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="disk failure"):
            c.save_settings({"network": {"proxy_url": "http://new.invalid:8080", "proxy_password": "new-private"}})
    assert c.settings_path.read_bytes() == original
    assert ProxyPolicy.from_context(c).url == "http://old.invalid:8080"
    # Read from disk using a new object, not an in-memory rollback cache.
    assert ProxyCredentials(c.data_root).password() == "old-private"
    c.save_settings({"network": {"proxy_url": "http://final.invalid:8080", "proxy_password": "final-private"}})
    assert ProxyPolicy.from_context(c).url == "http://final.invalid:8080"
    assert credentials.password() == "final-private"
    assert len(json.loads(credentials.path.read_text())["passwords"]) == 1


def test_interrupted_preparation_and_legacy_credentials_recover_on_next_save(api):
    client, state = api
    c = state.ctx
    credentials = ProxyCredentials(c.data_root)
    credentials.path.parent.mkdir(parents=True)
    credentials.path.write_text(json.dumps({"proxy_password": "legacy-private"}))
    # Process exit here leaves the old settings pointer active after restart too.
    credentials.prepare("uncommitted-private", "", "pending-revision")
    assert ProxyCredentials(c.data_root).password() == "legacy-private"
    assert PASSWORD_REVISION not in c.settings()
    c.save_settings({"network": {"proxy_password": "committed-private"}})
    assert credentials.password() == "committed-private"
    assert "uncommitted-private" not in credentials.path.read_text()
    result = client.put("/api/settings", json={PASSWORD_REVISION: "pending-revision"})
    assert result.status_code == 400
    assert credentials.password() == "committed-private"


def test_failed_post_commit_cleanup_does_not_report_a_committed_save_as_failed(api, monkeypatch):
    _, state = api
    credentials = ProxyCredentials(state.ctx.data_root)
    state.ctx.save_settings({"network": {"proxy_password": "old-private"}})
    with monkeypatch.context() as scoped:
        scoped.setattr(ProxyCredentials, "finish", lambda *_: (_ for _ in ()).throw(OSError("disk failure")))
        result = state.ctx.save_settings({"network": {"proxy_password": ""}})
    assert not result["network"]["proxy_password_configured"]
    assert credentials.password() == ""
    state.ctx.save_settings({"ui": {"theme": "dark"}})
    assert "old-private" not in credentials.path.read_text()


@pytest.mark.parametrize("url", ["socks5://localhost:1080", "http://alice:private-secret@localhost:7890", "https://host:0", "https://host:99999", "https://host/path", "http://host?key=secret", "http://host/#x", "http://ho st:80", "http://[bad"])
def test_bad_proxy_url_is_rejected_without_leaking_pasted_credentials(api, url):
    client, _ = api
    result = client.put("/api/settings", json={"network": {"proxy_mode": "custom", "proxy_url": url}})
    assert result.status_code == 400, result.text
    assert url not in result.text and "private-secret" not in result.text
    assert client.get("/api/settings").json()["network"]["proxy_mode"] == "system"


def test_bad_password_or_mode_does_not_mutate_saved_secret(api):
    client, state = api
    for patch in ({"proxy_password": "bad\nsecret"}, {"proxy_mode": "bad", "proxy_password": "not-saved"}):
        assert client.put("/api/settings", json={"network": patch}).status_code == 400
    assert not ProxyCredentials(state.ctx.data_root).path.exists()
    with pytest.raises(ValueError, match="requires"):
        validate_proxy_settings({"proxy_mode": "custom"})


def test_configured_state_is_derived_from_committed_secret_not_client_patch(api):
    client, _ = api
    fake = client.put("/api/settings", json={"network": {"proxy_password_configured": True}})
    assert not fake.json()["network"]["proxy_password_configured"]
    client.put("/api/settings", json={"network": {"proxy_password": "private-secret"}})
    stale = client.put("/api/settings", json={"network": {"proxy_password_configured": False}})
    assert stale.json()["network"]["proxy_password_configured"]


def test_proxy_policy_snapshots_reload_after_save_and_do_not_mutate_process(api, monkeypatch):
    _, state = api
    monkeypatch.setenv("HTTPS_PROXY", "http://inherited:3128")
    monkeypatch.setenv("ALL_PROXY", "http://fallback:1080")
    monkeypatch.setenv("hTtP_PrOxY", "http://mixed-case:1080")
    monkeypatch.setenv("NO_PROXY", "internal.example")
    monkeypatch.setenv("no_proxy", "localhost")
    original = dict(os.environ)
    system = ProxyPolicy.from_context(state.ctx)
    assert system.subprocess_env()["HTTPS_PROXY"] == "http://inherited:3128"
    state.ctx.save_settings({"network": {"proxy_mode": "direct"}})
    direct = ProxyPolicy.from_context(state.ctx)
    env = direct.subprocess_env()
    assert not any(key in env for key in PROXY_KEYS)
    assert "hTtP_PrOxY" not in env
    assert "internal.example" in env["NO_PROXY"] and "127.0.0.1" in env["NO_PROXY"]
    state.ctx.save_settings({"network": {"proxy_mode": "custom", "proxy_url": "https://[::1]:8443", "proxy_username": "a@b", "proxy_password": "p:/@"}})
    custom = ProxyPolicy.from_context(state.ctx)
    assert custom.subprocess_env()["HTTPS_PROXY"] == "https://a%40b:p%3A%2F%40@[::1]:8443"
    assert system.mode == "system" and direct.mode == "direct"
    assert dict(os.environ) == original


def test_real_http_request_uses_custom_proxy_and_keeps_destination_handlers(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Proxy-Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"proxied")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("no_proxy", "")
    try:
        policy = ProxyPolicy("custom", f"http://127.0.0.1:{server.server_port}", "alice", "private")
        redirect = urllib.request.HTTPRedirectHandler()
        opener = policy.opener(redirect)
        assert redirect in opener.handlers
        with opener.open("http://example.invalid/model.whl", timeout=5) as response:
            assert response.read() == b"proxied"
        assert requests == [("http://example.invalid/model.whl", "Basic YWxpY2U6cHJpdmF0ZQ==")]
        assert not any(isinstance(handler, urllib.request.ProxyHandler) for handler in ProxyPolicy("direct").opener().handlers)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_installer_real_child_uses_latest_policy_and_redacts_output(api, monkeypatch, tmp_path):
    _, state = api
    monkeypatch.setenv("PIP_INDEX_URL", "http://ignored.invalid/simple")
    state.ctx.save_settings({"network": {"proxy_mode": "custom", "proxy_url": "http://proxy.invalid:80", "proxy_username": "alice", "proxy_password": "long-private-secret"}})
    runner = Installer(tmp_path, context=state.ctx)
    logs = []
    script = "import os,json; print(json.dumps({k:os.environ.get(k) for k in ['HTTP_PROXY','PIP_INDEX_URL','PIP_CONFIG_FILE']})); print('long-private-secret')"
    runner.run([sys.executable, "-c", script], logs.append, threading.Event())
    output = "\n".join(logs)
    assert "long-private-secret" not in output and "alice:" not in output
    env = json.loads(logs[-2])
    assert env["HTTP_PROXY"] == "http://***@proxy.invalid:80"
    assert env["PIP_INDEX_URL"] is None and env["PIP_CONFIG_FILE"] == os.devnull
    state.ctx.save_settings({"network": {"proxy_mode": "direct"}})
    logs.clear()
    runner.run([sys.executable, "-c", "import os; print(os.environ.get('HTTP_PROXY','DIRECT'))"], logs.append, threading.Event())
    assert logs[-1] == "DIRECT"


def test_system_proxy_credentials_are_redacted_even_in_non_url_errors(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://user:p%2Fass@system.example:80")
    message = ProxyPolicy().redact("http://user:p%2Fass@system.example:80 auth password p/ass p%2Fass")
    assert message == "http://***@system.example:80 auth password *** ***"


def test_regularization_reloads_policy_and_keeps_provider_redirects(api, monkeypatch):
    _, state = api
    captured = []

    def opener(self, *handlers):
        captured.append((self.mode, handlers))
        return SimpleNamespace(open=lambda *_args, **_kwargs: "response")

    monkeypatch.setattr(ProxyPolicy, "opener", opener)
    state.ctx.save_settings({"network": {"proxy_mode": "direct"}})
    assert state.regularization._open("https://danbooru.donmai.us/posts.json", "danbooru") == "response"
    state.ctx.save_settings({"network": {"proxy_mode": "custom", "proxy_url": "http://localhost:7890"}})
    assert state.regularization._open("https://cdn.donmai.us/test.jpg", "danbooru", media=True) == "response"
    assert [row[0] for row in captured] == ["direct", "custom"]
    assert captured[0][1][0].source == "danbooru"
    assert captured[1][1][0].media is True
