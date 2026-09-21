"""Verified model catalog integration, using existing localhost tiny-weight fixtures only."""

import hashlib
import urllib.parse
from pathlib import Path

import pytest
import torch
from safetensors.torch import save

from Test.tests.e2e.test_model_downloads import download_service as inherited_download_service  # noqa: F401
from Test.tests.e2e.test_model_downloads import wait_for
from ypuddin.server import model_recommendations as catalog
from ypuddin.server.model_recommendations import RecommendedModel, RecommendedSource


def recommendation(payload, *, id_="tiny-anima", family="anima", kind="dit", filename="tiny.safetensors"):
    return RecommendedModel(
        id=id_,
        family=family,
        kind=kind,
        name="Tiny verified fixture",
        dtype="bf16",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="fixture/hf-repo",
                filename="split/" + filename,
                revision="main",
                url="https://huggingface.co/fixture/hf-repo/blob/main/split/" + filename,
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="different/ms-repo",
                filename="weights/" + filename,
                revision="master",
                url="https://modelscope.cn/models/different/ms-repo/files",
            ),
        ],
    )


@pytest.fixture
def tiny(request, monkeypatch):
    # Prevent fallback to a developer's actual HF CLI cache in the inherited fixture.
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_fixture_key")
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "ms_fixture_key")
    client, control, root, app = request.getfixturevalue("inherited_download_service")
    entry = recommendation(control.payload)
    monkeypatch.setattr(catalog, "RECOMMENDATIONS", [entry])
    return client, control, root, app, entry


def download(tiny, provider="huggingface", id_=None):
    return tiny[0].post(
        f"/api/models/recommendations/{id_ or tiny[4].id}/download", json={"provider": provider}
    )


@pytest.mark.parametrize("provider", ["huggingface", "modelscope"])
def test_recommended_download_verifies_bytes_and_explicit_provider_mapping(tiny, provider):
    client, control, root, app, entry = tiny
    result = download(tiny, provider)
    assert result.status_code == 202, result.text
    job = wait_for(client, result.json()["id"])
    assert job["status"] == "completed", job
    assert job["recommendation_id"] == entry.id
    assert job["expected_size"] == entry.size and job["sha256"] == entry.sha256
    target = Path(job["target_path"])
    assert target == root / "anima" / "dit" / entry.sha256[:12] / "tiny.safetensors"
    assert target.read_bytes() == control.payload
    request = control.requests[-1]
    if provider == "huggingface":
        assert (
            request.full_url == "https://huggingface.co/fixture/hf-repo/resolve/main/split/tiny.safetensors"
        )
        assert request.get_header("Authorization") == "Bearer hf_fixture_key"
    else:
        assert urllib.parse.urlsplit(request.full_url).path == "/api/v1/models/different/ms-repo/repo"
        assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {
            "Revision": ["master"],
            "FilePath": ["weights/tiny.safetensors"],
        }
        assert request.get_header("Cookie") == "m_session_id=ms_fixture_key"
    entries = client.get("/api/models/recommendations").json()
    assert entries[0]["available_path"] == str(target) and entries[0]["is_default"]
    assert entries[0]["model_id"] == job["model_id"]
    assert "fixture_key" not in str(app.state.ctx.db.get_kv("model_downloads", []))
    assert client.post(f"/api/models/recommendations/{entry.id}/use", json={}).json()["id"] == job["model_id"]


def test_turbo_download_and_reuse_keep_raw_training_default(tiny, monkeypatch):
    from ypuddin.models.krea2 import variants
    from ypuddin.server.family_config import initial_family_config

    client, control, root, app, _ = tiny
    control.payload = save(
        {
            "first.weight": torch.zeros(2, 64),
            "txtfusion.projector.weight": torch.zeros(1, 12),
            "blocks.0.attn.wq.weight": torch.zeros(2, 2),
        }
    )
    entry = recommendation(control.payload, id_="tiny-turbo", family="krea2").model_copy(
        update={"purpose": "inference", "variant": "turbo"}
    )
    monkeypatch.setattr(catalog, "RECOMMENDATIONS", [entry])
    monkeypatch.setitem(variants.KNOWN_VARIANTS, entry.sha256, "turbo")
    raw = root / "existing-raw.safetensors"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(
        save(
            {
                "first.weight": torch.ones(2, 64),
                "txtfusion.projector.weight": torch.ones(1, 12),
                "blocks.0.attn.wq.weight": torch.ones(2, 2),
            }
        )
    )
    registered = client.post(
        "/api/models",
        json={"family": "krea2", "kind": "dit", "path": str(raw), "variant": "raw", "is_default": True},
    )
    assert registered.status_code == 200, registered.text

    original = control.payload
    control.payload = original[:-1] + bytes([original[-1] ^ 1])
    # Even an old client sending is_default=True cannot promote a catalog Turbo.
    started = client.post(
        f"/api/models/recommendations/{entry.id}/download",
        json={"provider": "huggingface", "is_default": True},
    )
    assert started.status_code == 202, started.text
    failed = wait_for(client, started.json()["id"])
    assert failed["status"] == "failed", failed
    # Pre-feature persisted attempts retain trusted catalog identity through SHA256.
    old = app.state.model_downloads.tasks[failed["id"]]
    old.pop("variant", None)
    old.pop("purpose", None)
    control.payload = original
    retry = client.post(f"/api/models/downloads/{failed['id']}/retry")
    assert retry.status_code == 202, retry.text
    completed = wait_for(client, retry.json()["id"])
    assert completed["status"] == "completed", completed
    assert completed["purpose"] == "inference" and completed["variant"] == "turbo"
    assert not completed["is_default"]
    target = Path(completed["target_path"])
    assert variants.verified_variant(target) == "turbo"
    available = client.get("/api/models/recommendations").json()[0]
    assert available["available_path"] == str(target) and not available["is_default"]
    reused = client.post(f"/api/models/recommendations/{entry.id}/use", json={})
    assert reused.status_code == 200, reused.text
    assert reused.json()["purpose"] == "inference" and not reused.json()["is_default"]
    assert initial_family_config(app.state.ctx, "krea2")["model"]["dit_path"] == str(raw)
    assert len(control.requests) == 2


def test_same_bytes_from_two_providers_share_destination_and_conflict_while_running(tiny):
    client, control, _, _, entry = tiny
    control.mode = "race"
    first = download(tiny)
    assert first.status_code == 202
    assert control.started.wait(3)
    second = download(tiny, "modelscope")
    assert second.status_code == 409 and second.json()["error"]["code"] == "download.active"
    client.post(f"/api/models/downloads/{first.json()['id']}/cancel")
    control.release.set()
    assert wait_for(client, first.json()["id"])["status"] == "cancelled"
    control.mode = "ok"
    second = download(tiny, "modelscope")
    assert second.status_code == 202
    assert second.json()["target_path"] == first.json()["target_path"]
    completed = wait_for(client, second.json()["id"])
    assert completed["status"] == "completed"
    assert download(tiny).status_code == 409
    assert len(client.get("/api/models").json()) == 1
    assert completed["sha256"] == entry.sha256


@pytest.mark.parametrize("corruption", ["same-size", "short", "long"])
def test_corrupt_recommended_weights_never_register_or_publish(tiny, corruption):
    client, control, root, _, entry = tiny
    original = control.payload
    control.payload = (
        (original[:-1] + bytes([original[-1] ^ 1]))
        if corruption == "same-size"
        else original[:-1]
        if corruption == "short"
        else original + b"extra"
    )
    result = download(tiny)
    assert result.status_code == 202
    failed = wait_for(client, result.json()["id"])
    assert failed["status"] == "failed" and (
        "integrity" in failed["error"] or "verified size" in failed["error"]
    )
    assert failed["expected_size"] == entry.size
    assert not Path(failed["target_path"]).exists()
    assert client.get("/api/models").json() == []
    assert not list(root.rglob("*.safetensors"))


@pytest.mark.parametrize("catalog_change", ["removed", "updated"])
def test_retry_retains_original_verification_when_catalog_changes(tiny, monkeypatch, catalog_change):
    client, control, _, _, entry = tiny
    original = control.payload
    control.payload = original[:-1] + bytes([original[-1] ^ 1])
    first = download(tiny)
    failed = wait_for(client, first.json()["id"])
    assert failed["status"] == "failed"
    replacement = entry.model_copy(update={"sha256": "0" * 64, "size": entry.size + 100})
    monkeypatch.setattr(catalog, "RECOMMENDATIONS", [] if catalog_change == "removed" else [replacement])
    control.payload = original
    retry = client.post(f"/api/models/downloads/{failed['id']}/retry")
    assert retry.status_code == 202, retry.text
    row = wait_for(client, retry.json()["id"])
    assert row["status"] == "completed", row
    for key in ("target_path", "source_url", "recommendation_id", "sha256", "expected_size"):
        assert row[key] == failed[key]


def test_incomplete_persisted_verification_cannot_downgrade_to_unverified_retry(tiny):
    client, control, _, app, _ = tiny
    control.payload += b"broken"
    first = download(tiny)
    failed = wait_for(client, first.json()["id"])
    app.state.model_downloads.tasks[failed["id"]]["sha256"] = None
    assert client.post(f"/api/models/downloads/{failed['id']}/retry").status_code == 409
    assert len(app.state.model_downloads.tasks) == 1


def test_shared_vae_reuses_one_verified_path_across_families(tiny, monkeypatch):
    client, control, _, _, _ = tiny
    control.payload = save({"encoder.conv.weight": torch.zeros(4), "decoder.conv.weight": torch.ones(4)})
    anima = recommendation(
        control.payload, id_="anima-vae", kind="vae", filename="qwen_image_vae.safetensors"
    )
    krea = recommendation(
        control.payload, id_="krea2-vae", family="krea2", kind="vae", filename="qwen_image_vae.safetensors"
    )
    monkeypatch.setattr(catalog, "RECOMMENDATIONS", [anima, krea])
    result = download(tiny, id_=anima.id)
    model = wait_for(client, result.json()["id"])
    assert model["status"] == "completed", model
    entries = {item["id"]: item for item in client.get("/api/models/recommendations").json()}
    assert entries[krea.id]["available_path"] == model["target_path"] and entries[krea.id]["model_id"] is None
    response = client.post(f"/api/models/recommendations/{krea.id}/use", json={})
    assert response.status_code == 200, response.text
    assert response.json()["family"] == "krea2" and response.json()["kind"] == "vae"
    assert response.json()["path"] == model["target_path"] and response.json()["is_default"]
    assert len(control.requests) == 1
    assets = client.get("/api/models").json()
    assert len(assets) == 2 and {a["path"] for a in assets} == {model["target_path"]}


def test_local_registered_candidate_is_verified_once_and_later_edits_invalidate_proof(tiny, monkeypatch):
    client, control, root, _, entry = tiny
    path = root / "tiny.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(control.payload)
    registered = client.post("/api/models", json={"family": "anima", "kind": "dit", "path": str(path)}).json()
    assert client.get("/api/models/recommendations").json()[0]["available_path"] == str(path)
    original_open, reads = Path.open, []

    def opened(self, mode="r", *args, **kwargs):
        if self == path and mode == "rb":
            reads.append(self)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    response = client.post(f"/api/models/recommendations/{entry.id}/use", json={})
    assert response.status_code == 200 and response.json()["id"] == registered["id"]
    assert len(reads) == 1
    assert client.get("/api/models/recommendations").json()[0]["is_default"]
    assert client.post(f"/api/models/recommendations/{entry.id}/use", json={}).status_code == 200
    assert len(reads) == 1  # Polling and repeated use do not reread the whole model.
    path.write_bytes(control.payload[:-1] + bytes([control.payload[-1] ^ 1]))
    response = client.post(f"/api/models/recommendations/{entry.id}/use", json={})
    assert response.status_code == 400 and response.json()["error"]["code"] == "model.integrity"
    assert len(reads) == 2 and path.exists()
    assert client.get("/api/models/recommendations").json()[0]["available_path"] is None


def test_replaced_verified_model_cannot_reuse_catalog_identity_to_set_default(tiny):
    from Test.tests.unit.test_flux_model_inspection import sparse_headers, transformer_shapes
    from ypuddin.server.model_inspection import FLUX2_DEV_UNSUPPORTED_REASON

    client, control, root, app, entry = tiny
    path = root / "tiny.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(control.payload)
    asset = client.post("/api/models", json={"family": "anima", "kind": "dit", "path": str(path)}).json()
    assert client.post(f"/api/models/recommendations/{entry.id}/use", json={}).status_code == 200
    assert catalog.cached_sha256(app.state.ctx, path) == entry.sha256
    assert client.patch(f"/api/models/{asset['id']}", json={"is_default": False}).status_code == 200

    sparse_headers(path, transformer_shapes("flux2", 6144))
    assert catalog.cached_sha256(app.state.ctx, path) is None
    response = client.patch(f"/api/models/{asset['id']}", json={"is_default": True})
    assert response.status_code == 422 and FLUX2_DEV_UNSUPPORTED_REASON in response.text
    assert client.get("/api/models").json()[0]["is_default"] is False
    assert path.exists()


def test_unavailable_ids_sources_and_unpermitted_registered_paths_are_rejected(tiny, monkeypatch):
    client, control, root, app, entry = tiny
    assert download(tiny, id_="missing").status_code == 404
    assert client.post("/api/models/recommendations/missing/use", json={}).status_code == 404
    assert client.post(f"/api/models/recommendations/{entry.id}/use", json={}).status_code == 404
    assert download(tiny, "unknown-platform").status_code == 422
    monkeypatch.setattr(catalog, "RECOMMENDATIONS", [entry.model_copy(update={"sources": entry.sources[:1]})])
    assert download(tiny, "modelscope").status_code == 400
    path = root / "tiny.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(control.payload)
    assert (
        client.post("/api/models", json={"family": "anima", "kind": "dit", "path": str(path)}).status_code
        == 200
    )
    app.state.ctx.allowed_roots = [root / "different-root"]
    assert client.get("/api/models/recommendations").json()[0]["available_path"] is None
    assert client.post(f"/api/models/recommendations/{entry.id}/use", json={}).status_code == 404
    assert not control.requests
