import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import DatasetSourceConfig
from ypuddin.data.index import dataset_fingerprint, scan_sources
from ypuddin.server import create_app, routes_work


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda *a: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Membership", "family": "toy"}).json()
    ctx = app.state.ctx
    root = ctx.dataset_dir(project["id"], project["active_version_id"]) / "images"
    root.mkdir(parents=True)
    for name in ("a", "b"):
        Image.new("RGB", (64, 64), "red").save(root / f"{name}.png")
    (root / "a.json").write_text(
        json.dumps({"tags": ["edited tag"], "nl": "edited description", "extra": {"keep": 123}})
    )
    (root / "a.txt").write_text("preserved alternate tags")
    (root / "b.txt").write_text("b caption")
    Image.new("L", (64, 64), 128).save(root / "a.mask.png")
    response = client.post(f"/api/projects/{project['id']}/datasets", json={"path": str(root)})
    assert response.status_code == 201, response.text
    did = response.json()["source"]["id"]
    yield client, ctx, project, did, root
    app.state.dataset_pipeline.close()
    ctx.versions.close()
    client.close()
    ctx.db.close()


def sources(client, project):
    config = client.get(f"/api/projects/{project['id']}/config").json()
    return [DatasetSourceConfig.model_validate(source) for source in config["dataset"]["sources"]]


def test_membership_changes_training_selection_without_touching_any_sidecar(managed):
    client, ctx, p, did, root = managed
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    original = scan_sources(sources(client, p))
    assert len(original) == 2 and original[0].content_hash == original[1].content_hash
    response = client.post(f"/api/datasets/{did}/membership", json={"paths": ["a.png"], "included": False})
    assert response.status_code == 200, response.text
    assert [Path(row.path).name for row in scan_sources(sources(client, p))] == ["b.png"]
    active = client.get(f"/api/datasets/{did}/images?membership=training").json()
    unused = client.get(f"/api/datasets/{did}/images?membership=unused").json()
    assert [r["rel_path"] for r in active["items"]] == ["b.png"]
    assert unused["items"][0]["rel_path"] == "a.png" and unused["items"][0]["caption_format"] == "json"
    assert client.get(f"/api/datasets/{did}?include_cache=false").json()["stats"]["training_images"] == 1
    assert (
        client.post(
            f"/api/datasets/{did}/membership", json={"paths": ["a.png"], "included": True}
        ).status_code
        == 200
    )
    restored = scan_sources(sources(client, p))
    assert dataset_fingerprint(original, sources(client, p)) == dataset_fingerprint(
        restored, sources(client, p)
    )
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_rename_preserves_tags_masks_membership_and_updates_config_and_index(managed):
    client, ctx, p, did, root = managed
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    client.post(f"/api/datasets/{did}/membership", json={"paths": ["a.png"], "included": False})
    old_sources = sources(client, p)
    fingerprint = dataset_fingerprint(scan_sources(old_sources), old_sources)
    response = client.patch(f"/api/datasets/{did}", json={"name": "portraits"})
    assert response.status_code == 200, response.text
    renamed = root.with_name("portraits")
    assert not root.exists() and renamed.is_dir()
    assert {path.name: path.read_bytes() for path in renamed.iterdir()} == before
    updated = sources(client, p)
    assert updated[0].path == str(renamed) and updated[0].excluded_files == ["a.png"]
    assert dataset_fingerprint(scan_sources(updated), updated) == fingerprint
    assert (
        client.get(f"/api/datasets/{did}/images?membership=unused").json()["items"][0]["caption_format"]
        == "json"
    )
    assert client.patch(f"/api/datasets/{did}", json={"repeats": 5, "masked_loss": True}).status_code == 200
    assert sources(client, p)[0].repeats == 5
    info = client.get(f"/api/datasets/{did}?include_cache=false").json()
    assert info["source"]["repeats"] == 5 and info["masked_loss"] is True


def test_rename_conflicts_and_config_write_failure_preserve_original_files(managed, monkeypatch):
    client, ctx, p, did, root = managed
    root.with_name("taken").mkdir()
    assert client.patch(f"/api/datasets/{did}", json={"name": "taken"}).status_code == 409
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(routes_work, "_write_project_config", fail)
    with pytest.raises(OSError, match="disk full"):
        client.patch(f"/api/datasets/{did}", json={"name": "renamed"})
    assert root.is_dir() and not root.with_name("renamed").exists()
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert client.get(f"/api/datasets/{did}?include_cache=false").json()["source"]["path"] == str(root)
    assert client.get(f"/api/datasets/{did}/images").json()["total"] == 2


def test_selection_rejects_unknown_paths_and_does_not_change_a_running_versions_data(managed, monkeypatch):
    client, ctx, p, did, root = managed
    assert (
        client.post(
            f"/api/datasets/{did}/membership", json={"paths": ["../escape.png"], "included": False}
        ).status_code
        == 400
    )
    assert (
        client.post(
            f"/api/datasets/{did}/membership", json={"paths": ["missing.png"], "included": False}
        ).status_code
        == 409
    )
    original = ctx.db.fetchone
    monkeypatch.setattr(
        ctx.db,
        "fetchone",
        lambda sql, args=(): (
            {"id": "running"} if "SELECT id FROM jobs WHERE version_id" in sql else original(sql, args)
        ),
    )
    assert client.patch(f"/api/datasets/{did}", json={"name": "renamed"}).status_code == 409
    assert (
        client.post(
            f"/api/datasets/{did}/membership", json={"paths": ["a.png"], "included": False}
        ).status_code
        == 409
    )
    assert root.exists()


def test_plan_reuses_file_index_and_rechecks_changed_files(tmp_path, monkeypatch):
    from ypuddin.data import index
    from ypuddin.server import routes_core

    monkeypatch.setattr(routes_core, "gpu_info", lambda: [])
    root = tmp_path / "pictures"
    root.mkdir()
    Image.new("RGB", (64, 64), "red").save(root / "a.png")
    app = create_app(tmp_path / "server", frontend_dist=tmp_path / "missing")
    client = TestClient(app)
    calls = []
    original = index.content_hash

    def counted(path, *a, **kw):
        calls.append(str(path))
        return original(path, *a, **kw)

    monkeypatch.setattr(index, "content_hash", counted)
    cfg = {
        "model": {"family": "toy"},
        "dataset": {"sources": [{"path": str(root)}], "resolutions": [64]},
        "sampling": {"enabled": False},
    }
    try:
        first = client.post("/api/plan", json={"config": cfg}).json()
        assert first["ok"], first
        assert calls.count(str(root / "a.png")) == 1
        cfg["optimizer"] = {"lr": 0.0002}
        second = client.post("/api/plan", json={"config": cfg}).json()
        assert second["buckets"] == first["buckets"]
        assert calls.count(str(root / "a.png")) == 1
        Image.new("RGB", (96, 64), "blue").save(root / "a.png")
        third = client.post("/api/plan", json={"config": cfg}).json()
        assert third["ok"], third
        assert calls.count(str(root / "a.png")) == 2
    finally:
        app.state.dataset_pipeline.close()
        app.state.ctx.versions.close()
        client.close()
        app.state.ctx.db.close()


def test_bulk_selection_is_atomic_and_can_hold_out_unreadable_images(managed):
    client, ctx, p, did, root = managed
    bad = root / "broken.png"
    bad.write_bytes(b"not an image")
    endpoint = f"/api/projects/{p['id']}/versions/{p['active_version_id']}/dataset-membership"
    response = client.post(
        endpoint,
        json={
            "included": False,
            "images": [
                {"dataset_id": did, "rel_path": "a.png"},
                {"dataset_id": did, "rel_path": "missing.png"},
            ],
        },
    )
    assert response.status_code == 409
    assert sources(client, p)[0].excluded_files == []
    response = client.post(
        endpoint, json={"included": False, "images": [{"dataset_id": did, "rel_path": "broken.png"}]}
    )
    assert response.status_code == 200, response.text
    assert len(scan_sources(sources(client, p))) == 2
    report = {
        "images": [{"path": str(bad), "roles": ["train"], "issues": [{"severity": "error"}]}],
        "source_issues": [],
    }
    checked = client.app.state.dataset_pipeline._apply_membership(p["id"], p["active_version_id"], report)
    assert checked["training_errors"] == 0 and checked["images"][0]["training_enabled"] is False
    assert bad.read_bytes() == b"not an image"


def test_nested_source_split_preserves_membership_and_avoids_double_counting(managed):
    client, ctx, p, did, root = managed
    nested = root / "child"
    nested.mkdir()
    Image.new("RGB", (64, 64), "blue").save(nested / "c.png")
    child_id = routes_work._register_dataset(ctx, p["id"], routes_work.DatasetBody(path=str(nested)))
    routes_work._index_dataset(ctx, child_id)
    config = client.get(f"/api/projects/{p['id']}/config").json()
    config["dataset"]["sources"] = [{**config["dataset"]["sources"][0], "excluded_files": ["child/c.png"]}]
    routes_work._write_project_config(ctx, p["id"], config, p["active_version_id"])
    response = client.patch(f"/api/datasets/{child_id}", json={"repeats": 3})
    assert response.status_code == 200, response.text
    current = sources(client, p)
    assert current[0].excluded_dirs == ["child"] and current[1].excluded_files == ["c.png"]
    assert len(scan_sources(current)) == 2
    assert (
        client.post(
            f"/api/datasets/{child_id}/membership", json={"paths": ["c.png"], "included": True}
        ).status_code
        == 200
    )
    rows = scan_sources(sources(client, p))
    assert len(rows) == 3 and len({r.path for r in rows}) == 3
