import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app, routes_core, routes_work
from ypuddin.server.source_roles import managed_source_role, normalize_source_roles


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    monkeypatch.setattr(routes_core, "gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "roles", "family": "toy"}).json()
    yield client, app.state.ctx, project
    app.state.regularization.close()
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def image(path, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


def upload(api, is_reg=False):
    client, _, project = api
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "blue" if is_reg else "red").save(buffer, "PNG")
    response = client.post(
        f"/api/projects/{project['id']}/datasets/upload",
        files={"files": ("p.png", buffer.getvalue(), "image/png")},
        data={"is_reg": str(is_reg).lower()},
    )
    assert response.status_code == 200, response.text
    return response.json()["source"]


def test_managed_roots_fix_config_registry_and_job_snapshot_without_moving_data(api):
    client, c, p = api
    train, reg = upload(api), upload(api, True)
    c.db.update("datasets", reg["id"], {"is_reg": 0})
    assert client.get(f"/api/datasets/{reg['id']}").json()["source"]["is_reg"] is True
    url = f"/api/projects/{p['id']}/config"
    cfg = client.get(url).json()
    for source in cfg["dataset"]["sources"]:
        source["is_reg"] = not source["is_reg"]
    originals = {
        path: Path(path).read_bytes()
        for path in [str(Path(train["path"]) / "p.png"), str(Path(reg["path"]) / "p.png")]
    }
    response = client.put(url, json=cfg)
    assert response.status_code == 200, response.text
    fixed = {s["path"]: s["is_reg"] for s in response.json()["dataset"]["sources"]}
    assert fixed == {train["path"]: False, reg["path"]: True}
    assert bool(c.db.fetchone("SELECT is_reg FROM datasets WHERE id=?", (reg["id"],))["is_reg"])
    # Explicit job config bypassing config-save still uses the same server interpretation.
    job = client.post(
        "/api/jobs",
        json={
            "name": "role test",
            "project_id": p["id"],
            "version_id": p["active_version_id"],
            "config": cfg,
        },
    )
    assert job.status_code == 201, job.text
    snapshot = json.loads(
        c.db.fetchone("SELECT config_json FROM jobs WHERE id=?", (job.json()["id"],))["config_json"]
    )
    assert {s["path"]: s["is_reg"] for s in snapshot["dataset"]["sources"]} == fixed
    assert all(Path(path).read_bytes() == content for path, content in originals.items())


def test_plan_scopes_roles_before_automatic_validation_split(api):
    client, c, p = api
    reg = upload(api, True)
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    cfg["dataset"]["sources"][0]["is_reg"] = False
    cfg["validation"].update(enabled=True, split_ratio=0.999999999)
    scoped = client.post(
        "/api/plan", json={"config": cfg, "project_id": p["id"], "version_id": p["active_version_id"]}
    )
    assert scoped.status_code == 200 and scoped.json()["ok"], scoped.text
    unscoped = client.post("/api/plan", json={"config": cfg})
    assert not unscoped.json()["ok"]  # offline external config retains explicitly supplied purpose
    assert cfg["dataset"]["sources"][0]["is_reg"] is False
    roles = client.post(f"/api/projects/{p['id']}/source-roles", json={"config": cfg}).json()
    assert roles == [
        {
            "path": reg["path"],
            "section": "dataset",
            "is_reg": True,
            "managed": True,
            "root": str(c.reg_dir(p["id"])),
            "origin": "version",
            "images": 1,
        }
    ]


def test_unscoped_version_is_rejected_by_both_preflight_endpoints(api):
    client, _, p = api
    for endpoint in ("/api/plan", "/api/config/validate"):
        assert (
            client.post(endpoint, json={"config": {}, "version_id": p["active_version_id"]}).status_code
            == 422
        )


def test_real_ancestry_recognizes_aliases_but_not_names_or_escaped_links(api, tmp_path):
    _, c, p = api
    pid, vid = p["id"], p["active_version_id"]
    reg = c.reg_dir(pid, vid) / "nested" / "photos"
    image(reg / "p.png")
    alias = tmp_path / "alias"
    outside = tmp_path / "reg" / "photos"
    image(outside / "p.png")
    try:
        alias.symlink_to(reg, target_is_directory=True)
        (c.reg_dir(pid, vid) / "outside").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable on this platform")
    assert managed_source_role(c, pid, vid, str(alias)) == (True, str(c.reg_dir(pid, vid)))
    assert managed_source_role(c, pid, vid, str(c.reg_dir(pid, vid) / "outside")) is None
    assert (
        managed_source_role(c, pid, vid, str(c.reg_dir(pid, vid).with_name("reg-other") / "photos")) is None
    )
    cfg = {
        "dataset": {
            "sources": [
                {"path": str(outside), "is_reg": False},
                {"path": r"C:\legacy\reg\photos", "is_reg": False},
            ]
        }
    }
    assert normalize_source_roles(c, pid, cfg, vid) == cfg


def test_legacy_dataset_directory_preserves_explicit_metadata(api):
    _, c, p = api
    pid, vid = p["id"], p["active_version_id"]
    c.db.update("projects", pid, {"layout_version": 1})
    old = c.dataset_dir(pid, vid) / "old"
    image(old / "p.png")
    assert managed_source_role(c, pid, vid, str(old)) is None
    cfg = {"dataset": {"sources": [{"path": str(old), "is_reg": True}]}}
    assert normalize_source_roles(c, pid, cfg, vid) == cfg


def test_external_registered_metadata_is_fallback_and_advanced_edit_stays_explicit(api, tmp_path):
    client, c, p = api
    path = tmp_path / "old_external"
    image(path / "p.png")
    did = routes_work._register_dataset(c, p["id"], routes_work.DatasetBody(path=str(path), is_reg=True))
    normalized = normalize_source_roles(c, p["id"], {"dataset": {"sources": [{"path": str(path)}]}})
    assert normalized["dataset"]["sources"][0]["is_reg"] is True
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    cfg["dataset"]["sources"][0]["is_reg"] = False
    assert (
        client.put(f"/api/projects/{p['id']}/config", json=cfg).json()["dataset"]["sources"][0]["is_reg"]
        is False
    )
    assert c.db.fetchone("SELECT is_reg FROM datasets WHERE id=?", (did,))["is_reg"] == 0
    assert (path / "p.png").is_file()


@pytest.mark.parametrize("is_reg", [False, True])
def test_external_import_role_determines_destination_not_folder_name(api, tmp_path, is_reg):
    client, c, p = api
    source = tmp_path / ("training_named_folder" if is_reg else "reg_named_folder")
    image(source / "p.png")
    original = (source / "p.png").read_bytes()
    response = client.post(f"/api/projects/{p['id']}/datasets", json={"path": str(source), "is_reg": is_reg})
    assert response.status_code == 201, response.text
    registered = response.json()["source"]
    assert registered["is_reg"] == is_reg
    assert Path(registered["path"]).is_relative_to(c.dataset_dir(p["id"], is_reg=is_reg))
    assert (source / "p.png").read_bytes() == original


def test_managed_source_from_other_version_does_not_reclassify_legacy_reference(api):
    _, c, p = api
    other = c.project_dir(p["id"]) / "v999" / "reg" / "existing"
    image(other / "p.png")
    assert (
        normalize_source_roles(c, p["id"], {"dataset": {"sources": [{"path": str(other), "is_reg": False}]}})[
            "dataset"
        ]["sources"][0]["is_reg"]
        is False
    )
