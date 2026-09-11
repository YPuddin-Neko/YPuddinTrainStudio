"""Readable project IDs, stable version directories and independently bound run files."""

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app, routes_work


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    yield client, app.state.ctx
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def project(api, id_="Character_01"):
    response = api[0].post("/api/projects", json={"id": id_, "name": "布丁・キャラクター 🐈"})
    assert response.status_code == 201, response.text
    return response.json()


def png():
    out = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(out, format="PNG")
    return out.getvalue()


def version(client, project, **options):
    response = client.post(f"/api/projects/{project['id']}/versions", json={"name": "二版", **options})
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = client.get(f"/api/projects/{project['id']}/versions/{response.json()['id']}").json()
        if row["status"] != "copying":
            assert row["status"] == "ready", row
            return row
        time.sleep(0.01)
    pytest.fail("version copy timed out")


def job(api, p, *, config=None):
    client, _ = api
    if config is None:
        config = client.get(f"/api/projects/{p['id']}/config").json()
    config["model"].update(family="toy", dtype="fp32")
    config["dataset"].update(resolutions=[64], bucket_step=16, num_workers=0)
    config["loop"].update(epochs=1, mixed_precision="no")
    response = client.post(
        "/api/jobs",
        json={"name": "训练", "project_id": p["id"], "version_id": p["active_version_id"], "config": config},
    )
    assert response.status_code == 201, response.text
    return response.json()


def upload(api, p, **fields):
    response = api[0].post(
        f"/api/projects/{p['id']}/datasets/upload",
        files=[("files", ("a.png", png(), "image/png"))],
        data=fields,
    )
    assert response.status_code == 200, response.text
    return response.json()["source"]


@pytest.mark.parametrize(
    "id_",
    [
        "",
        "汉字",
        "a-b",
        "a.b",
        "../escape",
        "C:\\escape",
        " white",
        "a/b",
        "CON",
        "prn",
        "AUX",
        "nul",
        "COM1",
        "lpt9",
        "x" * 65,
    ],
)
def test_invalid_ids_do_not_create_project_or_files(api, id_):
    client, c = api
    response = client.post("/api/projects", json={"id": id_, "name": "任意显示名"})
    assert response.status_code == 422, response.text
    assert client.get("/api/projects").json() == []
    assert not (c.data_root / "project").exists()


def test_display_name_id_and_sequential_version_paths_are_independent(api):
    client, c = api
    p = project(api)
    assert p["id"] == "Character_01" and p["layout_version"] == 2
    assert p["name"] == "布丁・キャラクター 🐈"
    v1 = client.get(f"/api/projects/{p['id']}/versions").json()[0]
    root = c.data_root / "project" / p["id"] / "v1"
    assert v1["number"] == 1 and Path(v1["paths"]["root"]) == root
    for key in ("traindata", "reg", "samples", "output"):
        assert Path(v1["paths"][key]) == root / key and (root / key).is_dir()
    assert v1["paths"]["datasets"] == v1["paths"]["traindata"]
    assert v1["paths"]["runs"] == v1["paths"]["output"]
    assert client.patch(f"/api/projects/{p['id']}", json={"name": "新的多语言名"}).status_code == 200
    v2 = version(client, p, data_mode="empty")
    v3 = version(client, p, name="third", data_mode="empty")
    assert (v2["number"], v3["number"]) == (2, 3)
    assert Path(v2["paths"]["root"]) == root.parent / "v2"
    assert Path(v3["paths"]["root"]) == root.parent / "v3"
    assert v1["id"] != v2["id"] != v3["id"]
    assert (root / "config.json").is_file()


def test_project_ids_collide_case_insensitively_and_existing_files_are_preserved(api):
    client, c = api
    project(api)
    assert client.post("/api/projects", json={"id": "character_01", "name": "duplicate"}).status_code == 409
    orphan = c.data_root / "project" / "Orphan"
    orphan.mkdir()
    (orphan / "keep.txt").write_text("do not overwrite")
    assert client.post("/api/projects", json={"id": "orphan", "name": "collision"}).status_code == 409
    assert (orphan / "keep.txt").read_text() == "do not overwrite"


def test_reg_upload_has_separate_directory_metadata_and_config(api):
    client, c = api
    p = project(api)
    train = upload(api, p)
    response = client.post(
        f"/api/projects/{p['id']}/datasets/upload",
        files=[("files", ("a.png", png(), "image/png")), ("files", ("a.caption", b"person", "text/plain"))],
        data={
            "name": "正则",
            "repeats": "2",
            "is_reg": "true",
            "prior_weight": "0.5",
            "class_prompt": "person",
            "caption_ext": ".caption",
        },
    )
    assert response.status_code == 200, response.text
    reg = response.json()["source"]
    assert Path(train["path"]).is_relative_to(c.dataset_dir(p["id"]))
    assert Path(reg["path"]).is_relative_to(c.reg_dir(p["id"]))
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    source = next(source for source in cfg["dataset"]["sources"] if source["is_reg"])
    assert (
        source["prior_weight"] == 0.5
        and source["caption_ext"] == ".caption"
        and source["class_prompt"] == "person"
    )
    assert client.get(f"/api/datasets/{reg['id']}/images").json()["items"][0]["caption"] == "person"
    v2 = version(client, p)
    cloned = client.get(f"/api/projects/{p['id']}/config", params={"version_id": v2["id"]}).json()["dataset"][
        "sources"
    ]
    for source in cloned:
        assert Path(source["path"]).is_relative_to(
            Path(v2["paths"]["reg" if source["is_reg"] else "traindata"])
        )
    copied = next(source for source in cloned if source["is_reg"])
    (Path(copied["path"]) / "a.caption").write_text("changed copy")
    assert (Path(reg["path"]) / "a.caption").read_text() == "person"


def test_job_samples_and_outputs_bind_independently_and_retry_gets_new_directories(api):
    client, c = api
    p = project(api)
    upload(api, p)
    j = job(api, p)
    root = c.version_dir(p["id"])
    assert Path(j["run_dir"]) == root / "output" / j["id"]
    assert Path(j["samples_dir"]) == root / "samples" / j["id"]
    cfg = client.get(f"/api/jobs/{j['id']}/config").json()
    assert cfg["sampling"]["output_dir"] == j["samples_dir"]
    from_snapshot = job(api, p, config=cfg)
    assert Path(from_snapshot["run_dir"]).parent == Path(j["run_dir"]).parent
    sample = Path(j["samples_dir"]) / "sample.png"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(png())
    response = client.get(f"/api/jobs/{j['id']}/files", params={"kind": "sample", "path": "sample.png"})
    assert response.status_code == 200 and response.content == png()
    retry = client.post(f"/api/jobs/{j['id']}/retry").json()
    assert retry["id"] != j["id"]
    assert Path(retry["samples_dir"]) == root / "samples" / retry["id"]
    assert (
        client.get(f"/api/jobs/{retry['id']}/config").json()["sampling"]["output_dir"] == retry["samples_dir"]
    )
    assert (
        client.get(
            f"/api/jobs/{retry['id']}/files", params={"kind": "sample", "path": "sample.png"}
        ).status_code
        == 404
    )


def test_settings_and_job_specific_output_roots_preserve_version_and_job_isolation(api, tmp_path):
    client, c = api
    p = project(api)
    upload(api, p)
    global_root = tmp_path / "global_outputs"
    c.save_settings({"paths": {"output_mode": "custom", "output_dir": str(global_root)}})
    j = job(api, p)
    assert Path(j["run_dir"]) == global_root / p["id"] / "v1" / j["id"]
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    custom = tmp_path / "custom_per_run"
    cfg["checkpoint"]["output_dir"] = str(custom)
    second = job(api, p, config=cfg)
    assert Path(second["run_dir"]) == custom / p["id"] / "v1" / second["id"]
    assert Path(second["samples_dir"]).parent == c.samples_dir(p["id"])
    c.save_settings({"paths": {"output_mode": "project"}})
    third = job(api, p)
    assert Path(third["run_dir"]).parent == c.default_runs_dir(p["id"])
    assert client.get(f"/api/jobs/{j['id']}").json()["run_dir"] == j["run_dir"]


def test_old_custom_output_settings_are_inferred_without_rewriting_them(api, tmp_path):
    _, c = api
    custom = str(tmp_path / "legacy_custom")
    raw = json.dumps({"paths": {"output_dir": custom}})
    c.settings_path.write_text(raw)
    assert c.settings()["paths"]["output_mode"] == "custom"
    assert c.settings_path.read_text() == raw


def test_old_job_sample_location_remains_downloadable(api):
    client, c = api
    p = project(api)
    upload(api, p)
    j = job(api, p)
    c.db.update("jobs", j["id"], {"samples_dir": None})
    sample = Path(j["run_dir"]) / "samples" / "old.png"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(png())
    assert (
        client.get(f"/api/jobs/{j['id']}/files", params={"kind": "sample", "path": "old.png"}).content
        == png()
    )


def test_import_of_configured_reg_source_keeps_reg_directory_and_advanced_caption(api, tmp_path):
    client, c = api
    p = project(api)
    source = tmp_path / "regularization"
    source.mkdir()
    (source / "a.png").write_bytes(png())
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    cfg["dataset"]["sources"] = [
        {"path": str(source), "is_reg": True, "prior_weight": 0.3, "caption": {"shuffle": False}}
    ]
    assert client.put(f"/api/projects/{p['id']}/config", json=cfg).status_code == 200
    response = client.post(f"/api/projects/{p['id']}/datasets", json={"path": str(source)})
    assert response.status_code == 201, response.text
    registered = response.json()["source"]
    assert registered["is_reg"] and Path(registered["path"]).is_relative_to(c.reg_dir(p["id"]))
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    assert cfg["dataset"]["sources"][0]["caption"] == {"shuffle": False}


def test_failed_project_creation_cleans_only_its_new_directory(api, monkeypatch):
    _, c = api

    def fail(*args, **kwargs):
        raise OSError("fixture write failed")

    monkeypatch.setattr(routes_work, "_write_project_config", fail)
    with pytest.raises(OSError, match="fixture write failed"):
        routes_work.create_project(routes_work.ProjectBody(id="Broken", name="display"), c)
    assert not c.db.fetchone("SELECT id FROM projects WHERE id='Broken'")
    assert not c.db.fetchone("SELECT id FROM project_versions WHERE project_id='Broken'")
    assert not (c.data_root / "project" / "Broken").exists()


def test_project_delete_removes_new_layout_only_when_explicitly_requested(api):
    client, c = api
    first = project(api, "KeepFiles")
    root = c.project_dir(first["id"])
    assert client.delete(f"/api/projects/{first['id']}").status_code == 200
    assert root.exists()
    second = project(api, "RemoveFiles")
    root = c.project_dir(second["id"])
    assert client.delete(f"/api/projects/{second['id']}", params={"delete_files": True}).status_code == 200
    assert not root.exists()


def test_versioned_legacy_project_keeps_its_root_and_files_when_numbered(api):
    client, c = api
    c.db.insert(
        "projects",
        {"id": "p_old", "name": "旧项目", "created_at": 1, "updated_at": 1, "active_version_id": "v_old"},
    )
    c.db.insert(
        "project_versions",
        {"id": "v_old", "project_id": "p_old", "name": "old version", "created_at": 1, "updated_at": 1},
    )
    old_root = c.data_root / "projects" / "p_old" / "versions" / "v_old"
    (old_root / "datasets").mkdir(parents=True)
    (old_root / "config.json").write_text("{}")
    original = old_root / "datasets" / "keep.txt"
    original.write_text("old data remains here")
    c.db._migrate_versions()
    row = client.get("/api/projects/p_old/versions/v_old").json()
    assert row["number"] == 1 and Path(row["paths"]["root"]) == old_root
    assert client.get("/api/projects/p_old").json()["layout_version"] == 1
    assert original.read_text() == "old data remains here"
    assert not (c.data_root / "project" / "p_old").exists()
    c.db._migrate_versions()
    assert c.resolve_version("p_old", "v_old")["number"] == 1
