"""Version migration, actual data snapshots, operation locking and immutable run ownership."""

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app, routes_work, versions
from ypuddin.server.db import SCHEMA, Database


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Version tests"}).json()
    yield client, app.state.ctx, project
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def data(root: Path, color="red") -> Path:
    root.mkdir(parents=True)
    Image.new("RGB", (64, 64), color).save(root / "a.png")
    (root / "a.txt").write_text("original caption")
    Image.new("L", (64, 64), 127).save(root / "a.mask.png")
    return root


def wait_version(client, pid, vid):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = client.get(f"/api/projects/{pid}/versions/{vid}").json()
        if row["status"] != "copying":
            return row
        time.sleep(0.02)
    pytest.fail("copy did not finish")


def fork(api, **kwargs):
    client, _, p = api
    response = client.post(f"/api/projects/{p['id']}/versions", json={"name": "v2", **kwargs})
    assert response.status_code == 202, response.text
    return wait_version(client, p["id"], response.json()["id"])


def test_legacy_migration_preserves_paths_and_snapshots(tmp_path):
    database = tmp_path / "studio.db"
    conn = sqlite3.connect(database)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO projects (id,name,created_at,updated_at) VALUES ('p_old','Old',1,2)")
    conn.execute(
        "INSERT INTO datasets (id,project_id,path,created_at) VALUES ('d_old','p_old','/external/dataset',1)"
    )
    conn.execute(
        "INSERT INTO jobs (id,type,name,project_id,status,created_at,run_dir,config_json) VALUES ('j_old','train','Old','p_old','completed',1,'/old/run','{\"recipe\":42}')"
    )
    conn.execute(
        "INSERT INTO artifacts (id,project_id,job_id,name,path,created_at) VALUES ('a_old','p_old','j_old','w','/old/run/w',1)"
    )
    conn.commit()
    conn.close()
    db = Database(database)
    project = db.fetchone("SELECT * FROM projects")
    vid = project["active_version_id"]
    assert db.fetchone("SELECT * FROM project_versions")["legacy_layout"] == 1
    assert db.fetchone("SELECT * FROM datasets")["path"] == "/external/dataset"
    job = db.fetchone("SELECT * FROM jobs")
    assert job["run_dir"] == "/old/run" and job["config_json"] == '{"recipe":42}'
    for table in ("datasets", "jobs", "artifacts"):
        assert db.fetchone(f"SELECT version_id FROM {table}")["version_id"] == vid
    db.close()
    db = Database(database)
    assert len(db.fetchall("SELECT * FROM project_versions")) == 1
    db.close()


def test_fork_copies_config_only_sources_validation_and_sidecars(api, tmp_path):
    client, c, p = api
    external = data(tmp_path / "external")
    validation = data(tmp_path / "validation", "blue")
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    cfg["model"].update(family="toy", dtype="fp32")
    cfg["dataset"].update(
        sources=[{"path": str(external), "resolutions": [64], "repeats": 4, "caption": {"shuffle": True}}],
        resolutions=[64],
        bucket_step=16,
    )
    cfg["validation"].update(sources=[{"path": str(validation), "repeats": 2}], enabled=True)
    cfg["checkpoint"]["resume"] = "/old/state"
    cfg["adapter"]["resume_weights"] = "/old/adapter"
    assert client.put(f"/api/projects/{p['id']}/config", json=cfg).status_code == 200
    v = fork(api)
    assert v["status"] == "ready", v
    assert v["progress"]["files_done"] == v["progress"]["files_total"] == 6
    assert v["stats"]["datasets"] == 2 and v["stats"]["images"] == 2
    cloned = client.get(f"/api/projects/{p['id']}/config?version_id={v['id']}").json()
    original_cfg = client.get(f"/api/projects/{p['id']}/config").json()
    assert original_cfg == cfg
    assert cloned["dataset"]["sources"][0]["caption"] == {"shuffle": True}
    assert cloned["dataset"]["sources"][0]["repeats"] == 4
    assert cloned["checkpoint"]["resume"] is None and cloned["adapter"]["resume_weights"] is None
    for source, original in (
        (cloned["dataset"]["sources"][0], external),
        (cloned["validation"]["sources"][0], validation),
    ):
        directory = Path(source["path"])
        assert directory.is_relative_to(Path(v["paths"]["datasets"]))
        for name in ("a.png", "a.txt", "a.mask.png"):
            assert (directory / name).read_bytes() == (original / name).read_bytes()
            assert (directory / name).stat().st_ino != (original / name).stat().st_ino
        (directory / "a.txt").write_text("changed clone")
        assert (original / "a.txt").read_text() == "original caption"
    assert client.get(f"/api/projects/{p['id']}").json()["active_version_id"] == p["active_version_id"]


def test_import_is_independent_and_version_switch_does_not_redirect_edits(api, tmp_path):
    client, c, p = api
    original = data(tmp_path / "external")
    result = client.post(f"/api/projects/{p['id']}/datasets", json={"path": str(original)}).json()
    source = result["source"]
    managed = Path(source["path"])
    assert managed.is_relative_to(c.version_dir(p["id"], p["active_version_id"]))
    assert managed != original
    v = fork(api)
    assert v["status"] == "ready", v
    assert client.patch(f"/api/projects/{p['id']}", json={"active_version_id": v["id"]}).status_code == 200
    assert client.get(f"/api/projects/{p['id']}/datasets").json()[0]["source"]["id"] != source["id"]
    oldimages = client.get(f"/api/datasets/{source['id']}/images").json()["items"]
    assert (
        client.put(
            f"/api/datasets/{source['id']}/images/{oldimages[0]['hash']}/caption",
            json={"caption": "old version edit"},
        ).status_code
        == 200
    )
    assert (managed / "a.txt").read_text().strip() == "old version edit"
    assert (original / "a.txt").read_text() == "original caption"
    clonepath = Path(client.get(f"/api/projects/{p['id']}/datasets").json()[0]["source"]["path"])
    assert (clonepath / "a.txt").read_text() == "original caption"
    assert client.delete(f"/api/datasets/{source['id']}").status_code == 200
    assert len(client.get(f"/api/projects/{p['id']}/config").json()["dataset"]["sources"]) == 1
    assert (
        client.get(f"/api/projects/{p['id']}/config?version_id={p['active_version_id']}").json()["dataset"][
            "sources"
        ]
        == []
    )


def test_copy_failure_releases_busy_without_touching_source(api, tmp_path, monkeypatch):
    client, c, p = api
    original = data(tmp_path / "source")
    ds = client.post(f"/api/projects/{p['id']}/datasets", json={"path": str(original)}).json()["source"]
    entered, release = threading.Event(), threading.Event()

    def fail(*args):
        entered.set()
        release.wait(5)
        raise OSError("simulated full disk")

    monkeypatch.setattr(versions, "copy_source", fail)
    response = client.post(f"/api/projects/{p['id']}/versions", json={"name": "broken"})
    vid = response.json()["id"]
    assert entered.wait(3)
    assert client.put(f"/api/projects/{p['id']}/config", json={}).status_code == 409
    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 409
    assert client.delete(f"/api/projects/{p['id']}").status_code == 409
    assert client.post(f"/api/projects/{p['id']}/versions", json={"name": "concurrent"}).status_code == 409
    release.set()
    row = wait_version(client, p["id"], vid)
    assert row["status"] == "failed" and "full disk" in row["error"]
    assert row["progress"]["phase"] == "failed"
    assert not Path(row["paths"]["root"]).exists()
    assert Path(ds["path"]).is_dir() and (original / "a.txt").read_text() == "original caption"
    assert c.resolve_version(p["id"])["busy"] is None
    assert client.patch(f"/api/projects/{p['id']}", json={"active_version_id": vid}).status_code == 409


def test_empty_defaults_archive_and_cross_project_boundaries(api):
    client, c, p = api
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    cfg["optimizer"]["lr"] = 0.0123
    client.put(f"/api/projects/{p['id']}/config", json=cfg)
    empty = fork(api, data_mode="empty")
    inherited = client.get(f"/api/projects/{p['id']}/config?version_id={empty['id']}").json()
    assert inherited["optimizer"]["lr"] == 0.0123 and inherited["dataset"]["sources"] == []
    blank = fork(api, name="blank", data_mode="empty", copy_config=False)
    assert (
        client.get(f"/api/projects/{p['id']}/config?version_id={blank['id']}").json()["optimizer"]["lr"]
        != 0.0123
    )
    assert (
        client.patch(
            f"/api/projects/{p['id']}/versions/{p['active_version_id']}", json={"archived": True}
        ).status_code
        == 200
    )
    assert (
        client.patch(
            f"/api/projects/{p['id']}/versions/{empty['id']}",
            json={"archived": True, "name": "Archived recipe"},
        ).status_code
        == 200
    )
    assert (
        client.patch(f"/api/projects/{p['id']}", json={"active_version_id": empty["id"]}).status_code == 409
    )
    other = client.post("/api/projects", json={"name": "Other"}).json()
    assert client.get(f"/api/projects/{other['id']}/config?version_id={blank['id']}").status_code == 404
    assert (
        client.patch(f"/api/projects/{other['id']}", json={"active_version_id": blank["id"]}).status_code
        == 404
    )


def test_jobs_artifacts_and_retry_keep_original_version(api, tmp_path):
    client, c, p = api
    directory = data(tmp_path / "external")
    cfg = {
        "model": {"family": "toy", "dtype": "fp32"},
        "dataset": {
            "sources": [{"path": str(directory)}],
            "resolutions": [64],
            "bucket_step": 16,
            "num_workers": 0,
            "cache_dir": "/do-not-use",
        },
        "loop": {"mixed_precision": "no", "epochs": 1},
    }
    client.put(f"/api/projects/{p['id']}/config", json=cfg)
    v = fork(api)
    job = client.post(
        "/api/jobs",
        json={"name": "old version run", "project_id": p["id"], "version_id": p["active_version_id"]},
    )
    assert job.status_code == 201, job.text
    j = job.json()
    assert j["version_id"] == p["active_version_id"]
    stored = client.get(f"/api/jobs/{j['id']}/config").json()
    assert stored["dataset"]["cache_dir"] == str(c.cache_dir(p["id"], p["active_version_id"]))
    assert Path(j["run_dir"]).parent == c.runs_dir(p["id"], p["active_version_id"])
    client.patch(f"/api/projects/{p['id']}", json={"active_version_id": v["id"]})
    weights = Path(j["run_dir"]) / "weights.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"dummy artifact")
    c.supervisor._register_artifact(j["id"], {"path": str(weights), "step": 1})
    assert len(client.get(f"/api/artifacts?version_id={p['active_version_id']}").json()) == 1
    assert client.get(f"/api/artifacts?version_id={v['id']}").json() == []
    assert client.get(f"/api/jobs?version_id={v['id']}").json()["total"] == 0
    retry = client.post(f"/api/jobs/{j['id']}/retry").json()
    assert (
        retry["version_id"] == j["version_id"] and Path(retry["run_dir"]).parent == Path(j["run_dir"]).parent
    )
    assert client.delete(f"/api/projects/{p['id']}").status_code == 409


def test_validation_only_import_keeps_role_and_caption_settings(api, tmp_path):
    client, c, p = api
    source = data(tmp_path / "held-out")
    (source / "a.txt").rename(source / "a.caption")
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    cfg["validation"]["sources"] = [
        {"path": str(source), "caption_ext": ".caption", "repeats": 7, "resolutions": [64]}
    ]
    assert client.put(f"/api/projects/{p['id']}/config", json=cfg).status_code == 200
    response = client.post(f"/api/projects/{p['id']}/datasets", json={"path": str(source)})
    assert response.status_code == 201, response.text
    ds = response.json()["source"]
    saved = client.get(f"/api/projects/{p['id']}/config").json()
    assert saved["dataset"]["sources"] == []
    assert saved["validation"]["sources"] == [
        {"path": ds["path"], "caption_ext": ".caption", "repeats": 7, "resolutions": [64], "is_reg": False}
    ]
    assert ds["caption_ext"] == ".caption" and ds["repeats"] == 7
    assert client.get(f"/api/datasets/{ds['id']}").json()["stats"]["captioned"] == 1


def test_continuous_forks_do_not_accumulate_managed_directory_prefixes(api, tmp_path):
    client, c, p = api
    original = data(tmp_path / "d_deadbeef-d_1234-角色训练集")
    source = client.post(f"/api/projects/{p['id']}/datasets", json={"path": str(original)}).json()["source"]
    assert Path(source["path"]).name == source["id"] + "-角色训练集"
    previous = p["active_version_id"]
    for i in range(3):
        v = fork(api, name=f"fork-{i}", source_version_id=previous)
        assert v["status"] == "ready", v
        ds = client.get(f"/api/projects/{p['id']}/datasets?version_id={v['id']}").json()[0]["source"]
        assert Path(ds["path"]).name == ds["id"] + "-角色训练集"
        duplicate = client.post(
            f"/api/projects/{p['id']}/datasets?version_id={v['id']}", json={"path": str(original)}
        )
        assert duplicate.status_code == 409, duplicate.text
        assert c.db.fetchone("SELECT origin_path FROM datasets WHERE id=?", (ds["id"],))[
            "origin_path"
        ] == str(original)
        previous = v["id"]


@pytest.mark.parametrize(
    "prefix",
    [
        "../escape",
        "..\\escape",
        "/absolute",
        "C:\\file",
        "CON",
        "NUL.txt",
        "LPT1",
        "bad*glob",
        "trailing.",
    ],
)
def test_checkpoint_prefix_rejects_output_escape_at_service(api, tmp_path, prefix):
    client, c, p = api
    source = data(tmp_path / "training")
    config = {
        "model": {"family": "toy"},
        "dataset": {"sources": [{"path": str(source)}]},
        "checkpoint": {"name": prefix},
    }
    result = client.post("/api/jobs", json={"name": "unsafe prefix", "project_id": p["id"], "config": config})
    assert result.status_code == 400, result.text
    assert "checkpoint.name" in result.text
    assert c.db.fetchall("SELECT * FROM jobs") == []


def test_archiving_active_version_preserves_one_ready_target_and_results(api, tmp_path):
    client, c, project = api
    pid, v1 = project["id"], project["active_version_id"]
    route = f"/api/projects/{pid}/versions/{v1}"
    assert client.patch(route, json={"archived": True}).status_code == 409
    v2 = fork(api, data_mode="empty")
    weight = tmp_path / "weight.safetensors"
    weight.write_bytes(b"original weight bytes")
    for aid, jid in [("a_first", "j_first"), ("a_second", "j_second")]:
        c.db.insert(
            "artifacts",
            {
                "id": aid,
                "project_id": pid,
                "version_id": v1,
                "job_id": jid,
                "name": weight.name,
                "path": str(weight),
                "created_at": 1,
            },
        )
    response = client.patch(route, json={"archived": True})
    assert response.status_code == 200 and response.json()["archived"]
    assert client.get(f"/api/projects/{pid}").json()["active_version_id"] == v2["id"]
    rows = client.get(
        "/api/artifacts", params={"project_id": pid, "version_id": v1, "job_id": "j_first"}
    ).json()
    assert [row["id"] for row in rows] == ["a_first"]
    assert client.get("/api/artifacts/a_first/download").content == weight.read_bytes()
    assert client.delete("/api/artifacts/a_first?delete_file=true").status_code == 409
    assert client.post("/api/artifacts/a_first/convert", json={"format": "kohya"}).status_code == 409
    assert weight.read_bytes() == b"original weight bytes"
    assert (
        client.patch(f"/api/projects/{pid}/versions/{v2['id']}", json={"archived": True}).status_code == 409
    )
    assert client.patch(route, json={"archived": False}).status_code == 200
    assert client.delete("/api/artifacts/a_first").status_code == 200
    assert weight.exists()


def test_rescan_while_already_indexing_is_idempotent_but_archived_is_locked(api, tmp_path, monkeypatch):
    client, c, project = api
    source = data(tmp_path / "images")
    ds = client.post(f"/api/projects/{project['id']}/datasets", json={"path": str(source)}).json()["source"]
    called = []
    monkeypatch.setattr(routes_work, "_index_dataset", lambda *args: called.append(True))
    c.db.update("datasets", ds["id"], {"index_status": "indexing"})
    assert client.post(f"/api/datasets/{ds['id']}/rescan").json() == {"ok": True}
    assert called == []
    c.db.update("project_versions", project["active_version_id"], {"archived": 1})
    assert client.post(f"/api/datasets/{ds['id']}/rescan").status_code == 409


def test_legacy_version_directory_view_matches_existing_data_root(api):
    client, c, project = api
    pid, vid = project["id"], project["active_version_id"]
    c.db.update("project_versions", vid, {"legacy_layout": 1})
    c.db.update("projects", pid, {"layout_version": 1})
    row = client.get(f"/api/projects/{pid}/versions/{vid}").json()
    assert row["paths"]["root"] == str(c.project_dir(pid))
    assert row["paths"]["datasets"] == str(c.project_dir(pid) / "datasets")
    assert row["paths"]["config"] == str(c.project_dir(pid) / "config.json")
    assert row["paths"]["runs"] == str(c.version_dir(pid, vid) / "runs")
