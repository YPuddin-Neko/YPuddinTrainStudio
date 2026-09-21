"""The local-directory and version-copy flows keep imported concepts identifiable."""

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import TrainConfig
from ypuddin.data.index import scan_sources
from ypuddin.server import create_app, dataset_uploads, routes_work


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Directory sync", "family": "toy"}).json()
    yield client, app.state.ctx, project
    app.state.regularization.close()
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def png(color="red", mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, (64, 64), color).save(buffer, "PNG")
    return buffer.getvalue()


def image(path, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png(color))


def upload(api, entries, **fields):
    client, _, project = api
    return client.post(
        f"/api/projects/{project['id']}/datasets/upload",
        files=[("files", (name, body, "application/octet-stream")) for name, body in entries],
        data=fields,
    )


def local(api, path, **fields):
    client, _, project = api
    return client.post(f"/api/projects/{project['id']}/datasets", json={"path": str(path), **fields})


def fork(api):
    client, _, project = api
    response = client.post(f"/api/projects/{project['id']}/versions", json={"name": "copy"})
    assert response.status_code == 202, response.text
    vid = response.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        version = client.get(f"/api/projects/{project['id']}/versions/{vid}").json()
        if version["status"] != "copying":
            return version
        time.sleep(0.01)
    pytest.fail("version copy did not finish")


@pytest.mark.parametrize("is_reg", [False, True])
def test_local_directory_keeps_its_name_structure_and_original_limits(api, tmp_path, monkeypatch, is_reg):
    client, context, project = api
    source = tmp_path / "角色合集"
    image(source / "one/a.png")
    image(source / "two/a.png", "blue")
    (source / "one/a.txt").write_text("caption")
    (source / "two/a.json").write_text(json.dumps({"tags": ["blue"]}))
    (source / "two/a.mask.png").write_bytes(png(128, "L"))
    # Local copies are bounded by their existing path/snapshot rules, not multipart limits.
    monkeypatch.setattr(dataset_uploads, "MAX_UPLOAD_BYTES", 1)
    monkeypatch.setattr(dataset_uploads, "MAX_FILES", 1)
    result = local(api, source, repeats=6, is_reg=is_reg)
    assert result.status_code == 201, result.text
    target = Path(result.json()["source"]["path"])
    assert target == context.dataset_dir(project["id"], is_reg=is_reg) / source.name
    assert (target / "two/a.json").read_bytes() == (source / "two/a.json").read_bytes()
    assert (target / "one/a.png").stat().st_ino != (source / "one/a.png").stat().st_ino
    info = client.get(f"/api/datasets/{result.json()['source']['id']}").json()
    assert info["stats"]["images"] == info["stats"]["captioned"] == 2
    assert info["stats"]["masks"] == 1 and info["source"]["repeats"] == 6


def test_local_sync_reuses_browser_import_source_and_keeps_its_options(api, tmp_path):
    client, context, project = api
    first = upload(api, [("concept/a.png", png())], repeats="7")
    assert first.status_code == 200, first.text
    source = tmp_path / "concept"
    image(source / "a.png")
    image(source / "nested/b.png", "blue")
    result = local(api, source, repeats=2)
    assert result.status_code == 201, result.text
    assert result.json()["source"]["id"] == first.json()["source"]["id"]
    assert result.json()["source"]["repeats"] == 7
    assert (context.dataset_dir(project["id"]) / "concept/nested/b.png").is_file()
    assert len(client.get(f"/api/projects/{project['id']}/datasets").json()) == 1
    assert (source / "a.png").read_bytes() == png()


def test_local_conflicting_sync_keeps_managed_files_and_registry(api, tmp_path):
    client, context, project = api
    source = tmp_path / "concept"
    image(source / "a.png")
    assert local(api, source).status_code == 201
    original = client.get(f"/api/projects/{project['id']}/config").json()
    image(source / "new/b.png", "green")
    image(source / "a.png", "blue")
    result = local(api, source)
    assert result.status_code == 409 and result.json()["error"]["code"] == "upload.conflict"
    target = context.dataset_dir(project["id"]) / "concept"
    assert (target / "a.png").read_bytes() == png()
    assert not (target / "new").exists()
    assert client.get(f"/api/projects/{project['id']}/config").json() == original


def test_local_sync_database_failure_removes_only_new_files(api, tmp_path, monkeypatch):
    client, context, project = api
    source = tmp_path / "concept"
    image(source / "a.png")
    assert local(api, source).status_code == 201
    image(source / "new/b.png", "blue")
    update = context.db.update

    def fail_index(table, id_, fields):
        if table == "datasets" and fields.get("index_status") == "indexing":
            raise OSError("simulated update failure")
        return update(table, id_, fields)

    monkeypatch.setattr(context.db, "update", fail_index)
    result = local(api, source)
    assert result.status_code == 400, result.text
    target = context.dataset_dir(project["id"]) / "concept"
    assert (target / "a.png").read_bytes() == png()
    assert not (target / "new").exists()
    assert client.get(f"/api/projects/{project['id']}/datasets").json()[0]["index_status"] == "ready"


def test_local_import_does_not_register_configured_parent_over_an_indexed_child(api, tmp_path):
    client, context, project = api
    root = context.dataset_dir(project["id"])
    child = root / "old"
    image(child / "a.png")
    did = routes_work._register_dataset(context, project["id"], routes_work.DatasetBody(path=str(child)))
    routes_work._index_dataset(context, did)
    cfg = client.get(f"/api/projects/{project['id']}/config").json()
    cfg["dataset"]["sources"] = [{"path": str(root)}]
    assert client.put(f"/api/projects/{project['id']}/config", json=cfg).status_code == 200
    source = tmp_path / "new"
    image(source / "b.png", "blue")
    result = local(api, source)
    assert result.status_code == 409, result.text
    assert result.json()["error"]["code"] == "dataset.overlap"
    assert not (root / "new").exists()
    assert (child / "a.png").read_bytes() == png()
    assert len(client.get(f"/api/projects/{project['id']}/datasets").json()) == 1


def test_fork_preserves_uploaded_concept_paths_and_version_isolation(api):
    client, context, project = api
    uploaded = upload(
        api,
        [
            ("one/a.png", png()),
            ("one/nested/a.txt", b"caption"),
            ("one/nested/a.png", png("green")),
            ("two/a.png", png("blue")),
        ],
        repeats="5",
    )
    assert uploaded.status_code == 200, uploaded.text
    assert upload(api, [("one/a.png", png("yellow"))], is_reg="true").status_code == 200
    old_root = context.version_dir(project["id"])
    version = fork(api)
    assert version["status"] == "ready", version
    new_root = Path(version["paths"]["root"])
    datasets = client.get(
        f"/api/projects/{project['id']}/datasets", params={"version_id": version["id"]}
    ).json()
    assert {Path(item["source"]["path"]).relative_to(new_root).as_posix() for item in datasets} == {
        "traindata/one",
        "traindata/two",
        "reg/one",
    }
    for name in ("traindata/one/a.png", "traindata/one/nested/a.txt", "traindata/two/a.png", "reg/one/a.png"):
        assert (new_root / name).read_bytes() == (old_root / name).read_bytes()
        assert (new_root / name).stat().st_ino != (old_root / name).stat().st_ino
    old_config = client.get(f"/api/projects/{project['id']}/config").json()
    new_config = client.get(
        f"/api/projects/{project['id']}/config", params={"version_id": version["id"]}
    ).json()
    assert old_config["dataset"]["cache_dir"] != new_config["dataset"]["cache_dir"]
    assert len(scan_sources(TrainConfig.model_validate(new_config).dataset.sources)) == 4
    (new_root / "traindata/one/nested/a.txt").write_text("edited clone")
    assert (old_root / "traindata/one/nested/a.txt").read_text() == "caption"
    result = client.post(
        f"/api/projects/{project['id']}/datasets/upload",
        params={"version_id": version["id"]},
        files=[("files", ("one/b.png", png("yellow"), "image/png"))],
    )
    assert result.status_code == 200, result.text
    assert result.json()["source"]["path"] == str(new_root / "traindata/one")
    assert (
        len(
            client.get(f"/api/projects/{project['id']}/datasets", params={"version_id": version["id"]}).json()
        )
        == 3
    )
    assert not (old_root / "traindata/one/b.png").exists()


def test_fork_external_same_named_sources_get_readable_disjoint_directories(api, tmp_path):
    client, _, project = api
    one, two = tmp_path / "one/concept", tmp_path / "two/concept"
    image(one / "a.png")
    image(two / "a.png", "blue")
    cfg = client.get(f"/api/projects/{project['id']}/config").json()
    cfg["dataset"]["sources"] = [{"path": str(one)}, {"path": str(two)}]
    assert client.put(f"/api/projects/{project['id']}/config", json=cfg).status_code == 200
    version = fork(api)
    assert version["status"] == "ready", version
    copied = client.get(f"/api/projects/{project['id']}/config", params={"version_id": version["id"]}).json()
    assert {Path(item["path"]).name for item in copied["dataset"]["sources"]} == {"concept", "concept-2"}
    assert len(scan_sources(TrainConfig.model_validate(copied).dataset.sources)) == 2


@pytest.mark.parametrize("via_upload", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_reg_import_never_reuses_a_source_above_role_directories(api, tmp_path, via_upload, registered):
    client, context, project = api
    broad = context.version_dir(project["id"])
    if registered:
        did = routes_work._register_dataset(context, project["id"], routes_work.DatasetBody(path=str(broad)))
        routes_work._index_dataset(context, did)
    else:
        cfg = client.get(f"/api/projects/{project['id']}/config").json()
        cfg["dataset"]["sources"] = [{"path": str(broad)}]
        assert client.put(f"/api/projects/{project['id']}/config", json=cfg).status_code == 200
    source = tmp_path / "concept"
    image(source / "a.png")
    result = (
        upload(api, [("concept/a.png", png())], is_reg="true")
        if via_upload
        else local(api, source, is_reg=True)
    )
    assert result.status_code == 409, result.text
    assert result.json()["error"]["code"] == "dataset.overlap"
    assert list(context.reg_dir(project["id"]).iterdir()) == []
