"""Real multipart/ZIP ingestion, rollback, and project-to-training source contracts."""

import io
import json
import stat
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import TrainConfig
from ypuddin.data.captions import read_caption
from ypuddin.data.index import scan_sources
from ypuddin.server import create_app, dataset_uploads, routes_work
from ypuddin.train.plan import plan


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    pid = client.post("/api/projects", json={"name": "上传训练"}).json()["id"]
    yield client, app.state.ctx, pid
    client.close()
    app.state.ctx.db.close()


def png(color="red", mode="RGB"):
    out = io.BytesIO()
    Image.new(mode, (64, 64), color=color).save(out, format="PNG")
    return out.getvalue()


def archive(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in entries:
            zf.writestr(name, body)
    return output.getvalue()


def upload(api, entries, **fields):
    client, _, pid = api
    return client.post(
        f"/api/projects/{pid}/datasets/upload",
        files=[("files", (name, body, "application/octet-stream")) for name, body in entries],
        data=fields,
    )


def assert_no_import(api):
    client, context, pid = api
    assert client.get(f"/api/projects/{pid}/datasets").json() == []
    assert client.get(f"/api/projects/{pid}/config").json()["dataset"]["sources"] == []
    root = context.dataset_dir(pid)
    assert not root.exists() or list(root.iterdir()) == []


def test_upload_images_captions_masks_and_train_config(api):
    client, context, pid = api
    response = upload(
        api,
        [
            ("folder/a.PNG", png()),
            ("folder/a.TXT", "角色 caption".encode()),
            ("folder/a.MASK.PNG", png(128, "L")),
        ],
        name="角色 / 训练",
        repeats="3",
    )
    assert response.status_code == 200, response.text
    source = response.json()["source"]
    path = Path(source["path"])
    assert path.is_relative_to(context.dataset_dir(pid))
    assert path == context.dataset_dir(pid) / "folder"
    assert (path / "a.txt").read_text() == "角色 caption"
    info = client.get(f"/api/datasets/{source['id']}").json()
    assert info["index_status"] == "ready"
    assert info["stats"]["images"] == info["stats"]["captioned"] == info["stats"]["masks"] == 1
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert cfg["dataset"]["sources"][0]["path"] == str(path)
    assert cfg["dataset"]["sources"][0]["repeats"] == 3
    cfg["model"]["family"] = "toy"
    cfg["model"]["dtype"] = "fp32"
    cfg["dataset"].update(resolutions=[64], bucket_step=16, num_workers=0)
    cfg["loop"].update(epochs=1, mixed_precision="no")
    result = plan(TrainConfig.model_validate(cfg), device="cpu")
    assert result["ok"], result
    assert result["images"] == 1


def test_zip_nested_paths_and_reimport_never_overwrites(api):
    first = upload(api, [("images.zip", archive([("set/a.png", png()), ("set/a.txt", "first")]))])
    assert first.status_code == 200, first.text
    original = Path(first.json()["source"]["path"])
    second = upload(api, [("set/a.png", png("blue")), ("set/a.txt", b"second")])
    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "upload.conflict"
    assert original.name == "set"
    assert (original / "a.txt").read_text() == "first"
    client, _, pid = api
    assert len(client.get(f"/api/projects/{pid}/config").json()["dataset"]["sources"]) == 1


@pytest.mark.parametrize("as_zip", [False, True])
@pytest.mark.parametrize("is_reg", [False, True])
def test_concept_folders_keep_paths_and_matching_sidecars(api, as_zip, is_reg):
    client, context, pid = api
    entries = [
        ("角色/a.png", png()),
        ("角色/a.txt", "红衣角色".encode()),
        ("角色/detail/a.png", png("green")),
        ("角色/detail/a.mask.png", png(128, "L")),
        ("画风/a.png", png("blue")),
        ("画风/a.json", json.dumps({"tags": ["watercolor"], "nl": "A blue subject."}).encode()),
    ]
    if as_zip:
        entries = [("unrelated_archive_name.zip", archive(entries))]
    response = upload(api, entries, name="must-not-wrap-existing-folders", is_reg=str(is_reg).lower())
    assert response.status_code == 200, response.text
    datasets = response.json()["datasets"]
    assert len(datasets) == 2
    assert response.json()["source"] == datasets[0]["source"]
    root = context.dataset_dir(pid, is_reg=is_reg)
    assert {Path(item["source"]["path"]) for item in datasets} == {root / "角色", root / "画风"}
    assert (root / "角色/detail/a.mask.png").is_file()
    assert (root / "角色/a.txt").read_text() == "红衣角色"
    assert read_caption(root / "画风/a.json") == "watercolor. A blue subject."
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert len(cfg["dataset"]["sources"]) == 2
    assert all(item["is_reg"] == is_reg for item in cfg["dataset"]["sources"])
    records = scan_sources(TrainConfig.model_validate(cfg).dataset.sources)
    assert len(records) == 3 and len({record.path for record in records}) == 3
    info = [client.get(f"/api/datasets/{item['source']['id']}").json() for item in datasets]
    assert sum(item["stats"]["images"] for item in info) == 3
    assert sum(item["stats"]["captioned"] for item in info) == 2
    assert sum(item["stats"]["masks"] for item in info) == 1


def test_loose_files_get_one_automatic_folder_beside_real_folders(api):
    client, context, pid = api
    response = upload(
        api,
        [("concept/a.png", png()), ("loose.png", png("blue")), ("loose.txt", b"loose caption")],
    )
    assert response.status_code == 200, response.text
    root = context.dataset_dir(pid)
    assert {path.name for path in root.iterdir()} == {"concept", "images"}
    assert (root / "images/loose.txt").read_text() == "loose caption"
    second = upload(api, [("b.png", png())])
    assert second.status_code == 200, second.text
    assert Path(second.json()["source"]["path"]) == root / "images-2"
    assert len(client.get(f"/api/projects/{pid}/config").json()["dataset"]["sources"]) == 3


def test_same_folder_sync_reuses_source_and_preserves_training_options(api):
    client, context, pid = api
    first = upload(api, [("concept/a.png", png()), ("concept/a.txt", b"caption")], repeats="7")
    assert first.status_code == 200, first.text
    before = first.json()["source"]
    response = upload(
        api,
        [("concept/a.png", png()), ("concept/a.txt", b"caption"), ("concept/nested/b.png", png("blue"))],
        repeats="2",
    )
    assert response.status_code == 200, response.text
    assert response.json()["source"]["id"] == before["id"]
    assert response.json()["source"]["repeats"] == 7
    assert (context.dataset_dir(pid) / "concept/nested/b.png").is_file()
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert len(cfg["dataset"]["sources"]) == 1 and cfg["dataset"]["sources"][0]["repeats"] == 7
    info = client.get(f"/api/datasets/{before['id']}").json()
    assert info["index_status"] == "ready" and info["stats"]["images"] == 2


def test_one_conflict_rejects_all_new_files_and_keeps_existing_batch(api):
    client, context, pid = api
    assert upload(api, [("existing/a.png", png())]).status_code == 200
    cfg = client.get(f"/api/projects/{pid}/config").json()
    response = upload(api, [("new/b.png", png()), ("existing/a.png", png("blue"))])
    assert response.status_code == 409, response.text
    root = context.dataset_dir(pid)
    assert {path.name for path in root.iterdir()} == {"existing"}
    assert (root / "existing/a.png").read_bytes() == png()
    assert client.get(f"/api/projects/{pid}/config").json() == cfg


@pytest.mark.parametrize("incoming", ["concept/a.jpg", "CONCEPT/b.png", "concept/a.png/child.png"])
def test_sync_rejects_existing_stem_case_and_file_directory_conflicts(api, incoming):
    _, context, pid = api
    assert upload(api, [("concept/a.png", png())]).status_code == 200
    response = upload(api, [("new/z.png", png()), (incoming, png("blue"))])
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "upload.conflict"
    root = context.dataset_dir(pid)
    assert not (root / "new").exists()
    assert (root / "concept/a.png").read_bytes() == png()


def test_sync_does_not_follow_existing_nested_directory_symlink(api, tmp_path):
    _, context, pid = api
    assert upload(api, [("concept/a.png", png())]).status_code == 200
    outside = tmp_path / "outside"
    outside.mkdir()
    link = context.dataset_dir(pid) / "concept/nested"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    response = upload(api, [("new/z.png", png()), ("concept/nested/b.png", png("blue"))])
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "upload.path"
    assert list(outside.iterdir()) == []
    assert link.is_symlink()
    assert not (context.dataset_dir(pid) / "new").exists()


def test_registration_failure_rolls_back_only_new_files_in_existing_folder(api, monkeypatch):
    client, context, pid = api
    assert upload(api, [("existing/a.png", png())]).status_code == 200
    cfg = client.get(f"/api/projects/{pid}/config").json()

    def fail_write(*_args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(routes_work, "_write_project_config", fail_write)
    response = upload(api, [("existing/a.png", png()), ("existing/new/b.png", png("blue"))])
    assert response.status_code == 400, response.text
    root = context.dataset_dir(pid)
    assert (root / "existing/a.png").read_bytes() == png()
    assert not (root / "existing/new").exists()
    assert client.get(f"/api/projects/{pid}/config").json() == cfg
    sources = client.get(f"/api/projects/{pid}/datasets").json()
    assert len(sources) == 1 and sources[0]["index_status"] == "ready"


def test_second_concept_database_failure_rolls_back_entire_batch(api, monkeypatch):
    _, context, _ = api
    original = context.db.insert
    calls = 0

    def fail_second(table, row):
        nonlocal calls
        if table == "datasets":
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated second concept failure")
        return original(table, row)

    monkeypatch.setattr(context.db, "insert", fail_second)
    response = upload(api, [("one/a.png", png()), ("two/a.png", png("blue"))])
    assert response.status_code == 400, response.text
    assert_no_import(api)


@pytest.mark.parametrize("registered", [True, False])
def test_existing_ancestor_source_is_reused_without_duplicate_scanning(api, registered):
    client, context, pid = api
    root = context.dataset_dir(pid)
    if registered:
        did = routes_work._register_dataset(context, pid, routes_work.DatasetBody(path=str(root), repeats=9))
        routes_work._index_dataset(context, did)
    else:
        cfg = client.get(f"/api/projects/{pid}/config").json()
        cfg["dataset"]["sources"] = [{"path": str(root), "repeats": 9}]
        assert client.put(f"/api/projects/{pid}/config", json=cfg).status_code == 200
    response = upload(api, [("one/a.png", png()), ("two/a.png", png("blue"))])
    assert response.status_code == 200, response.text
    assert len(response.json()["datasets"]) == 1
    assert response.json()["source"]["path"] == str(root)
    assert response.json()["source"]["repeats"] == 9
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert len(cfg["dataset"]["sources"]) == 1
    records = scan_sources(TrainConfig.model_validate(cfg).dataset.sources)
    assert len(records) == 2 and len({record.path for record in records}) == 2


def test_existing_child_source_blocks_parent_import_without_leaving_files(api):
    client, context, pid = api
    child = context.dataset_dir(pid) / "collection/child"
    child.mkdir(parents=True)
    (child / "a.png").write_bytes(png())
    did = routes_work._register_dataset(context, pid, routes_work.DatasetBody(path=str(child)))
    routes_work._index_dataset(context, did)
    response = upload(api, [("collection/other/b.png", png("blue"))])
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "dataset.overlap"
    assert not (child.parent / "other").exists()
    assert (child / "a.png").read_bytes() == png()
    assert len(client.get(f"/api/projects/{pid}/datasets").json()) == 1


def test_same_named_concepts_remain_isolated_between_versions_and_roles(api):
    client, context, pid = api
    first = upload(api, [("concept/a.png", png())])
    assert first.status_code == 200, first.text
    original = Path(first.json()["source"]["path"])
    reg = upload(api, [("concept/a.png", png("green"))], is_reg="true")
    assert reg.status_code == 200, reg.text
    assert Path(reg.json()["source"]["path"]) == context.reg_dir(pid) / "concept"
    response = client.post(f"/api/projects/{pid}/versions", json={"name": "empty", "data_mode": "empty"})
    assert response.status_code == 202, response.text
    vid = response.json()["id"]
    for _ in range(500):
        version = client.get(f"/api/projects/{pid}/versions/{vid}").json()
        if version["status"] != "copying":
            break
        time.sleep(0.01)
    assert version["status"] == "ready"
    response = client.post(
        f"/api/projects/{pid}/datasets/upload",
        params={"version_id": vid},
        files=[("files", ("concept/a.png", png("blue"), "image/png"))],
    )
    assert response.status_code == 200, response.text
    assert Path(response.json()["source"]["path"]) == context.dataset_dir(pid, vid) / "concept"
    assert (original / "a.png").read_bytes() == png()


@pytest.mark.parametrize(
    "name",
    [
        "../escape.png",
        "/tmp/escape.png",
        "C:\\escape.png",
        "a/../../escape.png",
        "a/NUL.png",
        "a/foo:bar.png",
        "a./x.png",
    ],
)
def test_dangerous_zip_paths_are_rejected_without_partial_data(api, name):
    response = upload(api, [("bad.zip", archive([("safe.png", png()), (name, png())]))])
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "upload.path"
    assert_no_import(api)


def test_zip_symlink_and_high_compression_rejected(api):
    link = zipfile.ZipInfo("link.png")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    response = upload(api, [("bad.zip", archive([(link, "/outside/file")]))])
    assert response.status_code == 400 and response.json()["error"]["code"] == "upload.zip_entry"
    response = upload(api, [("bomb.zip", archive([("huge.png", b"\0" * (2 * 1024**2))]))])
    assert response.status_code == 413
    assert_no_import(api)


@pytest.mark.parametrize(
    "entries",
    [
        [("a.png", b"not an image")],
        [("a.png", png()), ("missing.txt", b"orphan")],
        [("a.png", png()), ("a.txt", b"\xff\xfe")],
        [("a.png", png()), ("A.png", png("blue"))],
        [("a.png", png()), ("a.jpg", png("blue"))],
        [("a.png", png()), ("malware.exe", b"ignored? no")],
        [("a.txt", b"no image")],
        [("set.zip", archive([("a.png", png())])), ("b.png", png())],
    ],
)
def test_invalid_batches_are_atomic(api, entries):
    response = upload(api, entries)
    assert response.status_code == 400, response.text
    assert "trace_id" in response.json()["error"]
    assert_no_import(api)


def test_request_and_expanded_size_limits(api, monkeypatch):
    monkeypatch.setattr(dataset_uploads, "MAX_UPLOAD_BYTES", 100)
    assert upload(api, [("a.png", png())]).status_code == 413
    assert_no_import(api)
    monkeypatch.setattr(dataset_uploads, "MAX_UPLOAD_BYTES", 1_000_000)
    monkeypatch.setattr(dataset_uploads, "MAX_EXPANDED_BYTES", 50)
    assert upload(api, [("a.zip", archive([("a.png", png())]))]).status_code == 413
    assert_no_import(api)


def test_chunked_request_is_limited_without_content_length(api, monkeypatch):
    client, _, pid = api
    monkeypatch.setattr(dataset_uploads, "MAX_UPLOAD_BYTES", 80)
    body = (
        b'--boundary\r\nContent-Disposition: form-data; name="files"; filename="a.png"\r\n\r\n'
        + png()
        + b"\r\n--boundary--\r\n"
    )
    response = client.post(
        f"/api/projects/{pid}/datasets/upload",
        content=iter([body[:40], body[40:]]),
        headers={"Content-Type": "multipart/form-data; boundary=boundary"},
    )
    assert response.status_code == 413, response.text
    assert_no_import(api)


def test_file_count_and_caption_limits(api, monkeypatch):
    monkeypatch.setattr(dataset_uploads, "MAX_FILES", 1)
    assert upload(api, [("a.png", png()), ("b.png", png())]).status_code == 400
    assert upload(api, [("a.zip", archive([("a.png", png()), ("b.png", png())]))]).status_code == 413
    monkeypatch.setattr(dataset_uploads, "MAX_FILES", 5000)
    monkeypatch.setattr(dataset_uploads, "MAX_CAPTION_BYTES", 2)
    assert upload(api, [("a.png", png()), ("a.txt", b"long")]).status_code == 413
    assert_no_import(api)


def test_invalid_form_and_registration_failure_leave_no_files(api, monkeypatch):
    assert upload(api, [("a.png", png())], repeats="0").status_code == 400
    assert upload(api, [("a.png", png())], repeats="x").status_code == 400
    assert upload(api, [("a.png", png())], repeats="9" * 100).status_code == 400

    def fail_write(*_args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(routes_work, "_write_project_config", fail_write)
    response = upload(api, [("a.png", png())])
    assert response.status_code == 400 and "disk failure" in response.text
    assert_no_import(api)


def test_spooled_files_close_on_midstream_limit(api, monkeypatch):
    import starlette.formparsers as parser_module

    handles = []
    original = parser_module.SpooledTemporaryFile

    def tracked(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(parser_module, "SpooledTemporaryFile", tracked)
    monkeypatch.setattr(dataset_uploads, "MAX_FILE_BYTES", 50)
    response = upload(api, [("a.png", png())])
    assert response.status_code == 413
    assert handles and all(handle.closed for handle in handles)
    assert_no_import(api)


def test_managed_root_cannot_redirect_outside_project(api, tmp_path):
    _, context, pid = api
    outside = tmp_path / "external-upload-target"
    outside.mkdir()
    try:
        context.dataset_dir(pid).rmdir()
        context.dataset_dir(pid).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    response = upload(api, [("a.png", png())])
    assert response.status_code == 400 and response.json()["error"]["code"] == "upload.path"
    assert list(outside.iterdir()) == []


def test_truncated_jpeg_is_rejected_before_registration(api):
    output = io.BytesIO()
    Image.new("RGB", (256, 256), "blue").save(output, format="JPEG")
    response = upload(api, [("broken.jpg", output.getvalue()[:-50])])
    assert response.status_code == 400 and response.json()["error"]["code"] == "upload.image"
    assert_no_import(api)


def test_path_registration_and_delete_keep_external_data(api, tmp_path):
    client, context, pid = api
    data = tmp_path / "external"
    data.mkdir()
    (data / "a.png").write_bytes(png())
    cfg = {"model": {"family": "toy"}, "dataset": {"sources": [{"path": str(data), "resolutions": [64]}]}}
    assert client.put(f"/api/projects/{pid}/config", json=cfg).status_code == 200
    response = client.post(f"/api/projects/{pid}/datasets", json={"path": str(data), "repeats": 4})
    assert response.status_code == 201, response.text
    did = response.json()["source"]["id"]
    cfg = client.get(f"/api/projects/{pid}/config").json()
    assert len(cfg["dataset"]["sources"]) == 1
    assert cfg["dataset"]["sources"][0]["resolutions"] == [64]
    assert cfg["dataset"]["sources"][0]["repeats"] == 4
    duplicate = client.post(f"/api/projects/{pid}/datasets", json={"path": str(data)})
    assert duplicate.status_code == 201
    assert duplicate.json()["source"]["id"] == did
    missing = client.post(f"/api/projects/{pid}/datasets", json={"path": str(tmp_path / "missing")})
    assert missing.status_code == 404
    assert client.delete(f"/api/datasets/{did}").status_code == 200
    assert client.get(f"/api/projects/{pid}/config").json()["dataset"]["sources"] == []
    assert (data / "a.png").exists()
    assert not routes_work._records_path(context, did).exists()


def test_missing_project_upload_does_not_write(api):
    client, context, _ = api
    response = client.post("/api/projects/p_missing/datasets/upload", files={"files": ("a.png", png())})
    assert response.status_code == 404
    assert not context.project_dir("p_missing").exists()


def test_openapi_describes_multipart(api):
    client, _, _ = api
    schema = client.get("/api/openapi.json").json()
    operation = schema["paths"]["/api/projects/{pid}/datasets/upload"]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
