"""Real multipart/ZIP ingestion, rollback, and project-to-training source contracts."""

import io
import stat
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import TrainConfig
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
    root = context.version_dir(pid) / "datasets"
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
    assert path.is_relative_to(context.version_dir(pid) / "datasets")
    assert (path / "folder/a.txt").read_text() == "角色 caption"
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


def test_zip_nested_paths_and_new_upload_never_overwrites(api):
    first = upload(api, [("images.zip", archive([("set/a.png", png()), ("set/a.txt", "first")]))])
    assert first.status_code == 200, first.text
    original = Path(first.json()["source"]["path"])
    second = upload(api, [("set/a.png", png("blue")), ("set/a.txt", b"second")])
    assert second.status_code == 200, second.text
    assert Path(second.json()["source"]["path"]) != original
    assert (original / "set/a.txt").read_text() == "first"
    client, _, pid = api
    assert len(client.get(f"/api/projects/{pid}/config").json()["dataset"]["sources"]) == 2


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
        (context.version_dir(pid) / "datasets").symlink_to(outside, target_is_directory=True)
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
    assert duplicate.status_code == 409
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
