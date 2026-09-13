"""Project categorization and explicit covers never mutate version training data."""

import io
import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from ypuddin.server import create_app, project_covers
from ypuddin.server.db import SCHEMA, Database


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    yield client, app.state.ctx
    client.close()
    for name in ("regularization", "model_downloads", "environment", "dataset_pipeline"):
        getattr(app.state, name).close()
    app.state.ctx.versions.close()
    app.state.ctx.db.close()


def create(client, id="portraits", **fields):
    response = client.post("/api/projects", json={"id": id, "name": id, **fields})
    assert response.status_code == 201, response.text
    return response.json()


def image_bytes(format="PNG", size=(1200, 800), color="red"):
    stream = io.BytesIO()
    options = {}
    if format == "PNG":
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("private-note", "not retained in thumbnail")
        options["pnginfo"] = metadata
    Image.new("RGB", size, color).save(stream, format, **options)
    return stream.getvalue()


def upload(client, pid, payload=None, **kwargs):
    return client.post(
        f"/api/projects/{pid}/cover",
        files={
            "file": (
                "../../v1/traindata/cover.png",
                image_bytes() if payload is None else payload,
                "image/png",
            )
        },
        **kwargs,
    )


def test_category_list_filtering_pagination_and_explicit_clear(api):
    client, _ = api
    a = create(client, "a", name="Cat 100%", category=" 人物 LoRA ")
    b = create(client, "b", name="Cat B", category="人物 LoRA")
    create(client, "c", name="Style", note="cat details", category="画风 LoRA")
    create(client, "d", name="Other")
    assert a["category"] == "人物 LoRA"
    assert a["cover_url"] is None and "cover_key" not in a
    assert client.patch(f"/api/projects/{b['id']}", json={"archived": True}).status_code == 200
    assert len(client.get("/api/projects").json()) == 3
    params = {"include_archived": True, "category": "人物 LoRA", "q": "cat", "page": 1, "page_size": 1}
    first = client.get("/api/projects", params=params).json()
    second = client.get("/api/projects", params={**params, "page": 2}).json()
    assert first["total"] == second["total"] == 2
    assert {first["items"][0]["id"], second["items"][0]["id"]} == {"a", "b"}
    assert [p["id"] for p in client.get("/api/projects", params={"q": "100%"}).json()] == ["a"]
    assert [p["id"] for p in client.get("/api/projects?archived=true").json()] == ["b"]
    assert [p["id"] for p in client.get("/api/projects?uncategorized=true").json()] == ["d"]
    categories = client.get("/api/project-categories").json()
    assert categories == {
        "items": [{"name": "人物 LoRA", "count": 2}, {"name": "画风 LoRA", "count": 1}],
        "uncategorized": 1,
        "total": 4,
    }
    assert client.patch("/api/projects/a", json={"note": "keep category"}).json()["category"] == "人物 LoRA"
    assert client.patch("/api/projects/a", json={"category": None}).json()["category"] is None
    assert client.patch("/api/projects/b", json={"category": "  "}).json()["category"] is None
    assert client.get("/api/project-categories").json()["uncategorized"] == 3
    assert client.get("/api/projects?page=0").status_code == 422
    assert (
        client.get("/api/projects", params={"category": "x\n"}).status_code == 200
    )  # trailing whitespace normalizes
    assert client.get("/api/projects", params={"category": "x\ny"}).status_code == 400
    assert client.post("/api/projects", json={"name": "bad", "category": "x" * 65}).status_code == 422
    assert client.post("/api/projects", json={"name": "bad", "category": "x\ny"}).status_code == 422


@pytest.mark.parametrize("family", ["anima", "krea2", "toy"])
def test_project_and_version_family_api_persist_the_selected_recipe(api, family):
    client, ctx = api
    project = create(client, family=family)
    assert project["active_family"] == family
    version_id = project["active_version_id"]
    original = ctx.config_path(project["id"], version_id).read_bytes()
    assert client.get(f"/api/projects/{project['id']}/config").json()["model"]["family"] == family
    assert client.get(f"/api/projects/{project['id']}/versions/{version_id}").json()["family"] == family
    target = "anima" if family == "krea2" else "krea2"
    response = client.post(
        f"/api/projects/{project['id']}/versions",
        json={
            "name": "Target family",
            "source_version_id": version_id,
            "family": target,
            "data_mode": "empty",
        },
    )
    assert response.status_code == 202, response.text
    new_id = response.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = client.get(f"/api/projects/{project['id']}/versions/{new_id}").json()
        if row["status"] != "copying":
            break
        time.sleep(0.02)
    assert row["status"] == "ready" and row["family"] == target, row
    updated = client.patch(f"/api/projects/{project['id']}", json={"active_version_id": new_id}).json()
    assert updated["active_family"] == target
    assert ctx.config_path(project["id"], version_id).read_bytes() == original


def test_unsupported_family_api_is_rejected_before_any_rows_or_directories(api):
    client, ctx = api
    for family in ("flux3", "sdxl-unknown", "unknown"):
        assert (
            client.post("/api/projects", json={"id": family, "name": family, "family": family}).status_code
            == 422
        )
    assert client.get("/api/projects").json() == []
    project = create(client)
    before = set(ctx.project_dir(project["id"]).rglob("*"))
    for family in ("flux3", "sdxl-unknown", "unknown"):
        response = client.post(
            f"/api/projects/{project['id']}/versions",
            json={"name": family, "family": family, "data_mode": "empty"},
        )
        assert response.status_code == 422
    assert len(client.get(f"/api/projects/{project['id']}/versions").json()) == 1
    assert set(ctx.project_dir(project["id"]).rglob("*")) == before


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_manual_cover_is_reencoded_and_does_not_touch_training_data(api, format):
    client, ctx = api
    project = create(client)
    config = ctx.config_path(project["id"]).read_bytes()
    version_root = ctx.version_dir(project["id"])
    original_files = {
        p.relative_to(version_root): p.read_bytes() for p in version_root.rglob("*") if p.is_file()
    }
    result = upload(client, project["id"], image_bytes(format))
    assert result.status_code == 200, result.text
    row = result.json()
    assert row["cover_url"].startswith(f"/api/projects/{project['id']}/cover?v=")
    response = client.get(row["cover_url"])
    assert response.status_code == 200 and response.headers["content-type"] == "image/webp"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "WEBP" and max(image.size) == 640
        assert "private-note" not in image.info and not image.getexif()
    assert ctx.config_path(project["id"]).read_bytes() == config
    assert {
        p.relative_to(version_root): p.read_bytes() for p in version_root.rglob("*") if p.is_file()
    } == original_files
    assert ctx.db.fetchone("SELECT count(*) n FROM datasets")["n"] == 0
    assert len(list((ctx.project_dir(project["id"]) / ".studio").glob("*.webp"))) == 1
    assert client.get(f"/api/projects/{project['id']}/cover?path=/etc/passwd").content == response.content
    assert (
        "image/webp"
        in client.get("/api/openapi.json").json()["paths"]["/api/projects/{pid}/cover"]["get"]["responses"][
            "200"
        ]["content"]
    )


def test_failed_cover_upload_preserves_project_then_retry_replace_and_delete(api, monkeypatch):
    client, ctx = api
    project = create(client)
    invalid = upload(client, project["id"], b"not an image")
    assert invalid.status_code == 400
    assert client.get(f"/api/projects/{project['id']}").status_code == 200
    assert len(client.get("/api/projects").json()) == 1
    first = upload(client, project["id"]).json()
    before = client.get(first["cover_url"]).content
    stored = ctx.db.fetchone("SELECT * FROM projects WHERE id=?", (project["id"],))
    old_path = project_covers.cover_path(ctx, stored)
    update = ctx.db.update

    def fail_update(table, id_, fields):
        if table == "projects" and fields.get("cover_key"):
            raise OSError("simulated disk/database failure")
        return update(table, id_, fields)

    with monkeypatch.context() as m:
        m.setattr(ctx.db, "update", fail_update)
        failed = upload(
            client,
            project["id"],
            image_bytes(color="blue"),
            data={"crop": json.dumps({"x": 0, "y": 0, "width": 0.5, "height": 0.5})},
        )
        assert failed.status_code == 500 and failed.json()["error"]["code"] == "project.cover_write"
    assert client.get(first["cover_url"]).content == before
    assert old_path.is_file() and list(old_path.parent.iterdir()) == [old_path]
    replacement = upload(client, project["id"], image_bytes(color="blue")).json()
    assert replacement["cover_url"] != first["cover_url"] and not old_path.exists()
    assert client.get(replacement["cover_url"]).content != before
    assert client.delete(f"/api/projects/{project['id']}/cover").json()["cover_url"] is None
    assert client.get(f"/api/projects/{project['id']}/cover").status_code == 404
    assert client.delete(f"/api/projects/{project['id']}/cover").status_code == 200
    assert list(old_path.parent.iterdir()) == []


@pytest.mark.parametrize("orientation", [None, 6])
def test_cover_crop_uses_oriented_source_pixels_before_thumbnailing(api, orientation):
    client, ctx = api
    project = create(client)
    before = {
        p.relative_to(ctx.version_dir(project["id"])): p.read_bytes()
        for p in ctx.version_dir(project["id"]).rglob("*")
        if p.is_file()
    }
    # EXIF 6 rotates the raw top (red) half into the visible right half.
    image = Image.new("RGB", (1000, 1600) if orientation else (1600, 1000), "blue")
    image.paste("red", (0, 0, 1000, 800) if orientation else (800, 0, 1600, 1000))
    output = io.BytesIO()
    exif = Image.Exif()
    if orientation:
        exif[274] = orientation
    image.save(output, "JPEG", exif=exif)
    result = upload(
        client,
        project["id"],
        output.getvalue(),
        data={"crop": json.dumps({"x": 0.5, "y": 0, "width": 0.5, "height": 0.5})},
    )
    assert result.status_code == 200, result.text
    with Image.open(io.BytesIO(client.get(result.json()["cover_url"]).content)) as saved:
        assert saved.size == (640, 400)
        red, green, blue = saved.convert("RGB").getpixel((320, 200))
        assert red > 240 and green < 15 and blue < 15
        assert not saved.getexif()
    assert {
        p.relative_to(ctx.version_dir(project["id"])): p.read_bytes()
        for p in ctx.version_dir(project["id"]).rglob("*")
        if p.is_file()
    } == before


@pytest.mark.parametrize(
    "crop",
    [
        "not JSON",
        "null",
        "[]",
        '{"x":0,"y":0,"width":1}',
        '{"x":0,"y":0,"width":1,"height":1,"extra":0}',
        '{"x":false,"y":0,"width":1,"height":1}',
        '{"x":"0","y":0,"width":1,"height":1}',
        '{"x":NaN,"y":0,"width":1,"height":1}',
        '{"x":0,"y":0,"width":Infinity,"height":1}',
        '{"x":0,"y":0,"width":0,"height":1}',
        '{"x":-0.1,"y":0,"width":1,"height":1}',
        '{"x":0.5,"y":0,"width":0.6,"height":1}',
        '{"x":0,"y":0.5,"width":1,"height":0.6}',
        '{"x":0,"y":0,"width":0.000001,"height":1}',
    ],
)
def test_invalid_crop_never_replaces_an_existing_cover(api, crop):
    client, ctx = api
    project = create(client)
    first = upload(client, project["id"]).json()
    before = client.get(first["cover_url"]).content
    stored = ctx.db.fetchone("SELECT * FROM projects WHERE id=?", (project["id"],))
    old_path = project_covers.cover_path(ctx, stored)
    failed = upload(client, project["id"], data={"crop": crop})
    assert failed.status_code == 400, failed.text
    assert failed.json()["error"]["code"] == "project.cover_crop"
    assert client.get(f"/api/projects/{project['id']}").json()["cover_url"] == first["cover_url"]
    assert client.get(first["cover_url"]).content == before
    assert list(old_path.parent.iterdir()) == [old_path]


def test_crop_multipart_fields_are_bounded_and_float_edge_rounding_is_allowed(api):
    client, _ = api
    project = create(client)
    crop = '{"x":0.1,"y":0,"width":0.9000000000000001,"height":1}'
    assert upload(client, project["id"], data={"crop": crop}).status_code == 200
    assert upload(client, project["id"], data={"crop": " " * 1025}).status_code == 400
    file = ("file", ("image.png", image_bytes(), "image/png"))
    for fields in (
        [("crop", (None, crop)), ("crop", (None, crop))],
        [("crop", ("crop.json", crop, "application/json"))],
        [("unexpected", (None, "value"))],
    ):
        assert client.post(f"/api/projects/{project['id']}/cover", files=[file, *fields]).status_code == 400


def test_cover_accepts_exactly_eight_mib_and_rejects_one_extra_byte(api):
    client, _ = api
    project = create(client)
    assert project_covers.MAX_BYTES == 8 * 1024 * 1024
    data = image_bytes("JPEG").ljust(project_covers.MAX_BYTES, b"\0")
    # The limit is the original file size, excluding the optional crop and multipart headers.
    crop = {"crop": '{"x":0,"y":0,"width":1,"height":1}'}
    accepted = upload(client, project["id"], data, data=crop)
    assert accepted.status_code == 200, accepted.text
    rejected = upload(client, project["id"], data + b"\0", data=crop)
    assert rejected.status_code == 413, rejected.text
    assert rejected.json()["error"]["code"] == "project.cover_size"
    assert client.get(f"/api/projects/{project['id']}").json()["cover_url"] == accepted.json()["cover_url"]


def test_cover_limits_and_symbolic_links_preserve_existing_files(api, tmp_path, monkeypatch):
    client, ctx = api
    project = create(client)
    assert upload(client, project["id"], image_bytes("GIF")).status_code == 415
    assert upload(client, project["id"], image_bytes(size=(8193, 1))).status_code == 413
    with monkeypatch.context() as m:
        m.setattr(project_covers, "MAX_PIXELS", 100)
        assert upload(client, project["id"], image_bytes(size=(11, 10))).status_code == 413
    with monkeypatch.context() as m:
        m.setattr(project_covers, "MAX_BYTES", 10)
        assert upload(client, project["id"]).status_code == 413
    assert (
        client.post(
            f"/api/projects/{project['id']}/cover",
            content=b"",
            headers={
                "content-type": "multipart/form-data; boundary=x",
                "content-length": str(project_covers.MAX_REQUEST_BYTES + 1),
            },
        ).status_code
        == 413
    )
    assert (
        client.post(
            f"/api/projects/{project['id']}/cover",
            files=[("file", ("a.png", image_bytes())), ("file", ("b.png", image_bytes()))],
        ).status_code
        == 400
    )
    assert (
        client.post(
            f"/api/projects/{project['id']}/cover", files={"files": ("a.png", image_bytes())}
        ).status_code
        == 400
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("untouched")
    (ctx.project_dir(project["id"]) / ".studio").symlink_to(outside, target_is_directory=True)
    assert upload(client, project["id"]).status_code == 409
    assert list(outside.iterdir()) == [sentinel] and sentinel.read_text() == "untouched"


def test_existing_database_gets_nullable_metadata_without_moving_legacy_paths(tmp_path):
    root = tmp_path / "studio"
    root.mkdir()
    old = root / "projects/old"
    old.mkdir(parents=True)
    config = old / "config.json"
    config.write_text(json.dumps({"model": {"family": "anima"}, "keep": "legacy"}))
    sentinel = old / "datasets/original.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("original captions")
    c = sqlite3.connect(root / "studio.db")
    c.executescript(SCHEMA)
    c.execute("INSERT INTO projects(id,name,created_at,updated_at) VALUES('old','Original',1,2)")
    c.execute(
        "INSERT INTO jobs(id,type,name,project_id,status,created_at,run_dir,config_json) VALUES('j','train','old','old','completed',1,'/custom/old/run','{}')"
    )
    c.commit()
    c.close()
    before = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    db = Database(root / "studio.db")
    project = db.fetchone("SELECT * FROM projects WHERE id='old'")
    assert project["category"] is project["cover_key"] is None
    assert project["created_at"] == 1 and project["updated_at"] == 2
    assert project["layout_version"] == 1
    assert db.fetchone("SELECT run_dir FROM jobs WHERE id='j'")["run_dir"] == "/custom/old/run"
    db.close()
    db = Database(root / "studio.db")
    assert db.fetchone("SELECT count(*) n FROM project_versions")["n"] == 1
    assert {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()} == before
    assert not (root / "project").exists()
    db.close()
