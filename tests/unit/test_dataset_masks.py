"""Exercise real PNG editing, index refresh and filesystem boundaries through HTTP."""

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.data.images import load_mask
from ypuddin.server import create_app, routes_work
from ypuddin.server import routes_dataset_masks as masks


def png(value=255, mode="L", size=(64, 64)):
    output = io.BytesIO()
    Image.new(mode, size, value).save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    if not any(getattr(route, "path", "").endswith("/mask/info") for route in app.routes):
        app.include_router(masks.router, prefix="/api")
    client = TestClient(app)
    pid = client.post("/api/projects", json={"name": "Mask project"}).json()["id"]
    response = client.post(
        f"/api/projects/{pid}/datasets/upload",
        files=[("files", ("a.png", png((30, 40, 50), "RGB"), "image/png"))],
    )
    assert response.status_code == 200, response.text
    source = response.json()["source"]
    image = client.get(f"/api/datasets/{source['id']}/images").json()["items"][0]
    endpoint = f"/api/datasets/{source['id']}/images/{image['hash']}/mask"
    yield client, app.state.ctx, source, endpoint
    client.close()
    app.state.ctx.db.close()


def save(api, body, revision=None, **kwargs):
    client, _, _, endpoint = api
    if revision is None:
        revision = client.get(endpoint + "/info").json()["revision"]
    return client.put(
        endpoint,
        files={"file": ("ignored-name.png", body, "image/png")},
        data={"revision": revision},
        **kwargs,
    )


def test_default_save_read_index_and_training_weights(api):
    client, context, source, endpoint = api
    info = client.get(endpoint + "/info").json()
    assert info["source"] == "full" and info["coverage"] == 1 and not info["has_mask"]
    assert Image.open(io.BytesIO(client.get(endpoint).content)).getextrema() == (255, 255)
    revision = info["revision"]
    response = save(api, png(0), revision)
    assert response.status_code == 200, response.text
    assert response.json()["coverage"] == 0 and response.json()["source"] == "sidecar"
    path = Path(source["path"]) / "a.mask.png"
    assert path.is_file() and not (path.parent / "ignored-name.png").exists()
    assert list(path.parent.glob("*.tmp")) == []
    assert Image.open(path).mode == "L"
    assert load_mask(str(path), None, 64, 64).sum().item() == 0
    assert client.get(f"/api/datasets/{source['id']}").json()["stats"]["masks"] == 1
    images = client.get(f"/api/datasets/{source['id']}/images").json()["items"]
    assert len(images) == 1 and images[0]["has_mask"]
    assert save(api, png(128), revision).status_code == 409  # no stale overwrite
    assert save(api, png(128)).status_code == 200
    assert 0.50 < load_mask(str(path), None, 64, 64).mean().item() < 0.51
    assert client.get(endpoint).headers["cache-control"] == "no-store"
    assert len(json.loads(routes_work._records_path(context, source["id"]).read_text())) == 1


def test_alpha_fallback_and_existing_legacy_mask(api):
    client, context, source, endpoint = api
    image = Path(source["path"]) / "a.png"
    image.write_bytes(png((20, 40, 60, 128), "RGBA"))
    routes_work._index_dataset(context, source["id"])
    h = client.get(f"/api/datasets/{source['id']}/images").json()["items"][0]["hash"]
    endpoint = f"/api/datasets/{source['id']}/images/{h}/mask"
    info = client.get(endpoint + "/info").json()
    assert info["source"] == "alpha" and 0.50 < info["coverage"] < 0.51
    assert Image.open(io.BytesIO(client.get(endpoint).content)).getextrema() == (128, 128)
    source_png = Image.open(io.BytesIO(client.get(endpoint + "/source").content))
    assert (
        source_png.mode == "RGB" and source_png.getpixel((0, 0))[0] > 100
    )  # transparency composited on white like training
    image.with_suffix(".mask").write_bytes(png(64, size=(32, 32)))
    info = client.get(endpoint + "/info").json()
    assert info["source"] == "sidecar" and info["resized"]
    assert Image.open(io.BytesIO(client.get(endpoint).content)).size == (64, 64)


@pytest.mark.parametrize("body", [b"not a PNG", png(0, size=(32, 32))])
def test_invalid_or_wrong_dimensions_never_create_file(api, body):
    assert save(api, body).status_code == 400
    assert not (Path(api[2]["path"]) / "a.mask.png").exists()


def test_limits_and_busy_index_or_job(api, monkeypatch):
    client, context, source, endpoint = api
    monkeypatch.setattr(masks, "MAX_REQUEST_BYTES", 32)
    assert save(api, png(0)).status_code == 413
    monkeypatch.setattr(masks, "MAX_REQUEST_BYTES", 32 * 1024**2)
    monkeypatch.setattr(masks, "MAX_PIXELS", 100)
    assert client.get(endpoint + "/info").status_code == 413
    monkeypatch.setattr(masks, "MAX_PIXELS", 16_777_216)
    context.db.update("datasets", source["id"], {"index_status": "indexing"})
    assert save(api, png(0)).status_code == 409
    context.db.update("datasets", source["id"], {"index_status": "ready"})
    context.db.insert(
        "jobs",
        {
            "id": "j_busy",
            "name": "active",
            "type": "train",
            "project_id": source["project_id"],
            "status": "running",
            "created_at": 0,
        },
    )
    assert save(api, png(0)).status_code == 409
    assert not (Path(source["path"]) / "a.mask.png").exists()


def test_unknown_ids_paths_symlinks_and_allowed_roots(api, tmp_path):
    client, context, source, endpoint = api
    assert client.get(endpoint.replace(source["id"], "d_missing") + "/info").status_code == 404
    assert client.get(endpoint + "/info", params={"rel_path": "../outside.png"}).status_code == 404
    outside = tmp_path / "outside.png"
    outside.write_bytes(png(30))
    target = Path(source["path"]) / "a.mask.png"
    target.symlink_to(outside)
    assert client.get(endpoint + "/info").status_code == 403
    target.unlink()
    image = target.with_name("a.png")
    image.unlink()
    image.symlink_to(outside)
    assert client.get(endpoint + "/info").status_code == 403
    image.unlink()
    image.write_bytes(png((30, 40, 50), "RGB"))
    context.allowed_roots = [tmp_path / "restricted"]
    # Managed uploads remain permitted through data_root; externally registered sources do not.
    context.db.update("datasets", source["id"], {"path": str(tmp_path)})
    records = json.loads(routes_work._records_path(context, source["id"]).read_text())
    records[0]["path"] = str(outside)
    routes_work._records_path(context, source["id"]).write_text(json.dumps(records))
    assert client.get(endpoint + "/info").status_code == 403
    assert Image.open(outside).getpixel((0, 0)) == 30


def test_duplicate_hash_uses_explicit_indexed_relative_path(api):
    client, context, source, endpoint = api
    root = Path(source["path"])
    (root / "b.png").write_bytes((root / "a.png").read_bytes())
    routes_work._index_dataset(context, source["id"])
    assert client.get(endpoint + "/info").status_code == 409
    info = client.get(endpoint + "/info", params={"rel_path": "b.png"}).json()
    response = save(api, png(0), info["revision"], params={"rel_path": "b.png"})
    assert response.status_code == 200, response.text
    assert (root / "b.mask.png").exists() and not (root / "a.mask.png").exists()


def test_exif_orientation_matches_preview_mask_and_saved_dimensions(api):
    client, context, source, _ = api
    root = Path(source["path"])
    exif = Image.Exif()
    exif[274] = 6
    image = Image.new("RGB", (64, 32), "red")
    image.save(root / "oriented.jpg", exif=exif)
    routes_work._index_dataset(context, source["id"])
    entry = next(
        item
        for item in client.get(f"/api/datasets/{source['id']}/images").json()["items"]
        if item["rel_path"] == "oriented.jpg"
    )
    endpoint = f"/api/datasets/{source['id']}/images/{entry['hash']}/mask"
    info = client.get(endpoint + "/info").json()
    assert (info["width"], info["height"]) == (32, 64)
    assert Image.open(io.BytesIO(client.get(endpoint + "/source").content)).size == (32, 64)
    response = save((client, context, source, endpoint), png(128, size=(32, 64)))
    assert response.status_code == 200, response.text
    assert Image.open(root / "oriented.mask.png").size == (32, 64)
