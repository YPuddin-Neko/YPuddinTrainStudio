"""Real file transactions for image painting, masks and byte-exact recovery."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app, routes_work
from ypuddin.server import routes_dataset_paint as paint
from ypuddin.server.dataset_pipeline import DatasetPipeline


def png(color=(30, 40, 50, 128), mode="RGBA", size=(64, 32)):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Paint", "family": "toy"}).json()
    response = client.post(
        f"/api/projects/{project['id']}/datasets/upload",
        files=[
            ("files", ("a.png", png(), "image/png")),
            ("files", ("a.txt", b"original, caption", "text/plain")),
        ],
    )
    assert response.status_code == 200, response.text
    row = response.json()["source"]
    yield client, app.state.ctx, app.state.dataset_pipeline, row, project
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def endpoint(api, filename="a.png"):
    client, _, _, row, _ = api
    images = client.get(f"/api/datasets/{row['id']}/images").json()["items"]
    image = next(item for item in images if item["rel_path"] == filename)
    return f"/api/datasets/{row['id']}/images/{image['hash']}/paint?rel_path={filename}"


def route(base, suffix):
    path, query = base.split("?")
    return path + suffix + "?" + query


def save(api, *, image=None, mask=None, revision=None, base=None):
    client = api[0]
    base = base or endpoint(api)
    if revision is None:
        response = client.get(route(base, "/info"))
        assert response.status_code == 200, response.text
        revision = response.json()["revision"]
    files = {}
    if image is not None:
        files["file"] = ("not-a-target.png", image, "image/png")
    if mask is not None:
        files["mask"] = ("ignored.png", mask, "image/png")
    return client.put(base, data={"revision": revision}, files=files)


def test_combined_save_changes_hash_preserves_alpha_caption_and_byte_exact_restore(api):
    client, c, manager, row, _ = api
    directory = Path(row["path"])
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    base = endpoint(api)
    original = client.get(route(base, "/info")).json()
    source = Image.open(io.BytesIO(client.get(route(base, "/source")).content))
    assert source.mode == "RGBA" and source.getpixel((0, 0)) == (30, 40, 50, 128)
    response = save(api, image=png((255, 0, 0, 128)), mask=png(0, "L"))
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["image_id"] != original["image_id"] and saved["can_restore"] and saved["has_mask"]
    assert saved["mask_source"] == "sidecar"
    assert Image.open(directory / "a.png").getpixel((0, 0)) == (255, 0, 0, 128)
    assert Image.open(directory / "a.mask.png").getextrema() == (0, 0)
    assert (directory / "a.txt").read_bytes() == before["a.txt"]
    assert (
        c.db.fetchone("SELECT index_status FROM datasets WHERE id=?", (row["id"],))["index_status"] == "ready"
    )
    operation = manager.operation(saved["operation_id"])
    assert operation["action"] == "paint" and operation["status"] == "completed"
    assert len(operation["result"]["changes"]) == 2
    restored = client.post(route(endpoint(api), "/restore"), json={"revision": saved["revision"]})
    assert restored.status_code == 200, restored.text
    assert restored.json()["image_id"] == original["image_id"] and not restored.json()["can_restore"]
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert not manager.operation(saved["operation_id"])["can_undo"]


def test_second_replace_failure_rolls_back_both_files_and_index(api, monkeypatch):
    directory = Path(api[3]["path"])
    old = (directory / "a.png").read_bytes()
    import ypuddin.server.dataset_pipeline as module

    replace = module.os.replace

    def fail_mask(source, target):
        if Path(target).name == "a.mask.png":
            raise OSError("simulated disk failure")
        return replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_mask)
    response = save(api, image=png((0, 255, 0, 128)), mask=png(0, "L"))
    assert response.status_code == 422 and "simulated disk failure" in response.text
    assert (directory / "a.png").read_bytes() == old and not (directory / "a.mask.png").exists()
    assert endpoint(api)
    assert not api[1].resolve_version(api[4]["id"], api[4]["active_version_id"])["busy"]


def test_index_failure_rolls_back_pixels_and_can_retry_from_editor(api, monkeypatch):
    directory = Path(api[3]["path"])
    old = (directory / "a.png").read_bytes()
    original = paint._index_dataset
    calls = []

    def fail_once(c, did):
        calls.append(did)
        if len(calls) == 1:
            c.db.update("datasets", did, {"index_status": "failed"})
        else:
            original(c, did)

    monkeypatch.setattr(paint, "_index_dataset", fail_once)
    response = save(api, image=png((0, 255, 0, 128)))
    assert response.status_code == 422 and (directory / "a.png").read_bytes() == old
    failed = api[1].db.fetchone("SELECT id FROM dataset_pipeline_operations ORDER BY created_at DESC")
    retry = api[0].post(f"/api/dataset-pipeline/operations/{failed['id']}/retry")
    assert retry.status_code == 409 and "reopen" in retry.text
    assert save(api, image=png((0, 255, 0, 128))).status_code == 200


def test_joint_revision_detects_sidecar_changes_and_restore_backup_corruption(api):
    client, _, manager, row, _ = api
    base = endpoint(api)
    revision = client.get(route(base, "/info")).json()["revision"]
    directory = Path(row["path"])
    (directory / "a.mask.png").write_bytes(png(0, "L"))
    assert save(api, image=png(), revision=revision).status_code == 409
    saved = save(api, image=png((0, 0, 255, 128)), mask=png(255, "L")).json()
    operation = manager.operation(saved["operation_id"])
    Path(operation["result"]["changes"][0]["backup"]).write_bytes(b"damaged")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    response = client.post(route(endpoint(api), "/restore"), json={"revision": saved["revision"]})
    assert response.status_code == 409 and "backup" in response.text
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert manager.operation(saved["operation_id"])["can_undo"]


@pytest.mark.parametrize(
    "image,mask", [(png(size=(32, 32)), None), (None, png(size=(64, 32))), (b"invalid", None)]
)
def test_invalid_uploads_preserve_original(api, image, mask):
    path = Path(api[3]["path"]) / "a.png"
    before = path.read_bytes()
    response = save(api, image=image, mask=mask)
    assert response.status_code == 400 and path.read_bytes() == before


def test_active_job_and_archived_version_reject_save(api):
    client, c, _, _, project = api
    c.db.insert(
        "jobs",
        {
            "id": "j_busy",
            "type": "train",
            "name": "busy",
            "project_id": project["id"],
            "version_id": project["active_version_id"],
            "status": "queued",
            "created_at": 0,
        },
    )
    assert save(api, image=png()).status_code == 409
    c.db.execute("DELETE FROM jobs WHERE id='j_busy'")
    c.db.update("project_versions", project["active_version_id"], {"archived": 1})
    assert save(api, mask=png(0, "L")).status_code == 409


def test_shared_sidecar_and_symlink_are_rejected_without_partial_rgb_write(api, tmp_path):
    _, c, _, row, _ = api
    directory = Path(row["path"])
    Image.new("RGB", (64, 32)).save(directory / "a.jpg")
    before = (directory / "a.png").read_bytes()
    response = save(api, image=png((0, 0, 0, 128)), mask=png(0, "L"))
    assert response.status_code == 409 and "shares" in response.text
    assert (directory / "a.png").read_bytes() == before
    external = tmp_path / "external.mask.png"
    external.write_bytes(png(255, "L"))
    (directory / "a.mask.png").symlink_to(external)
    assert save(api, image=png(), revision="ignored").status_code == 403
    assert Image.open(external).getextrema() == (255, 255)


def test_exif_orientation_alpha_and_jpeg_extension(api):
    client, c, _, row, _ = api
    directory = Path(row["path"])
    (directory / "a.png").unlink()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (32, 64), "red").save(directory / "a.jpg", exif=exif)
    routes_work._index_dataset(c, row["id"])
    base = endpoint(api, "a.jpg")
    assert Image.open(io.BytesIO(client.get(route(base, "/source")).content)).size == (64, 32)
    assert save(api, image=png(), base=base).status_code == 400  # JPEG cannot store non-opaque alpha.
    response = save(api, image=png((0, 255, 0, 255)), base=base)
    assert response.status_code == 200, response.text
    with Image.open(directory / "a.jpg") as image:
        assert image.format == "JPEG" and image.size == (64, 32) and image.getexif().get(274) in (None, 1)
    assert not (directory / "a.png").exists()


def test_interrupted_transaction_recovered_from_durable_journal(api):
    _, c, manager, row, _ = api
    directory = Path(row["path"])
    before = (directory / "a.png").read_bytes()
    saved = save(api, image=png((255, 255, 0, 128)), mask=png(0, "L")).json()
    c.db.update("dataset_pipeline_operations", saved["operation_id"], {"status": "running"})
    restarted = DatasetPipeline(c)
    try:
        assert (directory / "a.png").read_bytes() == before
        assert not (directory / "a.mask.png").exists()
        assert restarted.operation(saved["operation_id"])["status"] == "failed"
        assert endpoint(api)
    finally:
        restarted.close()


def test_completion_failure_restores_undo_marker_and_keeps_saved_pixels(api, monkeypatch):
    client, c, manager, row, _ = api
    saved = save(api, image=png((0, 0, 255, 128)), mask=png(0, "L")).json()
    directory = Path(row["path"])
    before_restore = {p.name: p.read_bytes() for p in directory.iterdir()}
    original_update = c.db.update

    def fail_restore_completion(table, oid, fields):
        if (
            table == "dataset_pipeline_operations"
            and oid != saved["operation_id"]
            and fields.get("status") == "completed"
        ):
            raise OSError("database completion failure")
        return original_update(table, oid, fields)

    monkeypatch.setattr(c.db, "update", fail_restore_completion)
    response = client.post(route(endpoint(api), "/restore"), json={"revision": saved["revision"]})
    assert response.status_code == 422
    assert before_restore == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert manager.operation(saved["operation_id"])["can_undo"]


def test_version_lease_blocks_other_mutation_during_paint(api, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    client, c, _, _, project = api
    entered, proceed = threading.Event(), threading.Event()
    original_encode = paint._encode

    def held_encode(*args):
        entered.set()
        assert proceed.wait(5)
        return original_encode(*args)

    monkeypatch.setattr(paint, "_encode", held_encode)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(save, api, image=png((0, 0, 255, 128)))
        assert entered.wait(5)
        try:
            assert c.resolve_version(project["id"], project["active_version_id"])["busy"]
            assert save(api, mask=png(0, "L")).status_code == 409
            response = client.put(
                f"/api/projects/{project['id']}/config",
                params={"version_id": project["active_version_id"]},
                json={"model": {"family": "toy"}},
            )
            assert response.status_code == 409
        finally:
            proceed.set()
        assert future.result().status_code == 200
    assert not c.resolve_version(project["id"], project["active_version_id"])["busy"]


def test_image_symlink_cannot_redirect_paint_to_another_image(api):
    _, c, _, row, _ = api
    directory = Path(row["path"])
    original = (directory / "a.png").read_bytes()
    (directory / "b.png").write_bytes(original)
    (directory / "a.png").unlink()
    (directory / "a.png").symlink_to(directory / "b.png")
    # Simulate an already-indexed path becoming a symlink after it was displayed.
    response = save(
        api,
        image=png((0, 0, 255, 128)),
        revision="ignored",
        base=f"/api/datasets/{row['id']}/images/{paint.content_hash(directory / 'b.png')}/paint?rel_path=a.png",
    )
    assert response.status_code == 403
    assert (directory / "b.png").read_bytes() == original


def test_completed_paint_survives_recovery_and_legacy_mask_restores_exactly(api):
    client, c, _, row, _ = api
    directory = Path(row["path"])
    (directory / "a.mask").write_bytes(png(64, "L"))
    saved = save(api, mask=png(192, "L")).json()
    restarted = DatasetPipeline(c)
    try:
        assert restarted.operation(saved["operation_id"])["status"] == "completed"
        assert Image.open(directory / "a.mask.png").getextrema() == (192, 192)
        response = client.post(route(endpoint(api), "/restore"), json={"revision": saved["revision"]})
        assert response.status_code == 200, response.text
        assert not (directory / "a.mask.png").exists()
        assert (directory / "a.mask").read_bytes() == png(64, "L")
    finally:
        restarted.close()


def test_canvas_rgba_mask_is_accepted_only_when_opaque_grayscale(api):
    response = save(api, mask=png((80, 80, 80, 255)))
    assert response.status_code == 200, response.text
    mask = Path(api[3]["path"]) / "a.mask.png"
    assert Image.open(mask).mode == "L" and Image.open(mask).getextrema() == (80, 80)
    before = mask.read_bytes()
    for color in ((80, 80, 80, 128), (80, 90, 80, 255)):
        response = save(api, mask=png(color))
        assert response.status_code == 400 and mask.read_bytes() == before


def test_restore_detects_changes_to_layer_not_in_original_upload(api):
    client, _, _, row, _ = api
    saved = save(api, image=png((0, 0, 255, 128))).json()
    (Path(row["path"]) / "a.mask.png").write_bytes(png(0, "L"))
    info = client.get(route(endpoint(api), "/info")).json()
    assert not info["can_restore"]
    response = client.post(route(endpoint(api), "/restore"), json={"revision": info["revision"]})
    assert response.status_code == 409
    assert Image.open(Path(row["path"]) / "a.png").getpixel((0, 0)) == (0, 0, 255, 128)
    assert saved["operation_id"]


def test_generic_pipeline_undo_uses_same_joint_guards_and_atomic_completion(api, monkeypatch):
    import time

    client, c, manager, row, project = api
    saved = save(api, image=png((0, 0, 255, 128)), mask=png(80, "L")).json()
    original_update = c.db.update

    def fail_restore_completion(table, oid, fields):
        if (
            table == "dataset_pipeline_operations"
            and oid != saved["operation_id"]
            and fields.get("status") == "completed"
        ):
            raise OSError("simulated completion failure")
        return original_update(table, oid, fields)

    monkeypatch.setattr(c.db, "update", fail_restore_completion)
    response = client.post(
        f"/api/projects/{project['id']}/versions/{project['active_version_id']}/pipeline/operations",
        json={"action": "restore", "restore_operation_id": saved["operation_id"]},
    )
    assert response.status_code == 202, response.text
    oid = response.json()["id"]
    deadline = time.monotonic() + 10
    while (
        manager.operation(oid)["status"] not in {"failed", "completed"}
        or c.resolve_version(project["id"], project["active_version_id"])["busy"]
    ):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert manager.operation(oid)["status"] == "failed"
    assert manager.operation(saved["operation_id"])["can_undo"]
    assert Image.open(Path(row["path"]) / "a.png").getpixel((0, 0)) == (0, 0, 255, 128)
    monkeypatch.setattr(c.db, "update", original_update)
    (Path(row["path"]) / "a.mask.png").write_bytes(png(0, "L"))
    response = client.post(
        f"/api/projects/{project['id']}/versions/{project['active_version_id']}/pipeline/operations",
        json={"action": "restore", "restore_operation_id": saved["operation_id"]},
    )
    oid = response.json()["id"]
    deadline = time.monotonic() + 10
    while (
        manager.operation(oid)["status"] not in {"failed", "completed"}
        or c.resolve_version(project["id"], project["active_version_id"])["busy"]
    ):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert manager.operation(oid)["status"] == "failed"
    assert "image or mask" in manager.operation(oid)["error"]
    assert Image.open(Path(row["path"]) / "a.mask.png").getextrema() == (0, 0)
