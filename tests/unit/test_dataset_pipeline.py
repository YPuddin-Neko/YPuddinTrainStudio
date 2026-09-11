"""Version pipeline executes reversible edits and delegates actual encoder caching."""

import threading
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
    project = client.post("/api/projects", json={"name": "Pipeline"}).json()
    yield client, app.state.ctx, app.state.dataset_pipeline, project
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def imported(api, tmp_path, *, mask=True):
    client, _, _, project = api
    root = tmp_path / "images"
    root.mkdir()
    image = Image.new("RGB", (80, 40), "red")
    image.save(root / "a.png")
    image.save(root / "b.png")
    (root / "a.txt").write_text("red, portrait", encoding="utf8")
    if mask:
        mask_image = Image.new("L", (80, 40), 0)
        mask_image.paste(255, (20, 0, 60, 40))
        mask_image.save(root / "a.mask.png")
    response = client.post(f"/api/projects/{project['id']}/datasets", json={"path": str(root)})
    assert response.status_code == 201, response.text
    row = response.json()["source"]
    return row, Path(row["path"])


def finish(api, response):
    client, c, _, p = api
    assert response.status_code == 202, response.text
    oid = response.json()["id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        row = client.get(f"/api/dataset-pipeline/operations/{oid}").json()
        if (
            row["status"] in {"completed", "failed", "cancelled"}
            and not c.resolve_version(p["id"], p["active_version_id"])["busy"]
        ):
            return row
        if row["job_id"]:
            return row
        time.sleep(0.01)
    pytest.fail(f"operation timed out: {row}")


def start(api, action, **options):
    client, _, _, p = api
    return finish(
        api,
        client.post(
            f"/api/projects/{p['id']}/versions/{p['active_version_id']}/pipeline/operations",
            json={"action": action, **options},
        ),
    )


def refs(row, *names):
    return [{"dataset_id": row["id"], "rel_path": name} for name in names]


def snapshot(api):
    client, _, _, p = api
    result = client.get(f"/api/projects/{p['id']}/versions/{p['active_version_id']}/pipeline")
    assert result.status_code == 200, result.text
    return result.json()


def test_quality_inspects_pixels_duplicates_sidecars_and_metadata_staleness(api, tmp_path, monkeypatch):
    row, root = imported(api, tmp_path)
    (root / "broken.png").write_bytes(b"not an image")
    op = start(api, "inspect")
    assert op["status"] == "completed", op
    report = op["result"]["inspection"]
    assert len(report["images"]) == 3 and len(report["duplicate_groups"]) == 1
    assert report["errors"] == 1 and report["captioned"] == 1 and report["masks"] == 1
    # Polling must not decode images, even when a cached report is returned.
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("poll decoded an image"))
    assert snapshot(api)["inspection"]["errors"] == 1
    assert "inspection" not in snapshot(api)["operations"][0]["result"]
    (root / "b.txt").write_text("new caption")
    assert snapshot(api)["stale"] and snapshot(api)["inspection"] is None


def test_exclusion_undo_preserves_original_import_and_shared_sidecars(api, tmp_path):
    row, root = imported(api, tmp_path)
    original = {p.name: p.read_bytes() for p in root.iterdir()}
    op = start(api, "exclude", images=refs(row, "a.png"))
    assert op["status"] == "completed", op
    assert (
        not (root / "a.png").exists() and not (root / "a.txt").exists() and not (root / "a.mask.png").exists()
    )
    assert (tmp_path / "images" / "a.png").exists()
    restored = start(api, "restore", restore_operation_id=op["id"])
    assert restored["status"] == "completed", restored
    assert {p.name: p.read_bytes() for p in root.iterdir()} == original
    assert not api[2].operation(op["id"])["can_undo"]
    Image.open(root / "a.png").save(root / "a.jpg")
    op = start(api, "exclude", images=refs(row, "a.png"))
    assert op["status"] == "completed" and (root / "a.txt").exists() and (root / "a.mask.png").exists()


def test_center_crop_resizes_mask_and_undo_is_byte_exact(api, tmp_path):
    row, root = imported(api, tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    op = start(
        api,
        "preprocess",
        images=refs(row, "a.png"),
        preprocess={"mode": "center_crop", "width": 32, "height": 32},
    )
    assert op["status"] == "completed", op
    with Image.open(root / "a.png") as image:
        assert image.size == (32, 32)
    with Image.open(root / "a.mask.png") as mask:
        assert mask.size == (32, 32) and mask.getextrema() == (255, 255)
    assert (root / "a.txt").read_bytes() == before["a.txt"]
    assert start(api, "restore", restore_operation_id=op["id"])["status"] == "completed"
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_caption_operations_and_undo_refuses_external_changes(api, tmp_path):
    row, root = imported(api, tmp_path)
    op = start(
        api,
        "captions",
        images=refs(row, "a.png", "b.png"),
        captions={"mode": "fill_missing", "text": "trigger, {filename}"},
    )
    assert op["status"] == "completed", op
    assert (root / "a.txt").read_text() == "red, portrait"
    assert (root / "b.txt").read_text().strip() == "trigger, b"
    op = start(api, "captions", images=refs(row, "a.png"), captions={"mode": "append", "text": "red, new"})
    assert (root / "a.txt").read_text().strip() == "red, portrait, new"
    (root / "a.txt").write_text("manual correction")
    undo = start(api, "restore", restore_operation_id=op["id"])
    assert undo["status"] == "failed" and "edited" in undo["error"]
    assert (root / "a.txt").read_text() == "manual correction"


def test_failure_after_partial_publication_rolls_back_and_retry_is_idempotent(api, tmp_path, monkeypatch):
    row, root = imported(api, tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    import ypuddin.server.dataset_pipeline as module

    replace = module.os.replace
    calls = 0

    def flaky(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture disk full")
        return replace(source, target)

    monkeypatch.setattr(module.os, "replace", flaky)
    op = start(
        api, "captions", images=refs(row, "a.png", "b.png"), captions={"mode": "append", "text": "added"}
    )
    assert op["status"] == "failed" and op["result"]["rolled_back"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    monkeypatch.setattr(module.os, "replace", replace)
    retry = finish(api, api[0].post(f"/api/dataset-pipeline/operations/{op['id']}/retry"))
    assert retry["status"] == "completed"
    assert (root / "a.txt").read_text().count("added") == 1


@pytest.mark.parametrize("during_copy", [False, True])
def test_undo_rejects_damaged_backup_without_replacing_current_caption(
    api, tmp_path, monkeypatch, during_copy
):
    import ypuddin.server.dataset_pipeline as module

    row, root = imported(api, tmp_path)
    op = start(api, "captions", images=refs(row, "a.png"), captions={"mode": "append", "text": "new"})
    assert op["status"] == "completed", op
    current = (root / "a.txt").read_bytes()
    backup = Path(op["result"]["changes"][0]["backup"])
    if during_copy:
        original_copy = module.shutil.copy2

        def damaged_copy(source, destination, *args, **kwargs):
            result = original_copy(source, destination, *args, **kwargs)
            if Path(source) == backup:
                Path(destination).write_text("DAMAGED BACKUP")
            return result

        monkeypatch.setattr(module.shutil, "copy2", damaged_copy)
    else:
        backup.write_text("DAMAGED BACKUP")
    undo = start(api, "restore", restore_operation_id=op["id"])
    assert undo["status"] == "failed" and "backup" in undo["error"], undo
    assert (root / "a.txt").read_bytes() == current
    assert api[2].operation(op["id"])["can_undo"]


def test_cancel_after_commit_rolls_back_and_blocks_parallel_version_changes(api, tmp_path, monkeypatch):
    row, root = imported(api, tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    inspect = api[2]._inspect

    def paused(oid, pid, vid):
        entered.set()
        assert proceed.wait(5)
        return inspect(oid, pid, vid)

    monkeypatch.setattr(api[2], "_inspect", paused)
    client, _, _, p = api
    response = client.post(
        f"/api/projects/{p['id']}/versions/{p['active_version_id']}/pipeline/operations",
        json={
            "action": "captions",
            "images": refs(row, "a.png"),
            "captions": {"mode": "append", "text": "cancelled"},
        },
    )
    assert entered.wait(5)
    oid = response.json()["id"]
    # A short publication phase is not interruptible; cancellation is possible in post-scan.
    api[2]._progress(oid, "inspecting", 0, 2)
    assert client.post(f"/api/projects/{p['id']}/versions", json={"name": "copy"}).status_code == 409
    assert client.put(
        f"/api/datasets/{row['id']}/images/bogus/caption", json={"caption": "bad"}
    ).status_code in {404, 409}
    assert client.post(f"/api/dataset-pipeline/operations/{oid}/cancel").status_code == 200
    proceed.set()
    result = finish(api, response)
    assert result["status"] == "cancelled" and result["result"]["rolled_back"]
    assert (root / "a.txt").read_text() == "red, portrait"


def test_prepare_uses_real_plan_and_creates_version_bound_cache_job(api, tmp_path):
    row, root = imported(api, tmp_path)
    client, c, manager, p = api
    url = f"/api/projects/{p['id']}/config"
    cfg = client.get(url).json()
    cfg["model"].update(family="toy", dtype="fp32")
    cfg["dataset"].update(resolutions=[64], bucket_step=16)
    cfg["memory"].update(activation_checkpointing="none")
    assert client.put(url, json=cfg).status_code == 200
    op = start(api, "prepare")
    assert op["job_id"], op
    job = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (op["job_id"],))
    assert job["type"] == "cache" and job["version_id"] == p["active_version_id"]
    assert op["result"]["plan"]["ok"]
    c.supervisor._handle_event(
        job["id"], {"type": "cache.progress", "kind": "latents", "done": 1, "total": 2}
    )
    assert manager.operation(op["id"])["done"] == 1
    assert client.post(f"/api/dataset-pipeline/operations/{op['id']}/cancel").status_code == 200
    assert manager.operation(op["id"])["status"] == "cancelled"


def test_restart_recovers_journal_and_marks_retryable(api, tmp_path):
    row, root = imported(api, tmp_path)
    manager = api[2]
    op = start(api, "captions", images=refs(row, "a.png"), captions={"mode": "append", "text": "added"})
    api[1].db.update("dataset_pipeline_operations", op["id"], {"status": "running"})
    manager._recover()
    assert (root / "a.txt").read_text() == "red, portrait"
    assert manager.operation(op["id"])["status"] == "failed"


def test_cross_version_image_and_relative_path_are_rejected(api, tmp_path):
    row, root = imported(api, tmp_path)
    op = start(api, "exclude", images=refs(row, "../a.png"))
    assert op["status"] == "failed" and "relative" in op["error"]
    p = api[3]
    fork = (
        api[0]
        .post(
            f"/api/projects/{p['id']}/versions",
            json={"name": "empty", "data_mode": "empty", "source_version_id": None},
        )
        .json()
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        version = api[1].resolve_version(p["id"], fork["id"])
        if version["status"] == "ready":
            break
        time.sleep(0.01)
    response = api[0].post(
        f"/api/projects/{p['id']}/versions/{fork['id']}/pipeline/operations",
        json={"action": "exclude", "images": refs(row, "a.png")},
    )
    result = finish(api, response)
    assert result["status"] == "failed" and "another" in result["error"]
    assert (root / "a.png").exists()


def test_prepare_handoff_keeps_lock_until_cache_job_is_registered(api, tmp_path, monkeypatch):
    row, root = imported(api, tmp_path)
    client, c, manager, p = api
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    cfg["model"].update(family="toy", dtype="fp32")
    cfg["dataset"].update(resolutions=[64], bucket_step=16)
    client.put(f"/api/projects/{p['id']}/config", json=cfg)
    entered, attempted, finished = threading.Event(), threading.Event(), threading.Event()
    result = {}
    original = routes_work.create_job

    def competing_edit():
        assert entered.wait(5)
        attempted.set()
        try:
            with c.versions.mutation(p["id"], p["active_version_id"]):
                result["allowed"] = True
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=competing_edit)
    thread.start()

    def checked_create(*args, **kwargs):
        entered.set()
        assert attempted.wait(5)
        assert not finished.wait(0.05), "a mutation entered the lease-to-job gap"
        return original(*args, **kwargs)

    monkeypatch.setattr(routes_work, "create_job", checked_create)
    op = start(api, "prepare")
    assert op["job_id"], op
    thread.join(5)
    assert finished.is_set() and not result.get("allowed")
    assert "queued" in result["error"] or "job" in result["error"].lower()


def test_resize_does_not_upscale_or_strip_alpha_and_caption_uses_config_extension(api, tmp_path):
    row, root = imported(api, tmp_path, mask=False)
    Image.new("RGBA", (80, 40), (255, 0, 0, 80)).save(root / "a.png")
    client, _, _, p = api
    url = f"/api/projects/{p['id']}/config"
    cfg = client.get(url).json()
    cfg["dataset"]["sources"][0]["caption_ext"] = ".caption"
    assert client.put(url, json=cfg).status_code == 200
    op = start(
        api,
        "preprocess",
        images=refs(row, "a.png"),
        preprocess={"mode": "resize", "width": 1024, "height": 1024},
    )
    assert op["status"] == "completed", op
    with Image.open(root / "a.png") as image:
        assert image.size == (80, 40) and image.mode == "RGBA" and image.getpixel((0, 0))[3] == 80
    op = start(
        api, "captions", images=refs(row, "a.png"), captions={"mode": "replace", "text": "configured suffix"}
    )
    assert op["status"] == "completed"
    assert (root / "a.caption").read_text().strip() == "configured suffix"
    assert (root / "a.txt").read_text() == "red, portrait"


def test_duplicate_keeper_prefers_curated_copy_over_alphabetical_download(api, tmp_path):
    row, root = imported(api, tmp_path)
    (root / "00-download.png").write_bytes((root / "a.png").read_bytes())
    report = start(api, "inspect")["result"]["inspection"]
    group = report["duplicate_groups"][0]
    assert report["images"][group[0]]["rel_path"] == "a.png"
    assert report["images"][group[0]]["has_mask"] and report["images"][group[0]]["caption"]


def test_visual_crop_uses_exact_source_pixels_and_matching_mask_rectangle(api, tmp_path):
    row, root = imported(api, tmp_path)
    with Image.open(root / "a.png") as image:
        expected = image.crop((10, 5, 40, 25)).tobytes()
    with Image.open(root / "a.mask.png") as mask:
        expected_mask = mask.crop((10, 5, 40, 25)).tobytes()
    before = (root / "a.png").read_bytes()
    op = start(
        api,
        "preprocess",
        images=refs(row, "a.png"),
        preprocess={"mode": "crop_rect", "crop": {"x": 10, "y": 5, "width": 30, "height": 20}},
    )
    assert op["status"] == "completed", op
    with Image.open(root / "a.png") as image:
        assert image.size == (30, 20) and image.tobytes() == expected
    with Image.open(root / "a.mask.png") as mask:
        assert mask.size == (30, 20) and mask.tobytes() == expected_mask
    assert start(api, "restore", restore_operation_id=op["id"])["status"] == "completed"
    assert (root / "a.png").read_bytes() == before


def test_visual_crop_rejects_outside_negative_fractional_and_multiple_images(api, tmp_path):
    row, root = imported(api, tmp_path)
    before = (root / "a.png").read_bytes()
    op = start(
        api,
        "preprocess",
        images=refs(row, "a.png"),
        preprocess={"mode": "crop_rect", "crop": {"x": 70, "y": 0, "width": 30, "height": 20}},
    )
    assert op["status"] == "failed" and "exceeds" in op["error"]
    client, _, _, project = api
    url = f"/api/projects/{project['id']}/versions/{project['active_version_id']}/pipeline/operations"
    for x in (-1, 0.5):
        response = client.post(
            url,
            json={
                "action": "preprocess",
                "images": refs(row, "a.png"),
                "preprocess": {"mode": "crop_rect", "crop": {"x": x, "y": 0, "width": 20, "height": 20}},
            },
        )
        assert response.status_code == 422, response.text
    response = client.post(
        url,
        json={
            "action": "preprocess",
            "images": refs(row, "a.png", "b.png"),
            "preprocess": {"mode": "crop_rect", "crop": {"x": 0, "y": 0, "width": 20, "height": 20}},
        },
    )
    assert response.status_code == 422
    assert (root / "a.png").read_bytes() == before


def test_tagging_drafts_use_caption_transaction_missing_append_undo(api, tmp_path, monkeypatch):
    import sys
    import types

    row, root = imported(api, tmp_path)
    seen = []

    def generate(images, options, progress, cancel):
        seen.extend(image["rel_path"] for image in images)
        assert not cancel.is_set()
        progress(len(images), len(images), "Local model fixture inference completed")
        return ["red, generated" for _ in images]

    monkeypatch.setitem(
        sys.modules, "ypuddin.server.dataset_tagging", types.SimpleNamespace(generate=generate)
    )
    options = {
        "model_path": "fixture.onnx",
        "tags_path": "tags.csv",
        "mode": "missing",
        "trigger_word": "subject",
    }
    op = start(api, "tag", images=refs(row, "a.png", "b.png"), tagging=options)
    assert op["status"] == "completed", op
    assert seen == ["b.png"]
    assert (root / "a.txt").read_text() == "red, portrait"
    assert (root / "b.txt").read_text().strip() == "subject, red, generated"
    assert start(api, "restore", restore_operation_id=op["id"])["status"] == "completed"
    assert not (root / "b.txt").exists()
    op = start(api, "tag", images=refs(row, "a.png"), tagging={**options, "mode": "append"})
    assert op["status"] == "completed"
    assert (root / "a.txt").read_text().strip() == "subject, red, portrait, generated"


def test_tagging_cancel_stops_inference_before_publishing_captions(api, tmp_path, monkeypatch):
    import sys
    import types

    row, root = imported(api, tmp_path)
    entered = threading.Event()

    def generate(images, options, progress, cancel):
        entered.set()
        assert cancel.wait(5)
        raise RuntimeError("cancelled")

    monkeypatch.setitem(
        sys.modules, "ypuddin.server.dataset_tagging", types.SimpleNamespace(generate=generate)
    )
    client, _, _, project = api
    url = f"/api/projects/{project['id']}/versions/{project['active_version_id']}/pipeline/operations"
    response = client.post(
        url,
        json={
            "action": "tag",
            "images": refs(row, "b.png"),
            "tagging": {"model_path": "fixture.onnx", "tags_path": "tags.csv"},
        },
    )
    assert entered.wait(5)
    assert client.post(f"/api/dataset-pipeline/operations/{response.json()['id']}/cancel").status_code == 200
    op = finish(api, response)
    assert op["status"] == "cancelled", op
    assert not (root / "b.txt").exists()
    assert (root / "a.txt").read_text() == "red, portrait"


def test_config_caption_extension_syncs_version_registry_index_editor_and_pipeline(api, tmp_path):
    row, root = imported(api, tmp_path)
    client, c, _, project = api
    (root / "a.caption").write_text("new suffix label")
    cfg = client.get(f"/api/projects/{project['id']}/config").json()
    cfg["dataset"]["sources"][0]["path"] = str(root / ".")
    cfg["dataset"]["sources"][0]["caption_ext"] = ".caption"
    assert client.put(f"/api/projects/{project['id']}/config", json=cfg).status_code == 200
    assert c.db.fetchone("SELECT caption_ext,index_status FROM datasets WHERE id=?", (row["id"],)) == {
        "caption_ext": ".caption",
        "index_status": "ready",
    }
    images = client.get(f"/api/datasets/{row['id']}/images").json()["items"]
    image = next(image for image in images if image["rel_path"] == "a.png")
    assert image["caption"] == "new suffix label"
    response = client.put(
        f"/api/datasets/{row['id']}/images/{image['hash']}/caption", json={"caption": "editor correction"}
    )
    assert response.status_code == 200, response.text
    assert (root / "a.caption").read_text().strip() == "editor correction"
    assert (root / "a.txt").read_text() == "red, portrait"
    report = start(api, "inspect")["result"]["inspection"]
    assert (
        next(image for image in report["images"] if image["rel_path"] == "a.png")["caption"]
        == "editor correction"
    )
    # A source mentioned in another project's draft cannot rewrite this registry row.
    other = client.post("/api/projects", json={"name": "other"}).json()
    cfg["dataset"]["sources"][0]["caption_ext"] = ".other"
    assert client.put(f"/api/projects/{other['id']}/config", json=cfg).status_code == 200
    assert (
        c.db.fetchone("SELECT caption_ext FROM datasets WHERE id=?", (row["id"],))["caption_ext"]
        == ".caption"
    )


@pytest.mark.parametrize("extension", [".png", ".mask.png", "../caption.txt", ".JPG"])
def test_caption_extension_cannot_overwrite_images_or_escape_source(api, tmp_path, extension):
    row, root = imported(api, tmp_path)
    client, _, _, project = api
    cfg = client.get(f"/api/projects/{project['id']}/config").json()
    cfg["dataset"]["sources"][0]["caption_ext"] = extension
    response = client.put(f"/api/projects/{project['id']}/config", json=cfg)
    assert response.status_code == 400, response.text
    assert (root / "a.png").exists()


def test_tagging_never_overwrites_caption_edited_externally_during_inference(api, tmp_path, monkeypatch):
    import sys
    import types

    row, root = imported(api, tmp_path)

    def generate(images, options, progress, cancel):
        (root / "a.txt").write_text("external correction")
        return ["new prediction"]

    monkeypatch.setitem(
        sys.modules, "ypuddin.server.dataset_tagging", types.SimpleNamespace(generate=generate)
    )
    op = start(
        api,
        "tag",
        images=refs(row, "a.png"),
        tagging={"model_path": "fixture.onnx", "tags_path": "tags.csv", "mode": "overwrite"},
    )
    assert op["status"] == "failed" and "changed during tagging" in op["error"], op
    assert (root / "a.txt").read_text() == "external correction"


def test_tagging_admission_and_queued_registration_share_environment_lock(api, tmp_path, monkeypatch):
    row, _ = imported(api, tmp_path)
    client, c, manager, project = api
    endpoint = f"/api/projects/{project['id']}/versions/{project['active_version_id']}/pipeline/operations"
    request = {
        "action": "tag",
        "images": refs(row, "a.png"),
        "tagging": {"model_path": "fixture.onnx", "tags_path": "tags.csv"},
    }
    c.db.set_kv("environment.maintenance", {"blocked": True})
    response = client.post(endpoint, json=request)
    assert response.status_code == 409 and "environment.maintenance" in response.text
    assert not c.resolve_version(project["id"], project["active_version_id"])["busy"]
    assert not c.db.fetchall("SELECT id FROM dataset_pipeline_operations")
    c.db.set_kv("environment.maintenance", {"blocked": False})
    lease_observed = threading.Event()
    original_insert = c.db.insert
    original_submit = manager.executor.submit
    registrations = []
    pending = []

    def insert(table, row):
        if table == "dataset_pipeline_operations":
            # Simulate maintenance attempting to acquire the DB lock exactly
            # between the tagger's lease and publishing its queued operation.
            def maintenance():
                lease_observed.set()
                with c.db.lock:
                    registrations.extend(c.db.fetchall("SELECT action FROM dataset_pipeline_operations"))

            thread = threading.Thread(target=maintenance)
            thread.start()
            assert lease_observed.wait(2)
            assert registrations == []
            insert.thread = thread
        return original_insert(table, row)

    monkeypatch.setattr(c.db, "insert", insert)
    monkeypatch.setattr(manager.executor, "submit", lambda *args: pending.append(args))
    response = client.post(endpoint, json=request)
    assert response.status_code == 202, response.text
    insert.thread.join(2)
    assert registrations == [{"action": "tag"}]
    # Run the intentionally deferred operation to release its acquired lease.
    monkeypatch.setattr(manager.executor, "submit", original_submit)
    monkeypatch.setattr(c.db, "insert", original_insert)
    oid = response.json()["id"]
    manager.cancel(oid)
    original_submit(*pending[0])
    assert finish(api, response)["status"] == "cancelled"
