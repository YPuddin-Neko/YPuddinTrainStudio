"""Real import stage counters, concurrent polling, retention and failure isolation."""

import asyncio
import io
import threading
import uuid
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app, dataset_uploads, routes_work
from ypuddin.server.errors import ApiError
from ypuddin.server.import_progress import ImportProgress, ImportProgressStore


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def test_stage_counters_speed_and_eta_are_actual_and_reset():
    clock = Clock()
    store = ImportProgressStore(clock=clock)
    progress = store.start("p", "a" * 16)
    progress.set_phase("receiving", bytes_total=100, files_total=2)
    assert progress.snapshot()["bytes_per_second"] is None
    clock.value = 2
    progress.advance(bytes_done=40, files_done=1)
    first = progress.snapshot()
    assert (first["bytes_done"], first["files_done"], first["bytes_per_second"], first["eta_seconds"]) == (
        40,
        1,
        20,
        3,
    )
    clock.value = 4
    progress.advance(bytes_done=60, files_done=1)
    assert progress.snapshot()["bytes_per_second"] == 25
    progress.set_phase("validating", files_total=2)
    progress.advance(files_done=1)
    clock.value = 5
    value = progress.snapshot()
    assert value["bytes_done"] == 0 and value["bytes_total"] is None
    assert value["files_done"] == 1 and value["files_total"] == 2
    assert value["elapsed_seconds"] == 5 and value["phase_elapsed_seconds"] == 1
    assert value["bytes_per_second"] is None and value["eta_seconds"] is None
    progress.set_phase("copying", bytes_total=80)
    progress.advance(bytes_done=16)
    clock.value = 7
    assert progress.snapshot()["bytes_per_second"] == 8
    progress.complete()
    done = progress.snapshot()
    clock.value = 100
    progress.advance(bytes_done=999)
    progress.fail("late.error")
    assert progress.snapshot() == done
    assert done["phase"] == "completed" and done["elapsed_seconds"] == 7


def test_tracker_project_binding_reuse_ttl_capacity_and_safe_errors():
    clock = Clock()
    store = ImportProgressStore(capacity=1, ttl_seconds=10, clock=clock)
    progress = store.start("p", "a" * 16)
    with pytest.raises(ApiError) as other:
        store.get("other-project", "a" * 16)
    assert other.value.status == 404
    with pytest.raises(ApiError) as duplicate:
        store.start("p", "a" * 16)
    assert duplicate.value.status == 409
    clock.value = 11
    with pytest.raises(ApiError) as capacity:
        store.start("p", "b" * 16)
    assert capacity.value.status == 503  # Active imports cannot expire mid-request.
    progress.fail("/private/secret/model?token=credential")
    assert store.get("p", "a" * 16)["error"] == "import.failed"
    clock.value = 20
    assert store.get("p", "a" * 16)["phase"] == "failed"
    clock.value = 21
    with pytest.raises(ApiError) as expired:
        store.get("p", "a" * 16)
    assert expired.value.status == 404
    assert store.start("p", "b" * 16)


@pytest.mark.parametrize("id", ["short", "../" + "a" * 16, "a" * 129, "space " + "a" * 16])
def test_invalid_ids_rejected(id):
    with pytest.raises(ApiError) as error:
        ImportProgressStore().start("p", id)
    assert error.value.code == "progress.invalid"


def test_track_keeps_original_exception_and_sanitizes_failure():
    store = ImportProgressStore()
    error = ApiError("private absolute path and credentials", code="upload.conflict", status=409)
    with pytest.raises(ApiError) as raised:
        with store.track("p", "a" * 16, "validating"):
            raise error
    assert raised.value is error
    assert store.get("p", "a" * 16)["error"] == "upload.conflict"
    with store.track("p", None, "validating") as progress:
        assert progress is None


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    pid = client.post("/api/projects", json={"name": "Progress", "family": "toy"}).json()["id"]
    yield app, client, pid
    app.state.regularization.close()
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def png():
    output = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def stages(monkeypatch):
    events = []
    original = ImportProgress.advance

    def capture(self, **kwargs):
        original(self, **kwargs)
        events.append(self.snapshot())

    monkeypatch.setattr(ImportProgress, "advance", capture)
    monkeypatch.setattr(dataset_uploads, "CHUNK", 16)
    return events


def test_upload_zip_reports_real_expanded_bytes_and_validated_files(api, stages):
    app, client, pid = api
    image, caption = png(), b"red"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("concept/a.png", image)
        output.writestr("concept/a.txt", caption)
    id = uuid.uuid4().hex
    response = client.post(
        f"/api/projects/{pid}/datasets/upload?progress_id={id}",
        files={"files": ("training.zip", archive.getvalue(), "application/zip")},
    )
    assert response.status_code == 200, response.text
    receiving = [event for event in stages if event["phase"] == "receiving"]
    assert receiving[-1]["bytes_done"] == int(response.request.headers["content-length"])
    assert receiving[-1]["files_done"] == 1
    extracting = [event for event in stages if event["phase"] == "extracting"]
    assert len(extracting) > 3  # Multiple actual writes, not one simulated completion update.
    assert extracting[-1]["bytes_done"] == extracting[-1]["bytes_total"] == len(image) + len(caption)
    assert extracting[-1]["files_done"] == extracting[-1]["files_total"] == 2
    assert [event["bytes_done"] for event in extracting] == sorted(
        event["bytes_done"] for event in extracting
    )
    assert any(event["phase"] == "validating" and event["files_done"] == 2 for event in stages)
    copying = [event for event in stages if event["phase"] == "copying"]
    assert copying[-1]["bytes_done"] == copying[-1]["bytes_total"] == len(image) + len(caption)
    assert client.get(f"/api/projects/{pid}/datasets/import-progress/{id}").json()["phase"] == "completed"
    assert app.state.ctx.import_progress.get(pid, id)["error"] is None


def test_local_directory_reports_actual_copy_bytes_and_keeps_idempotent_sync(api, tmp_path, stages):
    _, client, pid = api
    source = tmp_path / "concept"
    source.mkdir()
    image = png()
    (source / "a.png").write_bytes(image)
    id = uuid.uuid4().hex
    response = client.post(f"/api/projects/{pid}/datasets?progress_id={id}", json={"path": str(source)})
    assert response.status_code == 201, response.text
    copying = [event for event in stages if event["phase"] == "copying"]
    assert copying[0]["bytes_done"] == len(image)  # The snapshot has read and written the real file.
    assert copying[0]["bytes_total"] == len(image)
    assert copying[-1]["bytes_done"] == len(image) and copying[-1]["files_done"] == 1
    stages.clear()
    again = client.post(
        f"/api/projects/{pid}/datasets?progress_id={uuid.uuid4().hex}", json={"path": str(source)}
    )
    assert again.status_code == 201 and again.json()["source"]["id"] == response.json()["source"]["id"]
    # One snapshot copy only; the already-identical destination is not counted as newly copied.
    assert sum(event["files_done"] for event in stages if event["phase"] == "copying") == 1


def test_upload_failures_progress_reuse_and_unknown_ids(api):
    _, client, pid = api
    id = uuid.uuid4().hex
    url = f"/api/projects/{pid}/datasets/upload?progress_id={id}"
    response = client.post(url, files={"files": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 400
    progress = client.get(f"/api/projects/{pid}/datasets/import-progress/{id}").json()
    assert progress["phase"] == "failed" and progress["error"] == "upload.image"
    assert client.post(url, files={"files": ("a.png", png(), "image/png")}).status_code == 409
    assert client.get(f"/api/projects/{pid}/datasets/import-progress/{id}").json() == progress
    assert client.get(f"/api/projects/other/datasets/import-progress/{id}").status_code == 404
    assert client.get(f"/api/projects/{pid}/datasets/import-progress/{uuid.uuid4().hex}").status_code == 404


def test_registration_failure_keeps_progress_failed_and_rolls_back_only_new_files(api, monkeypatch):
    app, client, pid = api
    source = app.state.ctx.dataset_dir(pid) / "concept"
    source.mkdir(parents=True)
    (source / "existing.png").write_bytes(png())
    id = uuid.uuid4().hex

    def fail(*args, **kwargs):
        raise ApiError("private directory must not appear in progress", code="dataset.registration_failed")

    monkeypatch.setattr(routes_work, "_write_project_config", fail)
    response = client.post(
        f"/api/projects/{pid}/datasets/upload?progress_id={id}",
        files={"files": ("concept/new.png", png(), "image/png")},
    )
    assert response.status_code == 400
    assert (source / "existing.png").read_bytes() == png()
    assert not (source / "new.png").exists()
    assert app.state.ctx.db.fetchall("SELECT * FROM datasets") == []
    result = client.get(f"/api/projects/{pid}/datasets/import-progress/{id}").json()
    assert result["phase"] == "failed" and result["error"] == "dataset.registration_failed"


def test_local_missing_directory_failure_is_tracked_without_exposing_path(api, tmp_path):
    _, client, pid = api
    id = uuid.uuid4().hex
    missing = tmp_path / "private-source"
    response = client.post(f"/api/projects/{pid}/datasets?progress_id={id}", json={"path": str(missing)})
    assert response.status_code == 404
    result = client.get(f"/api/projects/{pid}/datasets/import-progress/{id}")
    assert result.json()["error"] == "fs.not_found"
    assert result.json()["phase"] == "failed"
    assert str(missing) not in result.text


def test_progress_read_does_not_wait_for_database_lock(api):
    app, client, pid = api
    id = uuid.uuid4().hex
    app.state.ctx.import_progress.start(pid, id)
    responses = []
    with app.state.ctx.db.lock:
        reader = threading.Thread(
            target=lambda: responses.append(client.get(f"/api/projects/{pid}/datasets/import-progress/{id}"))
        )
        reader.start()
        reader.join(timeout=2)
        finished_without_db = not reader.is_alive()
    reader.join(timeout=2)
    assert finished_without_db, "progress GET blocked on the import's database lock"
    assert responses[0].status_code == 200


def test_real_streaming_upload_can_be_polled_before_remaining_bytes_arrive(api):
    app, _, pid = api
    id = uuid.uuid4().hex
    image = png()
    header = b'--progress-boundary\r\nContent-Disposition: form-data; name="files"; filename="concept/a.png"\r\nContent-Type: image/png\r\n\r\n'
    prefix = header + image[:32]
    suffix = image[32:] + b"\r\n--progress-boundary--\r\n"

    async def run():
        paused, release = asyncio.Event(), asyncio.Event()

        async def body():
            yield prefix
            paused.set()
            await release.wait()
            yield suffix

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            uploading = asyncio.create_task(
                client.post(
                    f"/api/projects/{pid}/datasets/upload?progress_id={id}",
                    content=body(),
                    headers={
                        "Content-Type": "multipart/form-data; boundary=progress-boundary",
                        "Content-Length": str(len(prefix) + len(suffix)),
                    },
                )
            )
            try:
                await asyncio.wait_for(paused.wait(), timeout=3)
                polled = await asyncio.wait_for(
                    client.get(f"/api/projects/{pid}/datasets/import-progress/{id}"), timeout=2
                )
                value = polled.json()
                assert value["phase"] == "receiving"
                assert value["bytes_done"] == len(prefix)
                assert value["bytes_total"] == len(prefix) + len(suffix)
            finally:
                release.set()
            response = await asyncio.wait_for(uploading, timeout=5)
            assert response.status_code == 200, response.text

    asyncio.run(run())
