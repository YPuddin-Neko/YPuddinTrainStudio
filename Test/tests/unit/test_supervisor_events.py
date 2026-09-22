"""Bursting worker logs must not starve HTTP or hide their final outcome."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from ypuddin.server.bus import EventBus
from ypuddin.server.db import Database
from ypuddin.server.routes_work import router
from ypuddin.server.supervisor import EVENT_BATCH_SIZE, JobSupervisor


@pytest.fixture
def service(tmp_path):
    db = Database(tmp_path / "studio.db")
    bus = EventBus(history=10000)
    supervisor = JobSupervisor(db, bus, tmp_path, poll_interval=0.01)
    db.set_kv("queue.settings", {"held": True})
    app = FastAPI()
    app.state.ctx = SimpleNamespace(db=db, bus=bus, supervisor=supervisor)
    app.include_router(router, prefix="/api")
    yield SimpleNamespace(db=db, bus=bus, supervisor=supervisor, app=app, root=tmp_path)
    db.close()


def add_job(service, name, events):
    run = service.root / name
    run.mkdir()
    path = run / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    service.db.insert(
        "jobs",
        {
            "id": name,
            "name": name,
            "type": "train",
            "status": "running",
            "created_at": 1,
            "run_dir": str(run),
            "config_json": "{}",
        },
    )
    return path


def step(number):
    return {"type": "step", "step": number, "epoch": 0, "loss": 1 / number}


@pytest.mark.parametrize(
    "kind,status",
    [
        ("run.finished", "completed"),
        ("run.paused", "paused"),
        ("run.stopped", "cancelled"),
        ("run.failed", "failed"),
    ],
)
def test_exited_worker_drains_bounded_batches_before_finalizing(service, kind, status):
    s = service.supervisor
    events = [step(n) for n in range(1, EVENT_BATCH_SIZE * 2 + 2)] + [{"type": kind}]
    path = add_job(service, "exited", events)
    s._procs["exited"] = SimpleNamespace(poll=lambda: 0, returncode=0)
    s._devices["exited"] = "cuda:0"
    s._tick()
    row = service.db.fetchone("SELECT * FROM jobs WHERE id='exited'")
    assert json.loads(row["progress_json"])["step"] == EVENT_BATCH_SIZE
    assert row["status"] == "running" and row["exit_code"] is None
    assert "exited" in s._procs and "exited" in s._devices
    for _ in range(4):
        s._tick()
    row = service.db.fetchone("SELECT * FROM jobs WHERE id='exited'")
    assert row["status"] == status and row["exit_code"] == 0
    assert "exited" not in s._procs and "exited" not in s._devices
    assert s._offsets["exited"] == path.stat().st_size
    assert [e["data"]["step"] for e in service.bus.replay(0) if e["type"] == "job.step"] == list(
        range(1, EVENT_BATCH_SIZE * 2 + 2)
    )


def test_partial_and_malformed_lines_do_not_lose_the_following_event(service):
    path = add_job(service, "partial", [step(1)])
    with path.open("ab") as stream:
        stream.write(b'bad json\n42\n\xff\n{"type":"step","step":2')
    assert service.supervisor._pump_events("partial")
    previous = service.supervisor._offsets["partial"]
    assert previous < path.stat().st_size
    with path.open("ab") as stream:
        stream.write(b',"epoch":0}\n')
    assert service.supervisor._pump_events("partial")
    assert [e["data"]["step"] for e in service.bus.replay(0)] == [1, 2]


def test_failed_batch_rolls_back_cursor_outcome_and_publications(service, monkeypatch):
    s = service.supervisor
    path = add_job(service, "rollback", [step(1), {"type": "run.finished"}, {"type": "warning"}])
    handle = s._handle_event

    def fail(job_id, event):
        if event["type"] == "warning":
            raise RuntimeError("fixture write failure")
        handle(job_id, event)

    monkeypatch.setattr(s, "_handle_event", fail)
    with pytest.raises(RuntimeError, match="fixture write failure"):
        s._pump_events("rollback")
    assert s._offsets.get("rollback", 0) == 0
    assert "rollback" not in s._outcome_seen and not service.bus.replay(0)
    row = service.db.fetchone("SELECT * FROM jobs WHERE id='rollback'")
    assert row["status"] == "running" and json.loads(row["progress_json"]) == {}
    monkeypatch.setattr(s, "_handle_event", handle)
    assert s._pump_events("rollback")
    assert s._offsets["rollback"] == path.stat().st_size
    assert [e["type"] for e in service.bus.replay(0)] == [
        "job.step",
        "job.state",
        "queue.changed",
        "job.warning",
    ]


def test_batch_publications_are_visible_only_after_commit_and_thread_local(service):
    s = service.supervisor
    add_job(service, "commit", [step(1), step(2)])
    published = []
    original = service.bus.publish

    def publish(kind, data):
        assert not service.db.conn.in_transaction
        published.append(kind)
        return original(kind, data)

    service.bus.publish = publish
    assert s._pump_events("commit")
    assert published == ["job.step", "job.step"]
    pending = []
    token = s._pending_events.set(pending)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(s._publish, "external.control", {}).result(timeout=2)
        assert not pending and published[-1] == "external.control"
    finally:
        s._pending_events.reset(token)


def test_nested_database_transactions_preserve_inner_and_outer_rollback(service):
    db = service.db
    with db.transaction():
        db.set_kv("outer", 1)
        with pytest.raises(ValueError), db.transaction():
            db.set_kv("inner", 1)
            raise ValueError("rollback inner")
        assert db.get_kv("outer") == 1 and db.get_kv("inner") is None
    with pytest.raises(ValueError), db.transaction():
        with db.transaction():
            db.set_kv("outer", 2)
        raise ValueError("rollback outer")
    assert db.get_kv("outer") == 1 and not db.conn.in_transaction


@pytest.mark.asyncio
async def test_http_job_lookup_remains_responsive_during_slow_commit_burst(service, record_property):
    """Real SQLite/REST with 5 ms durable-write latency; no training/HTTP timeout change."""
    s = service.supervisor
    total = EVENT_BATCH_SIZE * 4
    add_job(service, "burst", [step(n) for n in range(1, total + 1)])
    add_job(service, "other", [{"type": "xyz.progress", "done": 1, "total": 64}])
    for name in ("burst", "other"):
        s._procs[name] = SimpleNamespace(poll=lambda: None)
    commits = []

    def slow_durable_write(sql):
        # In autocommit mode each UPDATE must flush independently; a batch
        # SAVEPOINT only flushes on its final RELEASE. SQLite still executes all SQL.
        if (sql.startswith("UPDATE") and not service.db.conn.in_transaction) or sql.startswith("RELEASE"):
            commits.append(sql)
            time.sleep(0.005)

    service.db.conn.set_trace_callback(slow_durable_write)
    task = asyncio.create_task(s._loop())
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.app), base_url="http://test"
        ) as client:
            await asyncio.sleep(0)
            response = await client.get("/api/jobs/other")
        elapsed = time.monotonic() - started
        record_property("http_elapsed_seconds", elapsed)
        record_property("durable_writes_before_response", len(commits))
        assert response.status_code == 200 and response.json()["progress"]["done"] == 1
        assert elapsed < 0.5, f"HTTP was starved for {elapsed:.3f}s by progress commits"
        assert len(commits) < 20, "One durable write was issued for each progress event"
    finally:
        s._stopping = True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        service.db.conn.set_trace_callback(None)
        s._procs.clear()


@pytest.mark.asyncio
async def test_backlog_yields_without_waiting_for_normal_poll_interval(service):
    s = service.supervisor
    s.poll = 60
    path = add_job(service, "backlog", [step(n) for n in range(1, EVENT_BATCH_SIZE * 4 + 1)])
    s._procs["backlog"] = SimpleNamespace(poll=lambda: None)
    task = asyncio.create_task(s._loop())

    async def wait_for_drain():
        while s._offsets.get("backlog", 0) < path.stat().st_size:
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(wait_for_drain(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        s._procs.clear()


@pytest.mark.asyncio
async def test_stop_drains_all_events_before_closing_database(service):
    s = service.supervisor
    path = add_job(
        service,
        "shutdown",
        [step(n) for n in range(1, EVENT_BATCH_SIZE * 2 + 1)]
        + [
            {"type": "run.paused", "preparing": True},
        ],
    )
    s._procs["shutdown"] = SimpleNamespace(poll=lambda: 0, returncode=0)
    s._devices["shutdown"] = "cuda:0"
    await s.stop()
    row = service.db.fetchone("SELECT * FROM jobs WHERE id='shutdown'")
    assert row["status"] == "paused" and row["exit_code"] == 0
    assert s._offsets["shutdown"] == path.stat().st_size and not s._procs and not s._devices
