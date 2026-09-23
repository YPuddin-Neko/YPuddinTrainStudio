"""Queue groups, contextual search and byte-accurate bounded log navigation."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ypuddin.server import create_app


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "none")
    client = TestClient(app)
    project = client.post("/api/projects", json={"id": "QueueStudy", "name": "衣装实验"}).json()
    c = app.state.ctx
    for index, (status, kind, priority) in enumerate(
        [
            ("running", "train", 0),
            ("paused", "train", 0),
            ("queued", "train", 10),
            ("queued", "train", 10),
            ("scheduled", "cache", 0),
            ("completed", "train", 0),
            ("failed", "train", 99),
            ("completed", "cache", 0),
        ]
    ):
        c.db.insert(
            "jobs",
            {
                "id": f"j_{index}",
                "project_id": project["id"],
                "version_id": project["active_version_id"],
                "type": kind,
                "name": f"Run {index}",
                "status": status,
                "priority": priority,
                "created_at": 1000 + index,
                "run_dir": str(tmp_path / f"j_{index}"),
                "config_json": "{}",
            },
        )
    yield client, c, project
    for manager in (
        app.state.regularization,
        app.state.dataset_pipeline,
        app.state.environment,
        app.state.model_downloads,
        c.versions,
    ):
        manager.close()
    client.close()
    c.db.close()


def test_group_type_status_and_context_search_are_and_filtered_before_pagination(api):
    client, _, project = api
    params = {"group": "history", "type": "train", "q": "衣装", "page_size": 1}
    first = client.get("/api/jobs", params=params).json()
    assert first["total"] == 2 and [j["id"] for j in first["items"]] == ["j_6"]
    second = client.get("/api/jobs", params=params | {"page": 2}).json()
    assert [j["id"] for j in second["items"]] == ["j_5"]
    row = second["items"][0]
    assert (row["project_name"], row["version_number"], row["version_id"]) == (
        "衣装实验",
        1,
        project["active_version_id"],
    )
    assert client.get("/api/jobs", params=params | {"status": "completed"}).json()["total"] == 1
    assert client.get("/api/jobs", params=params | {"status": "running"}).json()["total"] == 0
    assert client.get("/api/jobs", params=params | {"q": "%"}).json()["total"] == 0
    assert client.get("/api/jobs", params=params | {"version_id": "foreign"}).json()["total"] == 0
    waiting = client.get("/api/jobs", params={"group": "waiting"}).json()
    assert [j["id"] for j in waiting["items"]] == ["j_2", "j_3", "j_4"]
    assert client.get("/api/jobs", params={"status": "running,paused"}).json()["total"] == 2


def test_log_cursor_handles_crlf_unicode_tail_and_final_line(api):
    client, c, _ = api
    root = Path(c.db.fetchone("SELECT run_dir FROM jobs WHERE id='j_0'")["run_dir"])
    root.mkdir()
    lines = ["第一行", "warning 第二行", "第三行", "last without newline"]
    raw = "\r\n".join(lines).encode()
    (root / "run.log").write_bytes(raw)
    first = client.get("/api/jobs/j_0/log", params={"limit": 2}).json()
    assert [line["msg"] for line in first["lines"]] == lines[:2]
    assert first["has_more"] and first["lines"][1]["level"] == "warn"
    second = client.get("/api/jobs/j_0/log", params={"offset": first["next_offset"], "limit": 2}).json()
    assert [line["msg"] for line in second["lines"]] == lines[2:]
    assert not second["has_more"] and second["next_offset"] == len(raw)
    tail = client.get("/api/jobs/j_0/log", params={"tail": True, "limit": 1}).json()
    assert [line["msg"] for line in tail["lines"]] == lines[-1:]
    assert tail["next_offset"] == len(raw)
    assert (
        client.get("/api/jobs/j_0/log", params={"offset": -5, "limit": 1}).json()["lines"][0]["msg"]
        == lines[0]
    )


def test_log_reads_python_timestamp_level_and_multiline_traceback(api):
    from datetime import datetime

    client, c, _ = api
    root = Path(c.db.fetchone("SELECT run_dir FROM jobs WHERE id='j_0'")["run_dir"])
    root.mkdir()
    (root / "run.log").write_text(
        "2026-09-23 19:27:02,198 INFO ypuddin.models.anima.family: loaded Anima DiT\n"
        "2026-09-23 19:27:03,005 WARNING ypuddin.train: low memory\n"
        "Traceback (most recent call last):\n"
        '  File "loader.py", line 3\n'
        "RuntimeError: operator torchvision::nms does not exist\n"
        "2026-09-23 19:27:04,001 DEBUG ypuddin.train: cleanup\n"
    )
    lines = client.get("/api/jobs/j_0/log").json()["lines"]
    assert lines[0] == {
        "ts": datetime(2026, 9, 23, 19, 27, 2, 198000).timestamp(),
        "level": "info",
        "msg": "ypuddin.models.anima.family: loaded Anima DiT",
    }
    assert [line["level"] for line in lines] == ["info", "warn", "error", "error", "error", "debug"]
    assert lines[2]["ts"] is None
