"""Sample loss is exact training-step metadata, including old logs and SSE replay."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ypuddin.server import create_app
from ypuddin.server.sample_events import sample_event_loss, samples_with_loss


def sample(step, name, **fields):
    return {
        "type": "sample.saved",
        "step": step,
        "prompt_index": 0,
        "prompt": "a cat",
        "seed": 1,
        "path": f"/samples/{name}.png",
        "width": 64,
        "height": 64,
        "ts": 100.0 + step,
        **fields,
    }


def legacy_events():
    return [
        {"type": "run.started"},
        {"type": "step", "step": 0, "loss": 99.0},
        sample(0, "initial"),
        {"type": "step", "step": 1, "loss": 0.75},
        sample(1, "exact"),
        sample(2, "sparse-old"),
        sample(2, "sparse-new", loss=0.625),
        {"type": "step", "step": 3, "loss": 0.5},
        sample(3, "explicit-null", loss=None),
        {"type": "run.resumed", "step": 3},
        sample(3, "resumed-without-record"),
        sample(3, "resumed-checkpoint-record", loss=0.45),
        {"type": "step", "step": 3, "loss": 0.4},
        sample(3, "resumed-exact"),
        sample(4, "before-future-step"),
        {"type": "step", "step": 4, "loss": 0.3},
        sample(4, "after-step"),
        {"type": "run.started"},
        sample(4, "new-attempt"),
    ]


def test_legacy_loss_uses_exact_step_and_attempt_without_overwriting_explicit_null():
    assert [loss for _, loss in samples_with_loss(legacy_events())] == [
        None,
        0.75,
        None,
        0.625,
        None,
        None,
        0.45,
        0.4,
        None,
        0.3,
        None,
    ]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, "0.5", {}, 10**400])
def test_nonfinite_or_non_numeric_losses_do_not_escape_in_sample_json(invalid):
    events = [
        {"type": "step", "step": 1, "loss": invalid},
        sample(1, "legacy"),
        sample(1, "new", loss=invalid),
        sample(0, "initial", loss=0.5),
    ]
    assert [loss for _, loss in samples_with_loss(events)] == [None, None, None]


def test_sample_rest_and_sse_share_exact_loss_contract_and_job_isolation(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "absent")
    client = TestClient(app)  # No lifespan: this fixture never starts training or queue scheduling.
    ctx = app.state.ctx
    try:
        events = legacy_events()
        for jid, history in (("job_first", events), ("job_other", [sample(1, "other-job")])):
            run = tmp_path / jid
            run.mkdir()
            ctx.db.insert(
                "jobs",
                {
                    "id": jid,
                    "type": "train",
                    "name": jid,
                    "status": "completed",
                    "created_at": 1,
                    "run_dir": str(run),
                },
            )
            (run / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in history) + '\n[]\n{"partial":',
                encoding="utf-8",
            )
        response = client.get("/api/jobs/job_first/samples")
        assert response.status_code == 200
        rows = response.json()
        assert [row["loss"] for row in rows] == [loss for _, loss in samples_with_loss(events)]
        assert client.get("/api/jobs/job_other/samples").json()[0]["loss"] is None
        for event in events:
            if event["type"] == "sample.saved":
                ctx.supervisor._handle_event("job_first", event)
        pushed = [event["data"] for event in ctx.bus.replay(0) if event["type"] == "job.sample"]
        assert len(pushed) == len(rows)
        for live, rest in zip(pushed, rows, strict=True):
            assert {key: live[key] for key in rest} == rest
            assert live["job_id"] == "job_first"
        assert sample_event_loss(sample(1, "not-in-history"), tmp_path / "job_first/events.jsonl") is None
        assert sample_event_loss(sample(1, "new-no-file", loss=0.0), Path("absent")) == 0.0
    finally:
        client.close()
        for name in ("regularization", "model_downloads", "environment", "dataset_pipeline"):
            getattr(app.state, name).close()
        ctx.versions.close()
        ctx.db.close()
