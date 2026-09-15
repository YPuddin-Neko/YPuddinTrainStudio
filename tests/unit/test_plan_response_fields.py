"""API response contract for per-device sharding estimates; no GPU execution."""

from fastapi.testclient import TestClient

from ypuddin.server import create_app
from ypuddin.server.models import Plan


def test_plan_http_preserves_sharding_estimates_and_audit_fields(tmp_path, monkeypatch):
    payload = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "memory": {
            "estimate_scope": "per_device",
            "adapter_mb": 24000.0,
            "gradients_mb": 24000.0,
            "optimizer_mb": 150.0,
            "communication_mb_estimate": 900.0,
            "optimizer_workspace_mb_estimate": 200.0,
            "initialization_peak_mb_estimate": None,
            "peak_mb_estimate": 50000.0,
            "estimate_notes": ["每卡估算，实际占用取决于运行时通信。"],
            "sharding": {"rank_parameter_bytes": [25165824000, 25165824000], "groups": ["blocks.0"]},
        },
        "distributed": {
            "strategy": "fsdp",
            "parameter_storage": "sharded",
            "gradient_storage": "sharded",
            "optimizer_storage": "sharded",
            "world_size": 2,
            "per_device_batch_size": 1,
            "effective_batch_size": 2,
            "batches_per_rank": 4,
            "dropped_samples": 0,
            "tail_policy": "drop_incomplete_rank_group",
        },
    }
    monkeypatch.setattr("ypuddin.server.routes_core.make_plan", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr("ypuddin.server.routes_core.gpu_info", lambda: [])
    monkeypatch.setattr("ypuddin.server.supervisor.training_device_error", lambda *_args: None)
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    try:
        with TestClient(app) as client:
            response = client.post("/api/plan", json={"config": {"model": {"family": "toy"}}})
            assert response.status_code == 200, response.text
            assert response.json() == payload
        # These are explicit documented fields, not only extra/unknown values.
        schema = Plan.model_json_schema()
        assert "communication_mb_estimate" in schema["$defs"]["PlanMemory"]["properties"]
        assert schema["$defs"]["PlanDistributed"]["properties"]["strategy"]["enum"] == [
            "single",
            "ddp",
            "fsdp",
        ]
    finally:
        app.state.ctx.db.close()
