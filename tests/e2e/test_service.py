"""Service end-to-end: REST contract + supervisor running a real toy training subprocess."""

import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from ypuddin.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(tmp_path):
    """Real uvicorn server in a thread (ASGITransport cannot stream SSE)."""
    app = create_app(tmp_path / "studio_data", frontend_dist=tmp_path / "nonexistent", poll_interval=0.2)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=15)


async def _wait_status(client: httpx.AsyncClient, jid: str, wanted: set[str], timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = await client.get(f"/api/jobs/{jid}")
        job = r.json()
        if job["status"] in wanted:
            return job
        await asyncio.sleep(0.3)
    raise AssertionError(f"job {jid} did not reach {wanted}; last={job}")


@pytest.mark.asyncio
async def test_rest_contract_and_training_job(live_server, image_dataset):
    if True:
        async with httpx.AsyncClient(base_url=live_server, timeout=30) as client:
            # health / schema / presets / settings
            h = (await client.get("/api/health")).json()
            assert h["api_version"] == 1 and "toy" in h["families"]
            schema = (await client.get("/api/schema/train")).json()
            assert "x-ui-groups" in schema
            presets = (await client.get("/api/presets")).json()
            assert any(p["name"] == "toy-smoke" for p in presets)
            r = await client.get("/api/presets/nope")
            assert (
                r.status_code == 404
                and r.json()["error"]["code"] == "preset.not_found"
                and "trace_id" in r.json()["error"]
            )
            s = (await client.get("/api/settings")).json()
            assert s["ui"]["language"] == "zh-CN"
            r = await client.put("/api/settings", json={"ui": {"language": "en"}})
            assert r.json()["ui"]["language"] == "en"

            # project + dataset
            p = (await client.post("/api/projects", json={"name": "demo"})).json()
            pid = p["id"]
            d = (await client.post(f"/api/projects/{pid}/datasets", json={"path": str(image_dataset)})).json()
            did = d["source"]["id"]
            for _ in range(100):
                info = (await client.get(f"/api/datasets/{did}")).json()
                if info["index_status"] == "ready":
                    break
                await asyncio.sleep(0.1)
            assert info["index_status"] == "ready" and info["stats"]["images"] == 12
            imgs = (await client.get(f"/api/datasets/{did}/images?page_size=5")).json()
            assert imgs["total"] == 12 and len(imgs["items"]) == 5
            h0 = imgs["items"][0]["hash"]
            th = await client.get(f"/api/datasets/{did}/images/{h0}/thumb?size=64")
            assert th.status_code == 200 and th.headers["content-type"] == "image/jpeg"
            r = await client.put(
                f"/api/datasets/{did}/images/{h0}/caption", json={"caption": "1girl, edited"}
            )
            assert (await client.get(f"/api/datasets/{did}/images/{h0}/caption")).json()[
                "caption"
            ] == "1girl, edited"
            r = await client.post(
                f"/api/datasets/{did}/tags/batch",
                json={"hashes": [h0], "add": ["newtag"], "remove": ["edited"]},
            )
            assert r.json()["changed"] == 1
            assert (
                "newtag" in (await client.get(f"/api/datasets/{did}/images/{h0}/caption")).json()["caption"]
            )

            # plan + validate
            config = {
                "model": {"family": "toy", "dtype": "fp32"},
                "dataset": {
                    "sources": [{"path": str(image_dataset)}],
                    "resolutions": [64],
                    "bucket_step": 16,
                    "batch_size": 2,
                    "num_workers": 0,
                },
                "adapter": {"algo": "lokr", "rank": 4, "alpha": 4},
                "loop": {"epochs": 1, "mixed_precision": "no"},
                "checkpoint": {"save_every_epochs": 1, "name": "demo"},
                "sampling": {
                    "enabled": True,
                    "every_epochs": 1,
                    "prompts": [{"prompt": "1girl", "steps": 2}],
                    "width": 64,
                    "height": 64,
                },
            }
            bad = await client.post("/api/config/validate", json={"config": {"adapter": {"algo": "nope"}}})
            assert bad.json()["ok"] is False and bad.json()["errors"][0]["loc"].startswith("adapter.algo")
            plan = (await client.post("/api/plan", json={"config": config})).json()
            assert plan["ok"] and plan["total_steps"] > 0 and plan["params"]["trainable"] > 0

            # subscribe to SSE before enqueueing
            events: list[str] = []

            async def consume():
                async with httpx.AsyncClient(base_url=live_server, timeout=None) as sse:
                    async with sse.stream("GET", "/api/events") as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("event: "):
                                events.append(line[7:])

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0.1)

            job = (
                await client.post(
                    "/api/jobs",
                    json={"type": "train", "name": "demo-run", "project_id": pid, "config": config},
                )
            ).json()
            jid = job["id"]
            assert job["status"] == "queued"
            job = await _wait_status(client, jid, {"completed", "failed"})
            if job["status"] != "completed":
                log = (await client.get(f"/api/jobs/{jid}/log")).json()
                raise AssertionError(
                    "job failed: "
                    + job.get("error", "")
                    + "\n"
                    + "\n".join(line["msg"] for line in log["lines"][-30:])
                )
            assert job["progress"]["step"] == job["progress"]["total_steps"]
            metrics = (await client.get(f"/api/jobs/{jid}/metrics")).json()
            assert len(metrics["steps"]) == job["progress"]["total_steps"] and metrics["loss"]
            samples = (await client.get(f"/api/jobs/{jid}/samples")).json()
            assert samples and samples[0]["url"].startswith(f"/api/jobs/{jid}/files")
            img = await client.get(samples[0]["url"])
            assert img.status_code == 200
            cks = (await client.get(f"/api/jobs/{jid}/checkpoints")).json()
            assert any(c["kind"] == "weights" for c in cks)
            arts = (await client.get(f"/api/artifacts?project_id={pid}")).json()
            assert arts and arts[0]["algo"] == "lokr"
            conv = (
                await client.post(f"/api/artifacts/{arts[0]['id']}/convert", json={"format": "kohya"})
            ).json()
            assert conv["kind"] == "kohya"
            log = (await client.get(f"/api/jobs/{jid}/log")).json()
            assert isinstance(log["lines"], list)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            assert "job.state" in events and "job.step" in events and "job.sample" in events
            # queue settings + retry
            r = await client.put("/api/queue/settings", json={"held": True})
            assert r.json()["held"] is True
            retry = (await client.post(f"/api/jobs/{jid}/retry")).json()
            assert retry["status"] == "queued" and retry["id"] != jid
            await asyncio.sleep(0.6)
            assert (await client.get(f"/api/jobs/{retry['id']}")).json()["status"] == "queued"  # held
            r = await client.post(f"/api/jobs/{retry['id']}/cancel")
            assert r.json()["status"] == "cancelled"
            r = await client.post(f"/api/jobs/{jid}/pause")
            assert r.status_code == 409 and r.json()["error"]["code"] == "job.bad_state"
            # openapi export
            spec = (await client.get("/api/openapi.json")).json()
            assert "/api/jobs/{jid}" in spec["paths"]


@pytest.mark.asyncio
async def test_pause_resume_via_api(live_server, image_dataset):
    if True:
        async with httpx.AsyncClient(base_url=live_server, timeout=30) as client:
            config = {
                "model": {"family": "toy", "dtype": "fp32"},
                "dataset": {
                    "sources": [{"path": str(image_dataset)}],
                    "resolutions": [64],
                    "bucket_step": 16,
                    "batch_size": 1,
                    "num_workers": 0,
                },
                "adapter": {"algo": "lora", "rank": 4, "alpha": 4},
                "loop": {"epochs": 40, "mixed_precision": "no"},
                "checkpoint": {"save_every_epochs": None, "name": "pr"},
            }
            job = (
                await client.post("/api/jobs", json={"type": "train", "name": "pr", "config": config})
            ).json()
            jid = job["id"]
            # wait for a few steps then pause
            for _ in range(300):
                j = (await client.get(f"/api/jobs/{jid}")).json()
                if j["progress"].get("step", 0) >= 3:
                    break
                await asyncio.sleep(0.2)
            r = await client.post(f"/api/jobs/{jid}/pause")
            assert r.json()["status"] in ("pausing", "paused")
            j = await _wait_status(client, jid, {"paused", "completed", "failed"})
            assert j["status"] == "paused", j
            assert j["resume_from"] and j["resume_from"].endswith("state-paused")
            paused_step = j["progress"]["step"]
            r = await client.post(f"/api/jobs/{jid}/resume")
            assert r.json()["status"] == "queued"
            j = await _wait_status(client, jid, {"completed", "failed"})
            assert j["status"] == "completed", j.get("error")
            assert j["progress"]["step"] == j["progress"]["total_steps"] > paused_step


@pytest.mark.asyncio
async def test_response_models_cover_every_json_endpoint(live_server, image_dataset, tmp_path):
    """Every JSON route declares a response model; the live handlers must validate against them."""
    async with httpx.AsyncClient(base_url=live_server, timeout=30) as client:
        spec = (await client.get("/api/openapi.json")).json()
        untyped = []
        for path, ops in spec["paths"].items():
            for method, op in ops.items():
                content = op.get("responses", {}).get("200", {}).get("content", {})
                schema = content.get("application/json", {}).get("schema")
                if schema is None:
                    continue  # files / SSE
                if schema == {} or (
                    schema.get("type") == "object" and "properties" not in schema and "$ref" not in schema
                ):
                    untyped.append(f"{method.upper()} {path}")
        allowed_untyped = {
            # free-form JSON (schema / config documents)
            "GET /api/schema/train",
            "GET /api/projects/{pid}/config",
            "PUT /api/projects/{pid}/config",
            "GET /api/jobs/{jid}/config",
            "POST /api/presets/{name}/resolve",
            # binary / streaming responses
            "GET /api/events",
            "GET /api/datasets/{did}/images/{h}/thumb",
            "GET /api/datasets/{did}/images/{h}/file",
            "GET /api/jobs/{jid}/files",
            "GET /api/artifacts/{aid}/download",
        }
        assert set(untyped) <= allowed_untyped, untyped
        # endpoints not touched by the training-flow test
        assert (await client.get("/api/system/stats")).status_code == 200
        assert (await client.get("/api/system/info")).json()["ypuddin"]
        fs = (await client.get(f"/api/fs/list?path={image_dataset}")).json()
        assert fs["entries"] and all({"name", "is_dir", "size", "mtime"} <= set(e) for e in fs["entries"])
        r = await client.post(
            "/api/presets", json={"name": "sweep", "description": "d", "config": {"adapter": {"rank": 8}}}
        )
        assert r.status_code in (200, 201) and r.json()["name"] == "sweep" and r.json()["builtin"] is False
        assert (
            await client.put("/api/presets/sweep", json={"name": "sweep", "config": {"adapter": {"rank": 4}}})
        ).json()["config"]["adapter"]["rank"] == 4
        assert (await client.get("/api/presets/sweep")).json()["config"]["adapter"]["rank"] == 4
        res = (
            await client.post("/api/presets/sweep/resolve", json={"config": {"model": {"family": "toy"}}})
        ).json()
        assert res["config"]["adapter"]["rank"] == 4
        assert (await client.delete("/api/presets/sweep")).json()["ok"] is True
        imp = (
            await client.post("/api/config/import-toml", json={"toml": "[model]\nfamily = 'toy'\n"})
        ).json()
        assert imp["ok"] is True and imp["config"]["model"]["family"] == "toy"
        weights = tmp_path / "w.safetensors"
        weights.write_bytes(b"\0" * 16)
        mdl = (
            await client.post("/api/models", json={"family": "toy", "kind": "dit", "path": str(weights)})
        ).json()
        assert mdl["exists"] is True and mdl["is_default"] is False and mdl["size"] == 16
        assert any(x["id"] == mdl["id"] for x in (await client.get("/api/models")).json())
        scanned = (await client.post("/api/models/scan", json={"path": str(tmp_path)})).json()
        assert isinstance(scanned, list)
        assert (await client.delete(f"/api/models/{mdl['id']}")).json()["ok"] is True
        assert (await client.get("/api/queue/settings")).json()["held"] in (True, False)
        p = (await client.post("/api/projects", json={"name": "sweep", "note": "n"})).json()
        assert p["stats"] == {"jobs": 0, "artifacts": 0} and p["dataset_ids"] == []
        p2 = (await client.patch(f"/api/projects/{p['id']}", json={"name": "sweep2"})).json()
        assert p2["name"] == "sweep2"
        d = (await client.post(f"/api/projects/{p['id']}/datasets", json={"path": str(image_dataset)})).json()
        assert d["source"]["project_id"] == p["id"] and d["index_status"] in ("indexing", "ready")
        assert (await client.post(f"/api/datasets/{d['source']['id']}/rescan")).json()["ok"] is True
        assert (await client.delete(f"/api/datasets/{d['source']['id']}")).json()["ok"] is True
        assert (await client.delete(f"/api/projects/{p['id']}")).json()["ok"] is True
