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
            assert r.status_code == 404 and r.json()["error"]["code"] == "preset.not_found" and "trace_id" in r.json()["error"]
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
            r = await client.put(f"/api/datasets/{did}/images/{h0}/caption", json={"caption": "1girl, edited"})
            assert (await client.get(f"/api/datasets/{did}/images/{h0}/caption")).json()["caption"] == "1girl, edited"
            r = await client.post(f"/api/datasets/{did}/tags/batch", json={"hashes": [h0], "add": ["newtag"], "remove": ["edited"]})
            assert r.json()["changed"] == 1
            assert "newtag" in (await client.get(f"/api/datasets/{did}/images/{h0}/caption")).json()["caption"]

            # plan + validate
            config = {
                "model": {"family": "toy", "dtype": "fp32"},
                "dataset": {"sources": [{"path": str(image_dataset)}], "resolutions": [64], "bucket_step": 16, "batch_size": 2, "num_workers": 0},
                "adapter": {"algo": "lokr", "rank": 4, "alpha": 4},
                "loop": {"epochs": 1, "mixed_precision": "no"},
                "checkpoint": {"save_every_epochs": 1, "name": "demo"},
                "sampling": {"enabled": True, "every_epochs": 1, "prompts": [{"prompt": "1girl", "steps": 2}], "width": 64, "height": 64},
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

            job = (await client.post("/api/jobs", json={"type": "train", "name": "demo-run", "project_id": pid, "config": config})).json()
            jid = job["id"]
            assert job["status"] == "queued"
            job = await _wait_status(client, jid, {"completed", "failed"})
            if job["status"] != "completed":
                log = (await client.get(f"/api/jobs/{jid}/log")).json()
                raise AssertionError("job failed: " + job.get("error", "") + "\n" + "\n".join(l["msg"] for l in log["lines"][-30:]))
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
            conv = (await client.post(f"/api/artifacts/{arts[0]['id']}/convert", json={"format": "kohya"})).json()
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
                "dataset": {"sources": [{"path": str(image_dataset)}], "resolutions": [64], "bucket_step": 16, "batch_size": 1, "num_workers": 0},
                "adapter": {"algo": "lora", "rank": 4, "alpha": 4},
                "loop": {"epochs": 40, "mixed_precision": "no"},
                "checkpoint": {"save_every_epochs": None, "name": "pr"},
            }
            job = (await client.post("/api/jobs", json={"type": "train", "name": "pr", "config": config})).json()
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
