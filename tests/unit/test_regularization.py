"""Regularization uses real Toy subprocesses and controlled provider HTTP responses only."""

import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import CaptionConfig, DatasetSourceConfig, TrainConfig
from ypuddin.data.dataset import prepare_data_layout
from ypuddin.models import get_family
from ypuddin.server import create_app, routes_work
from ypuddin.server.environment import EnvironmentError, maintenance_blocked
from ypuddin.server.regularization import RESERVATION, RegularizationManager, _allowed_media, _Redirect
from ypuddin.server.routes_regularization import RegularizationRequest


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Class prior"}).json()
    pid, vid = project["id"], project["active_version_id"]
    config = client.get(f"/api/projects/{pid}/config").json()
    config["model"].update(family="toy", dtype="fp32")
    config["dataset"].update(resolutions=[64], bucket_step=16)
    config["dataset"]["caption"]["trigger_word"] = "instance_trigger"
    assert client.put(f"/api/projects/{pid}/config", json=config).status_code == 200
    yield client, app, pid, vid
    app.state.regularization.close()
    app.state.dataset_pipeline.close()
    app.state.environment.close()
    app.state.model_downloads.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def start(api, **fields):
    client, _, pid, vid = api
    return client.post(
        f"/api/projects/{pid}/versions/{vid}/regularization",
        json={
            "prompt": "dog\ncat",
            "count": 2,
            "width": 64,
            "height": 64,
            "steps": 2,
            "cfg": 1,
            **fields,
        },
    )


def finished(api, oid):
    manager = api[1].state.regularization
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = manager.get(oid)
        # Terminal status precedes lease release by a few instructions.
        if result["status"] in {"completed", "failed", "cancelled"} and oid not in manager.cancel_events:
            return result
        time.sleep(0.02)
    pytest.fail(f"Regularization timed out: {manager.get(oid)}")


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(stream, format="PNG")
    return stream.getvalue()


class Provider:
    def __init__(self, source="danbooru", colors=("red", "blue")):
        self.source, self.colors, self.calls = source, colors, []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.pause = False
        self.error = False
        self.malicious = False

    def open(self, request, timeout):
        self.calls.append(request)
        if request.get_header("Accept") == "application/json":
            self.entered.set()
            if self.pause:
                assert self.release.wait(10)
            if self.error:
                raise urllib.error.URLError("url contained top-secret-key")
            domain = "cdn.donmai.us" if self.source == "danbooru" else "img3.gelbooru.com"
            key = "tag_string" if self.source == "danbooru" else "tags"
            posts = [
                {
                    "id": i + 1,
                    "rating": "g" if self.source == "danbooru" else "general",
                    "file_url": f"https://{domain}/{i}.png",
                    key: "dog animal outdoors",
                    "file_ext": "png",
                }
                for i in range(len(self.colors))
            ]
            if self.malicious:
                posts = [
                    {**posts[0], "rating": "e"},
                    {**posts[0], "id": 90, "file_url": "https://localhost/secret.png"},
                    *posts,
                ]
            payload = posts if self.source == "danbooru" else {"post": posts}
            return io.BytesIO(json.dumps(payload).encode())
        index = int(urllib.parse.urlsplit(request.full_url).path.strip("/").split(".")[0])
        return io.BytesIO(png(self.colors[index]))


def assert_unpublished(api):
    client, app, pid, vid = api
    assert client.get(f"/api/projects/{pid}/config").json()["dataset"]["sources"] == []
    root = app.state.ctx.reg_dir(pid, vid)
    assert not root.exists() or list(root.iterdir()) == []
    assert app.state.ctx.db.fetchall("SELECT * FROM datasets") == []


@pytest.mark.parametrize(
    "algorithm",
    [
        None,
        {
            "sampler": "er_sde",
            "scheduler": "normal",
            "er_sde_order": 2,
            "er_sde_s_noise": 0.3,
            "shift": 1.75,
            "guidance": 2.5,
        },
    ],
)
def test_real_toy_manager_publishes_images_captions_and_trainable_source(api, algorithm):
    if algorithm:
        config = api[0].get(f"/api/projects/{api[2]}/config").json()
        config["sampling"].update(**algorithm, steps=77, cfg=9, seed=987)
        assert api[0].put(f"/api/projects/{api[2]}/config", json=config).status_code == 200
    response = start(
        api,
        prompt="dog, instance_trigger\ncat",
        excluded_tags=["instance_trigger"],
        prior_weight=0.3,
        repeats=2,
    )
    assert response.status_code == 202, response.text
    task = finished(api, response.json()["id"])
    assert task["status"] == "completed", task
    assert task["images"] == task["done"] == 2
    path = Path(task["path"])
    assert path.parent == api[1].state.ctx.reg_dir(api[2], api[3])
    assert {p.read_text() for p in path.glob("*.txt")} == {"dog", "cat"}
    assert [Image.open(p).size for p in path.glob("*.png")] == [(64, 64), (64, 64)]
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(manifest) == 2
    expected = algorithm or {
        "sampler": "euler",
        "scheduler": "uniform",
        "er_sde_order": 3,
        "er_sde_s_noise": 1.0,
    }
    for item in manifest:
        assert {key: item[key] for key in expected} == expected
        assert item["steps"] == 2 and item["cfg"] == 1
        if algorithm is None:
            assert item["guidance"] is None
    payload = json.loads(api[1].state.regularization._row(task["id"])["request_json"])
    assert {key: payload["sampling"][key] for key in expected} == expected
    assert [item["seed"] for item in manifest] == [payload["seed"], payload["seed"] + 1]
    config = api[0].get(f"/api/projects/{api[2]}/config").json()
    source = config["dataset"]["sources"][0]
    assert source["is_reg"] and source["prior_weight"] == 0.3 and source["repeats"] == 2
    assert not source["caption"]["trigger_word"]
    layout = prepare_data_layout(TrainConfig.model_validate(config), get_family("toy").spec.latent)
    assert len(layout.items) == 4 and all(item.is_reg and item.weight == 0.3 for item in layout.items)
    assert not maintenance_blocked(api[1].state.ctx.db)
    assert api[0].get(f"/api/projects/{api[2]}/versions/{api[3]}/regularization").json()["images"] == 2


@pytest.mark.parametrize("source", ["danbooru", "gelbooru"])
def test_web_collect_safe_only_tags_credentials_and_no_secret_persistence(api, source):
    provider = Provider(source)
    provider.malicious = True
    api[1].state.regularization.opener = provider
    response = start(
        api,
        source=source,
        prompt="dog",
        username="123",
        api_key="top-secret-key",
        excluded_tags=["watermark"],
    )
    assert response.status_code == 202, response.text
    task = finished(api, response.json()["id"])
    assert task["status"] == "completed", task
    requests = provider.calls
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requests[0].full_url).query)
    assert query["tags"] == ["dog -watermark rating:general"]
    if source == "gelbooru":
        assert query["user_id"] == ["123"] and query["api_key"] == ["top-secret-key"]
    else:
        assert requests[0].get_header("Authorization", "").startswith("Basic ")
    assert len(requests) == 3  # Unsafe-rated and foreign-host posts were never fetched.
    assert all(not req.get_header("Authorization") and "api_key" not in req.full_url for req in requests[1:])
    assert all(p.read_text() == "dog, animal, outdoors" for p in Path(task["path"]).glob("*.txt"))
    row = api[1].state.regularization._row(task["id"])
    assert "top-secret-key" not in json.dumps(row)
    assert "username" not in row["request_json"] and "api_key" not in row["request_json"]
    assert "top-secret-key" not in api[0].get(f"/api/regularization/{task['id']}").text
    assert all("top-secret-key" not in p.read_text() for p in Path(task["path"]).glob("*.json"))


def test_network_failure_rolls_back_without_reflecting_credentials(api):
    provider = Provider()
    provider.error = True
    api[1].state.regularization.opener = provider
    response = start(api, source="danbooru", prompt="dog", username="123", api_key="top-secret-key")
    task = finished(api, response.json()["id"])
    assert task["status"] == "failed" and "Could not read danbooru" in task["error"]
    assert "top-secret-key" not in json.dumps(task)
    assert_unpublished(api)


def test_cancel_web_preserves_existing_files_and_releases_version(api):
    provider = Provider()
    provider.pause = True
    manager = api[1].state.regularization
    manager.opener = provider
    sentinel = api[1].state.ctx.version_dir(api[2], api[3]) / "keep.txt"
    sentinel.write_text("existing runtime data")
    response = start(api, source="danbooru", prompt="dog")
    oid = response.json()["id"]
    assert provider.entered.wait(5)
    assert start(api, source="danbooru", prompt="cat").status_code == 409
    cancelled = api[0].post(f"/api/regularization/{oid}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelling"
    provider.release.set()
    assert finished(api, oid)["status"] == "cancelled"
    assert sentinel.read_text() == "existing runtime data"
    assert_unpublished(api)
    with api[1].state.ctx.versions.mutation(api[2], api[3]):
        pass


def test_ai_reservation_blocks_queue_environment_and_cancel_releases(api, monkeypatch):
    manager = api[1].state.regularization
    entered = threading.Event()

    def waiting(oid, payload, staging, images, cancelled):
        entered.set()
        assert cancelled.wait(10)
        manager._check(cancelled)

    monkeypatch.setattr(manager, "_generate", waiting)
    result = start(api)
    assert result.status_code == 202
    oid = result.json()["id"]
    assert entered.wait(5)
    assert maintenance_blocked(api[1].state.ctx.db)
    with pytest.raises(EnvironmentError, match="regularization"):
        api[1].state.environment._idle()
    # A different writable version cannot start another accelerator task either.
    p2 = api[0].post("/api/projects", json={"name": "Other"}).json()
    rejected = api[0].post(
        f"/api/projects/{p2['id']}/versions/{p2['active_version_id']}/regularization", json={"prompt": "dog"}
    )
    assert rejected.status_code == 409
    api[0].post(f"/api/regularization/{oid}/cancel")
    assert finished(api, oid)["status"] == "cancelled"
    assert not maintenance_blocked(api[1].state.ctx.db)
    assert_unpublished(api)


def test_environment_blocks_ai_before_creating_operation_but_not_web(api):
    db = api[1].state.ctx.db
    db.set_kv("environment.maintenance", {"blocked": True})
    assert start(api).status_code == 409
    assert db.fetchall("SELECT * FROM regularization_operations") == []
    provider = Provider()
    api[1].state.regularization.opener = provider
    response = start(api, source="danbooru", prompt="dog")
    assert response.status_code == 202
    assert finished(api, response.json()["id"])["status"] == "completed"


def test_validation_never_reflects_invalid_secret_and_query_is_constrained(api):
    response = start(api, source="unknown", api_key="top-secret-key", count=999)
    assert response.status_code == 422 and "top-secret-key" not in response.text
    assert start(api, source="danbooru", prompt="rating:explicit dog").status_code == 422
    assert (
        start(api, source="gelbooru", prompt="dog", username="not-a-number", api_key="secret").status_code
        == 422
    )
    assert start(api, width=63).status_code == 422
    assert_unpublished(api)


def test_restart_discards_only_unpublished_batch_and_marks_failure(api):
    manager = api[1].state.regularization
    manager.close()
    c, pid, vid = api[1].state.ctx, api[2], api[3]
    oid = "reg_interrupted"
    staging = c.reg_dir(pid, vid) / f".staging-{oid}"
    staging.mkdir(parents=True)
    (staging / ".regularization-owner").write_text("owned-test")
    (staging / "partial.png").write_bytes(png("red"))
    keep = c.reg_dir(pid, vid) / "older"
    keep.mkdir()
    (keep / "data.txt").write_text("preserved")
    c.db.insert(
        "regularization_operations",
        {
            "id": oid,
            "project_id": pid,
            "version_id": vid,
            "source": "ai",
            "status": "running",
            "phase": "generating",
            "total": 2,
            "request_json": '{"ownership_token":"owned-test"}',
            "created_at": time.time(),
        },
    )
    c.db.set_kv(RESERVATION, {"id": oid})
    replacement = RegularizationManager(c)
    api[1].state.regularization = replacement
    assert replacement.get(oid)["status"] == "failed"
    assert not staging.exists() and (keep / "data.txt").read_text() == "preserved"
    assert not maintenance_blocked(c.db)


def test_source_publish_index_failure_cleans_config_and_files(api, monkeypatch):
    api[1].state.regularization.opener = Provider()

    def failed_index(c, did):
        c.db.update("datasets", did, {"index_status": "failed"})

    monkeypatch.setattr(routes_work, "_index_dataset", failed_index)
    response = start(api, source="danbooru", prompt="dog")
    task = finished(api, response.json()["id"])
    assert task["status"] == "failed" and "indexed" in task["error"]
    assert_unpublished(api)


def test_media_hosts_and_redirects_never_forward_credentials():
    for url in [
        "http://cdn.donmai.us/a.png",
        "https://cdn.donmai.us.evil.com/x",
        "https://user:pass@cdn.donmai.us/x",
        "https://cdn.donmai.us:444/x",
        "https://127.0.0.1/x",
    ]:
        assert not _allowed_media(url, "danbooru")
    request = urllib.request.Request(
        "https://danbooru.donmai.us/posts.json", headers={"Authorization": "Basic secret", "Cookie": "secret"}
    )
    with pytest.raises(Exception, match="redirected"):
        _Redirect("danbooru", False).redirect_request(request, None, 302, "", {}, "https://cdn.donmai.us/x")
    redirected = _Redirect("danbooru", True).redirect_request(
        request, None, 302, "", {}, "https://cdn.donmai.us/x"
    )
    assert not redirected.get_header("Authorization") and not redirected.get_header("Cookie")


def test_secret_model_dump_is_not_a_credential_store():
    payload = RegularizationRequest(
        prompt="dog", username="alice", user_id="42", api_key="secret"
    ).model_dump(mode="json")
    assert not {"username", "user_id", "api_key"} & payload.keys()


def test_real_ai_child_cancel_exits_and_publishes_nothing(api):
    response = start(api, count=200, steps=100)
    assert response.status_code == 202
    oid = response.json()["id"]
    manager = api[1].state.regularization
    deadline = time.monotonic() + 10
    while oid not in manager.processes and time.monotonic() < deadline:
        time.sleep(0.02)
    process = manager.processes[oid]
    assert api[0].post(f"/api/regularization/{oid}/cancel").status_code == 200
    assert finished(api, oid)["status"] == "cancelled"
    assert process.poll() is not None
    assert not maintenance_blocked(api[1].state.ctx.db)
    assert_unpublished(api)


def test_exact_duplicates_do_not_create_a_second_trainable_batch(api):
    provider = Provider(colors=("red", "red", "blue"))
    api[1].state.regularization.opener = provider
    first = start(api, source="danbooru", prompt="dog")
    batch = finished(api, first.json()["id"])
    assert batch["status"] == "completed" and batch["duplicates"] == 1
    original = Path(batch["path"])
    second = start(api, source="danbooru", prompt="dog")
    rejected = finished(api, second.json()["id"])
    assert rejected["status"] == "failed" and rejected["duplicates"] == 3
    assert len(list(original.glob("*.png"))) == 2
    assert len(api[1].state.ctx.db.fetchall("SELECT * FROM datasets")) == 1
    assert list(original.parent.iterdir()) == [original]


def test_download_byte_limit_removes_partial_images(api, monkeypatch):
    import ypuddin.server.regularization as reg

    monkeypatch.setattr(reg, "MAX_FILE_BYTES", 8)
    api[1].state.regularization.opener = Provider()
    response = start(api, source="danbooru", prompt="dog")
    task = finished(api, response.json()["id"])
    assert task["status"] == "failed" and "byte limit" in task["error"]
    assert_unpublished(api)


def test_unconfirmed_orphan_process_keeps_accelerator_reserved(api, monkeypatch):
    import psutil

    api[1].state.regularization.close()
    c, pid, vid = api[1].state.ctx, api[2], api[3]
    oid = "reg_unknown_process"
    staging = c.reg_dir(pid, vid) / f".staging-{oid}"
    staging.mkdir(parents=True)
    (staging / "partial.png").write_bytes(png("red"))
    c.db.insert(
        "regularization_operations",
        {
            "id": oid,
            "project_id": pid,
            "version_id": vid,
            "source": "ai",
            "status": "running",
            "phase": "loading",
            "total": 2,
            "request_json": "{}",
            "created_at": time.time(),
            "worker_pid": 123456,
        },
    )
    c.db.set_kv(RESERVATION, {"id": oid})

    def denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    replacement = RegularizationManager(c)
    api[1].state.regularization = replacement
    assert replacement.get(oid)["status"] == "failed"
    assert staging.exists() and maintenance_blocked(c.db)
    assert c.db.get_kv(RESERVATION)["recovery_required"]


def test_regularization_never_enters_auto_validation_and_has_class_captions(tmp_path):
    train, prior = tmp_path / "train", tmp_path / "prior"
    train.mkdir()
    prior.mkdir()
    for index in range(30):
        path = train / f"{index}.png"
        Image.new("RGB", (64, 64), (index * 3, 7, 0)).save(path)
        path.with_suffix(".txt").write_text("dog")
    for index in range(4):
        path = prior / f"{index}.png"
        Image.new("RGB", (64, 64), (index * 7, 30, 90)).save(path)
        path.with_suffix(".txt").write_text("animal")
    cfg = TrainConfig()
    cfg.dataset.resolutions = [64]
    cfg.dataset.bucket_step = 16
    cfg.dataset.caption = CaptionConfig(trigger_word="instance_trigger")
    cfg.dataset.sources = [
        DatasetSourceConfig(path=str(train)),
        DatasetSourceConfig(path=str(prior), is_reg=True, prior_weight=0.25),
    ]
    cfg.validation.enabled = True
    cfg.validation.split_ratio = 0.5
    layout = prepare_data_layout(cfg, get_family("toy").spec.latent)
    priors = [item for item in layout.items if item.is_reg]
    assert len(priors) == 4 and all(
        item.weight == 0.25 and not item.caption_cfg.trigger_word for item in priors
    )
    assert layout.validation_items and all(not item.is_reg for item in layout.validation_items)
    assert all(
        item.caption_cfg.trigger_word == "instance_trigger" for item in layout.items if not item.is_reg
    )
    cfg.dataset.sources[1].caption = CaptionConfig(prefix="explicit class prefix")
    explicit = prepare_data_layout(cfg, get_family("toy").spec.latent)
    assert all(item.caption_cfg.prefix == "explicit class prefix" for item in explicit.items if item.is_reg)
    # Explicit validation still wins if users deliberately assign an overlapping directory.
    cfg.validation.sources = [DatasetSourceConfig(path=str(prior))]
    explicit = prepare_data_layout(cfg, get_family("toy").spec.latent)
    assert all(not item.is_reg for item in explicit.items)


def test_prior_weight_scales_per_sample_then_batch_mean_without_separate_normalization():
    import torch

    from ypuddin.config import ObjectiveConfig
    from ypuddin.objectives.flow import reduce_loss

    loss, per_sample = reduce_loss(
        torch.tensor([[[[4.0]]], [[[8.0]]]]),
        torch.tensor([0.5, 0.5]),
        ObjectiveConfig(weighting="none"),
        sample_weight=torch.tensor([1.0, 0.25]),
    )
    assert per_sample.tolist() == [4.0, 8.0]
    assert loss.item() == 3.0  # (4*1 + 8*.25) / 2; repeats govern the class/sample proportion.


def test_completed_batch_survives_restart_between_commit_and_reservation_release(api):
    api[1].state.regularization.opener = Provider()
    response = start(api, source="danbooru", prompt="dog")
    batch = finished(api, response.json()["id"])
    assert batch["status"] == "completed"
    api[1].state.regularization.close()
    context = api[1].state.ctx
    context.db.update("regularization_operations", batch["id"], {"source": "ai"})
    context.db.set_kv(RESERVATION, {"id": batch["id"]})
    replacement = RegularizationManager(context)
    api[1].state.regularization = replacement
    assert replacement.get(batch["id"])["status"] == "completed"
    assert len(list(Path(batch["path"]).glob("*.png"))) == 2
    assert context.db.fetchone("SELECT id FROM datasets WHERE id=?", (batch["dataset_id"],))
    source = api[0].get(f"/api/projects/{api[2]}/config").json()["dataset"]["sources"][0]
    assert source["path"] == batch["path"] and source["is_reg"]
    assert not maintenance_blocked(context.db)


def test_preexisting_batch_directory_is_never_deleted(api, monkeypatch):
    import ypuddin.server.regularization as reg

    monkeypatch.setattr(reg, "new_id", lambda prefix: "reg_existing")
    folder = api[1].state.ctx.reg_dir(api[2], api[3]) / "reg_existing"
    folder.mkdir(parents=True)
    (folder / "old.txt").write_text("original data")
    response = start(api, source="danbooru", prompt="dog")
    assert response.status_code == 409
    assert (folder / "old.txt").read_text() == "original data"
    assert api[1].state.ctx.db.fetchall("SELECT * FROM regularization_operations") == []


def test_publish_path_collision_after_start_keeps_unowned_directory(api, monkeypatch):
    manager = api[1].state.regularization
    manager.opener = Provider()
    collect = manager._collect
    collision = []

    def occupy_final(oid, payload, credentials, output, cancelled):
        collect(oid, payload, credentials, output, cancelled)
        folder = api[1].state.ctx.reg_dir(api[2], api[3]) / oid
        folder.mkdir()
        (folder / "old.txt").write_text("unrelated directory")
        collision.append(folder)

    monkeypatch.setattr(manager, "_collect", occupy_final)
    response = start(api, source="danbooru", prompt="dog")
    task = finished(api, response.json()["id"])
    assert task["status"] == "failed"
    assert (collision[0] / "old.txt").read_text() == "unrelated directory"
    assert api[1].state.ctx.db.fetchall("SELECT * FROM datasets") == []
    assert list(collision[0].parent.iterdir()) == collision
