"""Real neural-network XYZ inference, durable API results and queue control."""

import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from ypuddin.config import TrainConfig
from ypuddin.server import create_app
from ypuddin.server.db import now
from ypuddin.server.xyz import XyzRequest, expand_cells
from ypuddin.server.xyz_worker import generate
from ypuddin.train import Trainer


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("xyz-trained")
    data = root / "data"
    data.mkdir()
    Image.new("RGB", (32, 32), (160, 70, 20)).save(data / "cat.png")
    (data / "cat.txt").write_text("cat")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(data)}],
                "resolutions": [32],
                "batch_size": 1,
                "num_workers": 0,
            },
            "adapter": {"algo": "lora", "rank": 2, "alpha": 2, "preset": "attn-only"},
            "loop": {"max_steps": 2, "mixed_precision": "no"},
            "optimizer": {"lr": 0.01},
            "sampling": {"enabled": False},
            "checkpoint": {"output_dir": str(root / "train"), "save_on_finish": True},
            "logging": {"tensorboard": False},
        }
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    path = next((root / "train").glob("*final.safetensors"))
    yield cfg, path
    torch.set_num_threads(old)


@pytest.fixture
def api(tmp_path, trained, monkeypatch):
    monkeypatch.setattr("ypuddin.server.supervisor.gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", poll_interval=0.02)
    context = app.state.ctx
    cfg, path = trained
    context.db.insert(
        "jobs",
        {
            "id": "source",
            "type": "train",
            "name": "trained",
            "status": "completed",
            "project_id": None,
            "created_at": now(),
            "run_dir": str(path.parent),
            "config_json": json.dumps(cfg.to_dict()),
        },
    )
    context.db.insert(
        "artifacts",
        {
            "id": "a_trained",
            "job_id": "source",
            "name": path.name,
            "path": str(path),
            "kind": "weights",
            "step": 2,
            "created_at": now(),
        },
    )
    context.db.set_kv("queue.settings", {"held": True, "max_concurrent": 1})
    with TestClient(app) as client:
        yield client, context


def body(**patch):
    return {
        "prompt": "cat",
        "width": 32,
        "height": 32,
        "steps": 2,
        "cfg": 1,
        "seed": 7,
        "checkpoint_id": "a_trained",
        "x": {"key": "adapter_scale", "values": [0, 1]},
        **patch,
    }


def create(api, **patch):
    client, context = api
    result = client.post("/api/jobs/source/xyz", json=body(**patch))
    assert result.status_code == 202, result.text
    jid = result.json()["id"]
    snapshot = json.loads(
        context.db.fetchone("SELECT config_json FROM jobs WHERE id=?", (jid,))["config_json"]
    )
    snapshot.update(device="cpu", fingerprint_cache=str(context.data_root / "fingerprints"))
    return jid, snapshot


def test_queue_filters_xyz_jobs_and_preserves_image_progress(api):
    client, context = api
    jid, _ = create(api)
    context.db.update("jobs", jid, {"progress_json": json.dumps({"done": 2, "total": 8})})
    response = client.get("/api/jobs", params={"type": "xyz", "group": "waiting"})
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    job = response.json()["items"][0]
    assert job["id"] == jid and job["progress"]["done"] == 2
    assert job["progress"]["total"] == 8
    trains = client.get("/api/jobs", params={"type": "train"}).json()["items"]
    assert [row["id"] for row in trains] == ["source"]


@pytest.mark.parametrize(
    "patch",
    [
        {"x": {"key": "seed", "values": [True]}},
        {"x": {"key": "steps", "values": [1.5]}},
        {"x": {"key": "cfg", "values": [-1]}},
        {"x": {"key": "adapter_scale", "values": [5]}},
        {"x": {"key": "sampler", "values": [{}]}},
        {"x": {"key": "cfg", "values": [1, 1]}},
        {"x": {"key": "cfg", "values": [1, 2]}, "y": {"key": "cfg", "values": [3]}},
        {"x": {"key": "seed", "values": list(range(9))}, "y": {"key": "cfg", "values": list(range(8))}},
        {"checkpoint_id": None},
    ],
)
def test_axis_values_reject_ambiguous_or_excessive_requests(patch):
    with pytest.raises(ValidationError):
        XyzRequest.model_validate(body(**patch))


def test_cells_reset_base_seed_and_parameters():
    request = XyzRequest.model_validate(
        body(
            x={"key": "steps", "values": [2, 4]},
            y={"key": "cfg", "values": [1, 3]},
            z={"key": "seed", "values": [9, 10]},
        )
    )
    cells = expand_cells(request)
    assert [(c["steps"], c["cfg"], c["seed"]) for c in cells] == [
        (s, c, t) for t in [9, 10] for c in [1, 3] for s in [2, 4]
    ]
    assert [(c["x"], c["y"], c["z"]) for c in cells] == [
        (x, y, z) for z in range(2) for y in range(2) for x in range(2)
    ]


def test_api_options_and_validation_before_enqueue(api):
    client, context = api
    options = client.get("/api/jobs/source/xyz/options").json()
    assert options["defaults"]["checkpoint_id"] == "a_trained"
    assert options["checkpoints"][0]["step"] == 2
    assert "path" not in options["checkpoints"][0]
    for patch in (
        {"checkpoint_id": "a_foreign"},
        {"sampler": "unimplemented"},
        {"sampling_model_id": "missing"},
    ):
        response = client.post("/api/jobs/source/xyz", json=body(**patch))
        assert response.status_code == 422, response.text
    assert context.db.fetchall("SELECT * FROM jobs WHERE type='xyz'") == []


def test_real_xyz_grid_adapter_changes_and_cancel_preserves_cells(api):
    client, context = api
    jid, payload = create(api, y={"key": "cfg", "values": [1, 2]}, z={"key": "seed", "values": [7, 8]})
    row = context.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    output = Path(row["samples_dir"])
    events = []
    generate(payload, output, lambda typ, **data: events.append((typ, data)), lambda: False)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and len(manifest["cells"]) == 8 and len(manifest["grids"]) == 2
    assert manifest["axes"]["x"]["label"] == "Adapter strength"
    assert (output / manifest["cells"][0]["file"]).read_bytes() != (
        output / manifest["cells"][1]["file"]
    ).read_bytes()
    assert all(Image.open(output / cell["file"]).size == (32, 32) for cell in manifest["cells"])
    detail = client.get(f"/api/xyz/{jid}").json()
    # The same endpoint serves <img> previews and download links. Keep it inline,
    # while naming each published grid/cell distinctly instead of always file.png.
    for image in detail["manifest"]["grids"] + detail["manifest"]["cells"][:1]:
        response = client.get(image["url"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["content-disposition"] == f'inline; filename="{jid}-{image["file"]}"'
        assert response.content == (output / image["file"]).read_bytes()
    assert client.get(f"/api/xyz/{jid}/file?name=../config.json").status_code == 403
    assert client.get("/api/jobs/source/xyz").json()[0]["id"] == jid
    other, payload = create(api)
    output = Path(context.db.fetchone("SELECT samples_dir FROM jobs WHERE id=?", (other,))["samples_dir"])
    progress = {"done": 0}

    def emit(typ, **data):
        if typ == "xyz.progress":
            progress.update(data)

    with pytest.raises(InterruptedError):
        generate(payload, output, emit, lambda: progress["done"] >= 1)
    partial = json.loads((output / "manifest.json").read_text())
    assert not partial["complete"] and len(partial["cells"]) == len(partial["grids"]) == 1
    assert not list(output.glob("*.tmp"))


def test_api_queued_cancel_retry_and_real_child_process(api):
    client, context = api
    jid, _ = create(api)
    assert client.post(f"/api/jobs/{jid}/pause").status_code == 409
    assert client.post(f"/api/xyz/{jid}/cancel").json()["status"] == "cancelled"
    retry = client.post(f"/api/jobs/{jid}/retry")
    assert retry.status_code == 200, retry.text
    other = retry.json()["id"]
    assert other != jid and retry.json()["run_dir"] != client.get(f"/api/jobs/{jid}").json()["run_dir"]
    client.put("/api/queue/settings", json={"held": False, "max_concurrent": 1})
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = client.get(f"/api/xyz/{other}").json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert result["status"] == "completed", result
    assert result["done"] == result["total"] == 2 and result["manifest"]["complete"]
    assert len(result["manifest"]["cells"]) == 2
    assert client.get(f"/api/xyz/{jid}").json()["status"] == "cancelled"
    assert client.get(f"/api/jobs/{other}").json()["type"] == "xyz"


def test_checkpoint_replaced_after_enqueue_fails_without_publishing(api, tmp_path):
    _, context = api
    _, payload = create(api)
    payload["xyz"]["checkpoints"]["a_trained"]["signature"][0] += 1
    with pytest.raises(ValueError, match="changed after"):
        generate(payload, tmp_path / "result", lambda *a, **k: None, lambda: False)
    manifest = json.loads((tmp_path / "result/manifest.json").read_text())
    assert not manifest["complete"] and not manifest["cells"]


@pytest.mark.parametrize(
    "algo,dora",
    [("lora", False), ("lokr", False), ("loha", False), ("full", False), ("lora", True), ("lokr", True)],
)
def test_exported_adapters_keep_full_strength_and_zero_is_base(tmp_path, algo, dora):
    import copy

    from ypuddin.adapters import TargetPreset, build_metadata, inject, save_adapter_file
    from ypuddin.config import AdapterConfig
    from ypuddin.server.xyz_worker import bind_checkpoint

    torch.manual_seed(2)
    original = torch.nn.Sequential(torch.nn.Linear(8, 8))
    adapted = copy.deepcopy(original)
    cfg = AdapterConfig(algo=algo, rank=2, alpha=2, dora=dora)
    adapters = inject(adapted, cfg, TargetPreset("all", ("*",)))
    x = torch.randn(2, 8)
    optimizer = torch.optim.SGD(adapters.parameters(), lr=0.1)
    adapted(x).square().mean().backward()
    optimizer.step()
    adapted.eval()
    tensors, targets = adapters.export_state()
    path = tmp_path / "adapter.safetensors"
    save_adapter_file(
        path,
        tensors,
        build_metadata(
            targets=targets, adapter_cfg=cfg.model_dump(), family="toy", architecture="toy", title="test"
        ),
        dtype="fp32",
    )
    restored = copy.deepcopy(original)
    bindings = bind_checkpoint(restored, path, "toy", "lora_unet")
    torch.testing.assert_close(restored(x), adapted(x), rtol=2e-5, atol=1e-6)
    for *_, wrapper in bindings:
        wrapper.multiplier = 0
    torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)
    for *_, wrapper in bindings:
        wrapper.multiplier = 0.5
    if dora:
        torch.testing.assert_close(
            restored(x), original(x) + 0.5 * (adapted(x) - original(x)), rtol=2e-5, atol=1e-6
        )


@pytest.mark.parametrize("cancel_after_bind", [None, 1, 2])
def test_checkpoint_axis_switches_actual_weights_and_releases_swap_hooks(
    api, tmp_path, monkeypatch, cancel_after_bind
):
    from ypuddin.adapters import load_adapter_file, save_adapter_file
    from ypuddin.memory import BlockSwapper
    from ypuddin.models.toy import ToyFamily
    from ypuddin.server import xyz_worker

    client, context = api
    original = Path(context.db.fetchone("SELECT path FROM artifacts WHERE id='a_trained'")["path"])
    tensors, metadata = load_adapter_file(original)
    for key in tensors:
        if key.endswith("lora_up.weight"):
            tensors[key] = torch.zeros_like(tensors[key])
    second = save_adapter_file(tmp_path / "later.safetensors", tensors, metadata, dtype="fp32")
    context.db.insert(
        "artifacts",
        {
            "id": "a_later",
            "job_id": "source",
            "name": "later",
            "path": str(second),
            "kind": "weights",
            "step": 3,
            "created_at": now(),
        },
    )
    assert client.get("/api/jobs/source/xyz/options").json()["defaults"]["checkpoint_id"] == "a_later"
    _, payload = create(api, x={"key": "checkpoint", "values": ["a_trained", "a_later"]})
    payload["memory"]["blocks_to_swap"] = 1
    events, loaded_models, swappers = [], [], []
    bound = 0
    original_bind = xyz_worker.bind_checkpoint

    class ObservedFamily(ToyFamily):
        def load(self, *args, **kwargs):
            loaded = super().load(*args, **kwargs)
            loaded_models.append(loaded)
            original_to = loaded.backbone.to

            def move(*args, **kwargs):
                target = args[0] if args else kwargs.get("device")
                if isinstance(target, (str, torch.device)) and torch.device(target).type == "cpu":
                    events.append("to_cpu")
                return original_to(*args, **kwargs)

            monkeypatch.setattr(loaded.backbone, "to", move)
            monkeypatch.setattr(loaded.text, "unload", lambda: events.append("text_unload"))
            monkeypatch.setattr(loaded.latent, "unload", lambda: events.append("latent_unload"))
            return loaded

        def materialize_backbone(self, loaded):
            super().materialize_backbone(loaded)
            events.append("materialized")

    class ObservedSwapper(BlockSwapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            swappers.append(self)

        def remove(self):
            super().remove()
            events.append("swap_removed")

    def bind(*args, **kwargs):
        nonlocal bound
        bound += 1
        events.append(f"bind_{bound}")
        return original_bind(*args, **kwargs)

    monkeypatch.setattr("ypuddin.models.get_family", lambda _: ObservedFamily())
    monkeypatch.setattr("ypuddin.memory.BlockSwapper", ObservedSwapper)
    monkeypatch.setattr(xyz_worker, "bind_checkpoint", bind)
    output = tmp_path / "result"

    def run():
        generate(payload, output, lambda *a, **k: None, lambda: bound == cancel_after_bind)

    if cancel_after_bind is None:
        run()
    else:
        with pytest.raises(InterruptedError, match="cancelled"):
            run()
    # Materialization may have preloaded CUDA weights. First binding must retain
    # that placement; only an existing checkpoint/swapper needs parking.
    first = events.index("materialized")
    assert events[first + 1] == "bind_1"
    if bound == 2:
        second_bind = events.index("bind_2")
        assert events[second_bind - 2 : second_bind] == ["swap_removed", "to_cpu"]
    assert events[-4:] == ["swap_removed", "to_cpu", "text_unload", "latent_unload"]
    assert len(swappers) == bound
    assert all(not swapper._handles and not swapper._masters for swapper in swappers)
    assert all(
        not block._forward_pre_hooks
        and not block._forward_hooks
        and not block._backward_pre_hooks
        and not block._backward_hooks
        for block in loaded_models[0].backbone.blocks
    )
    manifest = json.loads((output / "manifest.json").read_text())
    expected_cells = 2 if cancel_after_bind is None else cancel_after_bind - 1
    assert manifest["complete"] is (cancel_after_bind is None)
    assert [cell["checkpoint_id"] for cell in manifest["cells"]] == ["a_trained", "a_later"][:expected_cells]
    assert manifest["axes"]["x"]["labels"][-1] == "later"
    assert len(manifest["grids"]) == (1 if expected_cells else 0)
    assert not list(output.glob("*.tmp"))
    if expected_cells == 2:
        assert (output / manifest["cells"][0]["file"]).read_bytes() != (
            output / manifest["cells"][1]["file"]
        ).read_bytes()


def test_xyz_file_endpoint_refuses_symlink_to_unpublished_data(api, tmp_path):
    client, context = api
    jid, payload = create(api)
    row = context.db.fetchone("SELECT samples_dir FROM jobs WHERE id=?", (jid,))
    output = Path(row["samples_dir"])
    generate(payload, output, lambda *a, **k: None, lambda: False)
    result = client.get(f"/api/xyz/{jid}").json()
    cell = result["manifest"]["cells"][0]
    path = output / cell["file"]
    outside = tmp_path / "unrelated.png"
    path.replace(outside)
    path.symlink_to(outside)
    assert client.get(cell["url"]).status_code == 403
    assert outside.is_file()


def test_unicode_checkpoint_labels_wrap_by_pixel_width(tmp_path):
    from ypuddin.server.xyz_worker import _font, label_lines, write_grids

    font = _font()
    name = "训练模型_人物版本第二次保存.safetensors"
    lines = label_lines(name, font, 96, max_lines=3)
    assert len(lines) == 3 and lines[-1].endswith("…")
    assert lines[0].startswith("训练") and all(font.getlength(line) <= 96 for line in lines)
    assert not any("\\u" in line for line in lines)
    for local_cjk in (
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(local_cjk).is_file():
            assert bytes(font.getmask("训")) != bytes(font.getmask("\U0010ffff"))
            break
    request = XyzRequest.model_validate(
        body(width=128, height=128, x={"key": "checkpoint", "values": ["a_trained"]})
    )
    Image.new("RGB", (128, 128), "steelblue").save(tmp_path / "cell.png")
    manifest = {
        "cells": [{"x": 0, "y": 0, "z": 0, "file": "cell.png"}],
        "axes": {"x": {"key": "checkpoint", "label": "Checkpoint", "labels": [name]}},
        "grids": [],
    }
    write_grids(tmp_path, request, manifest)
    assert manifest["grids"][0]["axes"]["x"]["labels"] == [name]
    assert Image.open(tmp_path / "grid_z00.png").size == (272, 276)


def test_checkpoint_restoration_shares_swap_master_instead_of_retaining_cpu_copy(tmp_path):
    import copy

    from ypuddin.adapters import TargetPreset, build_metadata, inject, save_adapter_file
    from ypuddin.config import AdapterConfig
    from ypuddin.memory import BlockSwapper
    from ypuddin.server.xyz_worker import bind_checkpoint

    torch.manual_seed(4)
    backbone = torch.nn.Sequential(torch.nn.Linear(8, 8))
    reference = copy.deepcopy(backbone)
    adapted = copy.deepcopy(backbone)
    cfg = AdapterConfig(algo="lora", rank=2)
    adapters = inject(adapted, cfg, TargetPreset("all", ("*",)))
    tensors, targets = adapters.export_state()
    path = save_adapter_file(
        tmp_path / "adapter.safetensors",
        tensors,
        build_metadata(
            targets=targets, adapter_cfg=cfg.model_dump(), family="toy", architecture="toy", title="test"
        ),
        dtype="fp32",
    )
    bindings = bind_checkpoint(backbone, path, "toy", "lora_unet")
    parent, attr, restoration, wrapper = bindings[0]
    assert restoration is wrapper.base
    old_pointer = restoration.weight.data_ptr()
    swapper = BlockSwapper([wrapper], 1, "cpu", pin_memory=False, prefetch=False)
    assert restoration.weight.data_ptr() != old_pointer
    assert restoration.weight.data_ptr() == wrapper.base.weight.data_ptr()
    swapper.remove()
    setattr(parent, attr, restoration)
    x = torch.randn(2, 8)
    torch.testing.assert_close(backbone(x), reference(x), rtol=0, atol=0)


def test_unconfirmed_krea_alternative_cannot_silently_inherit_raw(api, tmp_path):
    client, context = api
    source = context.db.fetchone("SELECT config_json FROM jobs WHERE id='source'")
    cfg = json.loads(source["config_json"])
    cfg["model"]["family"] = "krea2"
    cfg["model"]["krea2_variant"] = "raw"
    context.db.update("jobs", "source", {"config_json": json.dumps(cfg)})
    path = tmp_path / "arbitrary_local_weights.safetensors"
    path.write_bytes(b"legacy model file")
    context.db.insert(
        "models",
        {
            "id": "legacy",
            "family": "krea2",
            "kind": "dit",
            "path": str(path),
            "is_default": 0,
            "created_at": now(),
        },
    )
    options = client.get("/api/jobs/source/xyz/options").json()
    assert options["sampling_models"] == []
    response = client.post("/api/jobs/source/xyz", json=body(sampling_model_id="legacy"))
    assert response.status_code == 422 and response.json()["error"]["code"] == "xyz.variant"
    assert not context.db.fetchall("SELECT id FROM jobs WHERE type='xyz'")


@pytest.mark.parametrize(
    "variant, supplied, expected",
    [
        ("turbo", {}, (8, 0.0)),
        ("turbo", {"steps": 4}, (4, 0.0)),
        ("turbo", {"cfg": 2.0}, (8, 2.0)),
        ("turbo", {"steps": 20, "cfg": 4.0}, (20, 4.0)),
        ("raw", {}, (20, 4.0)),
    ],
)
def test_selected_turbo_api_fills_only_omitted_sampling_values(api, tmp_path, variant, supplied, expected):
    from tests.unit.test_krea2_turbo import krea_file

    client, context = api
    path = krea_file(tmp_path / "confirmed-local-model.safetensors")
    cfg = json.loads(context.db.fetchone("SELECT config_json FROM jobs WHERE id='source'")["config_json"])
    cfg["model"].update(
        family="krea2",
        krea2_variant="raw",
        dit_path=str(path),
        text_encoder_path=str(path),
        vae_path=str(path),
    )
    context.db.update("jobs", "source", {"config_json": json.dumps(cfg)})
    context.db.insert(
        "models",
        {
            "id": "selected",
            "family": "krea2",
            "kind": "dit",
            "path": str(path),
            "variant": variant,
            "purpose": "inference" if variant == "turbo" else "training",
            "is_default": 0,
            "created_at": now(),
        },
    )
    request = {
        "prompt": "cat",
        "width": 32,
        "height": 32,
        "sampling_model_id": "selected",
        "x": {"key": "seed", "values": [42]},
        **supplied,
    }
    response = client.post("/api/jobs/source/xyz", json=request)
    assert response.status_code == 202, response.text
    saved = response.json()["request"]
    assert (saved["steps"], saved["cfg"]) == expected
    persisted = client.get(f"/api/jobs/{response.json()['id']}/config").json()
    assert (persisted["xyz"]["request"]["steps"], persisted["xyz"]["request"]["cfg"]) == expected
    assert persisted["model"]["krea2_variant"] == variant
    # Axis values override the resolved fixed defaults without being rewritten.
    axes = {"x": {"key": "steps", "values": [3, 7]}, "y": {"key": "cfg", "values": [0.5, 2.5]}}
    response = client.post("/api/jobs/source/xyz", json=request | axes)
    assert response.status_code == 202, response.text
    resolved = XyzRequest.model_validate(response.json()["request"])
    assert resolved.x.model_dump() == axes["x"] and resolved.y.model_dump() == axes["y"]
    assert {(cell["steps"], cell["cfg"]) for cell in expand_cells(resolved)} == {
        (3, 0.5),
        (7, 0.5),
        (3, 2.5),
        (7, 2.5),
    }


def test_running_xyz_cancellation_stops_real_worker_and_retains_history(api):
    client, context = api
    jid, _ = create(api, steps=100, x={"key": "seed", "values": list(range(12))})
    client.put("/api/queue/settings", json={"held": False, "max_concurrent": 1})
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        task = client.get(f"/api/xyz/{jid}").json()
        if task["phase"] == "sampling" or task["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert task["phase"] == "sampling", task
    response = client.post(f"/api/xyz/{jid}/cancel")
    assert response.status_code == 200 and response.json()["status"] in {"cancelling", "cancelled"}
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        task = client.get(f"/api/xyz/{jid}").json()
        if task["status"] == "cancelled" and not context.supervisor.is_running(jid):
            break
        time.sleep(0.03)
    assert task["status"] == "cancelled" and not context.supervisor.is_running(jid), task
    assert not task["manifest"]["complete"] and task["done"] < task["total"]
    assert client.get("/api/jobs/source/xyz").json()[0]["id"] == jid


@pytest.mark.parametrize("failure", ["cancel_after_load", "text_error"])
def test_unmaterialized_backbone_cleanup_preserves_original_failure(api, monkeypatch, failure):
    from types import SimpleNamespace
    from unittest.mock import Mock

    jid, payload = create(api)
    context = api[1]
    output = Path(context.db.fetchone("SELECT samples_dir FROM jobs WHERE id=?", (jid,))["samples_dir"])
    text = Mock()
    text.encode.side_effect = ValueError("text encoding failed")
    latent = Mock()
    loaded = SimpleNamespace(
        backbone=torch.nn.Linear(2, 2, device="meta"),
        extra={"materialized": False},
        text=text,
        latent=latent,
    )
    has_loaded = False

    def load(*args, **kwargs):
        nonlocal has_loaded
        has_loaded = True
        return loaded

    family = SimpleNamespace(load=load, sampling_needs_uncond=lambda *_: False)
    monkeypatch.setattr("ypuddin.models.get_family", lambda _: family)
    expected = InterruptedError if failure == "cancel_after_load" else ValueError
    message = "XYZ sampling cancelled" if failure == "cancel_after_load" else "text encoding failed"
    with pytest.raises(expected, match=message):
        generate(
            payload, output, lambda *a, **kw: None, lambda: has_loaded and failure == "cancel_after_load"
        )
    text.unload.assert_called_once()
    latent.unload.assert_called_once()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] is False and manifest["cells"] == []
    assert not list(output.glob("*.tmp"))
