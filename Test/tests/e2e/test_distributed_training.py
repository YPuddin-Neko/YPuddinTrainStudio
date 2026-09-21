"""Real two-process DDP training, rank isolation, preview and exact resume on CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from Test.tests.checkpoint_assertions import assert_checkpoint_value_exact
from Test.tests.conftest import make_image_dataset

ROOT = Path(__file__).resolve().parents[3]


def _launch(config, *, expect_success=True, worker="Test.tests.ddp_worker"):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "YPUDDIN_DDP_TIMEOUT_SECONDS": "60"}
    # Keep local CPU tests off hostname DNS and proxy virtual interfaces.
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    # An outer torchrun must never leak its process group into this independent test.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=127.0.0.1:0",
            "--rdzv_conf=is_host=true",
            "--local_addr=127.0.0.1",
            "--nproc_per_node=2",
            "-m",
            worker,
            str(config),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=100,
    )
    if expect_success:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def _config(tmp_path, training):
    images = make_image_dataset(tmp_path / "images", n=13, sizes=((64, 64), (96, 64)))
    return {
        "model": {"family": "toy", "dtype": "fp32"},
        "training": training,
        "dataset": {
            "sources": [{"path": str(images)}],
            "resolutions": [64],
            "bucket_step": 16,
            "batch_size": 2,
            "num_workers": 0,
            "cache_dir": str(tmp_path / "cache"),
            "text_encoding": "online",
            "caption": {"shuffle": True, "caption_dropout": 0.1},
        },
        "adapter": {"algo": "lokr", "rank": 4, "alpha": 4, "preset": "attn-mlp", "module_dropout": 0.2},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "memory": {"base_precision": "fp32", "offload_text_encoder": False},
        "loop": {"epochs": None, "max_steps": 4, "grad_accum": 2, "mixed_precision": "no", "seed": 19},
        "checkpoint": {
            "name": "ddp",
            "output_dir": str(tmp_path / "reference"),
            "save_every_epochs": None,
            "save_state_every_steps": 1,
        },
        "validation": {
            "enabled": True,
            "split_ratio": 0.15,
            "every_steps": 2,
            "every_epochs": None,
            "timesteps": [0.5],
        },
        "sampling": {
            "enabled": True,
            "every_steps": 2,
            "every_epochs": None,
            "at_start": True,
            "width": 64,
            "height": 64,
            "prompts": [{"prompt": "cat", "steps": 2}],
        },
    }


@pytest.mark.parametrize(
    "components,resolution_mode",
    [
        (None, "bucket"),
        ("backbone", "bucket"),
        ("text", "bucket"),
        ("both", "bucket"),
        (None, "native"),
        ("backbone", "native"),
    ],
)
def test_two_rank_training_artifacts_and_exact_resume(tmp_path, components, resolution_mode):
    _assert_two_rank_training(tmp_path, components, resolution_mode)


@pytest.mark.parametrize(
    "resolution_mode,image_fit,cached",
    [
        ("bucket", "crop", True),
        ("bucket", "pad", False),
        ("native", "pad", True),
    ],
)
def test_standard_lora_frozen_text_two_rank_exact_resume(tmp_path, resolution_mode, image_fit, cached):
    _assert_two_rank_training(
        tmp_path, None, resolution_mode, algorithm="lora", image_fit=image_fit, cached=cached
    )


def _assert_two_rank_training(
    tmp_path, components, resolution_mode, *, algorithm="lokr", image_fit="crop", cached=True
):
    training = (
        {"mode": "adapter"}
        if components is None
        else {
            "mode": "full",
            "train_backbone": components in {"backbone", "both"},
            "train_text_encoder": components in {"text", "both"},
        }
    )
    config = _config(tmp_path, training)
    config["dataset"]["resolution_mode"] = resolution_mode
    config["dataset"].update(image_fit=image_fit, cache_latents=cached)
    config["adapter"]["algo"] = algorithm
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    _launch(path)
    reference = tmp_path / "reference"
    events = [json.loads(line) for line in (reference / "events.jsonl").read_text().splitlines()]
    assert sum(event["type"] == "run.started" for event in events) == 1
    assert sum(event["type"] == "run.finished" for event in events) == 1
    assert sum(event["type"] == "sample.saved" for event in events) == 3
    assert sum(event["type"] == "validation" for event in events) == 2
    plan = next(event for event in events if event["type"] == "distributed.plan")
    assert plan["world_size"] == 2 and plan["backend"] == "gloo"
    rank_reports = [json.loads((reference / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert rank_reports[0]["pid"] != rank_reports[1]["pid"]
    assert rank_reports[0]["progress"] == rank_reports[1]["progress"]
    assert set(rank_reports[0]["samples"]).isdisjoint(rank_reports[1]["samples"])
    if components is None:
        assert all(r["text_frozen"] and r["frozen_unchanged"] and r["adapter_updated"] for r in rank_reports)
    replicas = [load_file(reference / f"rank-{rank}.safetensors") for rank in (0, 1)]
    for key in replicas[0]:
        torch.testing.assert_close(replicas[0][key], replicas[1][key], rtol=0, atol=0)
    rng = torch.load(reference / "state-2" / "rng.pt", weights_only=False)
    assert rng["distributed"]["world_size"] == 2
    assert not torch.equal(
        rng["distributed"]["ranks"][0]["generators"]["main"],
        rng["distributed"]["ranks"][1]["generators"]["main"],
    )
    config["checkpoint"].update(output_dir=str(tmp_path / "resumed"), resume=str(reference / "state-2"))
    path.write_text(json.dumps(config))
    _launch(path)
    filename = "training.safetensors" if components is None else "model.safetensors"
    expected = load_file(reference / "state-4" / filename)
    actual = load_file(tmp_path / "resumed" / "state-4" / filename)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    checkpoint_paths = [reference / "state-4", tmp_path / "resumed" / "state-4"]
    expected_meta, actual_meta = [
        json.loads((checkpoint / "state.json").read_text()) for checkpoint in checkpoint_paths
    ]
    for key in ("progress", "sampler"):
        assert_checkpoint_value_exact(actual_meta[key], expected_meta[key], key)
    assert expected_meta["sampler"]["position"] == expected_meta["progress"]["batch_in_epoch"]
    for component in ("rng", "optimizer", "scheduler"):
        expected_state, actual_state = [
            torch.load(checkpoint / f"{component}.pt", map_location="cpu", weights_only=False)
            for checkpoint in checkpoint_paths
        ]
        assert_checkpoint_value_exact(actual_state, expected_state, component)
    resumed_events = [
        json.loads(line) for line in (tmp_path / "resumed" / "events.jsonl").read_text().splitlines()
    ]
    expected_losses = {event["step"]: event["loss"] for event in events if event["type"] == "step"}
    for event in resumed_events:
        if event["type"] == "step":
            assert event["loss"] == expected_losses[event["step"]]
    reference_sample = next(
        event["path"] for event in events if event["type"] == "sample.saved" and event["step"] == 4
    )
    resumed_sample = next(event["path"] for event in resumed_events if event["type"] == "sample.saved")
    assert Path(reference_sample).read_bytes() == Path(resumed_sample).read_bytes()


def test_unsupported_compile_fails_both_ranks_without_hanging(tmp_path):
    config = _config(tmp_path, {"mode": "adapter"})
    config["memory"]["compile"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    result = _launch(path, expect_success=False)
    assert result.returncode != 0 and "DDP does not yet support" in result.stderr


@pytest.mark.parametrize("nonfinite", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_image_weighted_accumulation_matches_global_sgd(tmp_path, nonfinite, native):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=127.0.0.1:0",
            "--rdzv_conf=is_host=true",
            "--local_addr=127.0.0.1",
            "--nproc_per_node=2",
            "-m",
            "Test.tests.ddp_math_worker",
            str(tmp_path),
        ]
        + (["nonfinite"] if nonfinite else [])
        + (["native"] if native else []),
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "YPUDDIN_DDP_TIMEOUT_SECONDS": "60"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = [json.loads((tmp_path / f"math-rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert ranks[0] == ranks[1]


def test_pause_file_coordinates_all_ranks_and_resumes(tmp_path, monkeypatch):
    config = _config(tmp_path, {"mode": "adapter"})
    config["loop"]["log_every"] = 1
    path = tmp_path / "pause.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("YPUDDIN_TEST_PAUSE_STEP", "1")
    _launch(path)
    reference = tmp_path / "reference"
    assert (reference / "state-paused" / "state.json").exists()
    assert not (reference / "ddp-final.safetensors").exists()
    ranks = [json.loads((reference / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert ranks[0]["progress"] == ranks[1]["progress"]
    assert ranks[0]["progress"]["step"] == 1
    monkeypatch.delenv("YPUDDIN_TEST_PAUSE_STEP")
    config["checkpoint"].update(output_dir=str(tmp_path / "resumed"), resume=str(reference / "state-paused"))
    path.write_text(json.dumps(config))
    _launch(path)
    assert (tmp_path / "resumed" / "ddp-final.safetensors").exists()
