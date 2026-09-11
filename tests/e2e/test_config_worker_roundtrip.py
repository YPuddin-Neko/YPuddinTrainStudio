"""The service's TOML worker boundary must preserve explicitly disabled epoch hooks."""

import json
import subprocess
import sys

from PIL import Image

from ypuddin.config import TrainConfig, load_config, write_config


def test_cli_worker_preserves_null_epoch_hooks(tmp_path):
    training = tmp_path / "data"
    training.mkdir()
    for i, color in enumerate(("red", "green")):
        Image.new("RGB", (64, 64), color).save(training / f"{i}.png")
    validation = tmp_path / "validation"
    validation.mkdir()
    Image.new("RGB", (64, 64), "blue").save(validation / "held-out.png")
    output = tmp_path / "run"
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(training)}],
                "resolutions": [64],
                "bucket_step": 16,
                "num_workers": 0,
                "batch_size": 1,
            },
            "loop": {"epochs": None, "max_steps": 3, "mixed_precision": "no"},
            "checkpoint": {"output_dir": str(output), "save_every_epochs": None},
            "sampling": {
                "enabled": True,
                "at_start": True,
                "every_epochs": None,
                "every_steps": 3,
                "width": 64,
                "height": 64,
                "prompts": [{"prompt": "toy", "steps": 1}],
            },
            "validation": {
                "enabled": True,
                "sources": [{"path": str(validation)}],
                "split_ratio": 0,
                "every_epochs": None,
                "every_steps": 3,
                "timesteps": [0.5],
            },
        }
    )
    snapshot = tmp_path / "job-config.toml"
    write_config(cfg, snapshot)
    result = subprocess.run(
        [sys.executable, "-m", "ypuddin.cli", "train", str(snapshot), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert load_config(snapshot) == cfg
    assert load_config(output / "config.toml") == cfg
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    samples = [event for event in events if event["type"] == "sample.saved"]
    assert len(samples) == 2, samples  # initial + step3, with no implicit epoch previews
    assert all("epoch" not in event["path"] for event in samples)
    assert len([event for event in events if event["type"] == "validation"]) == 1
    assert list(output.glob("*-epoch*.safetensors")) == []
