"""Cold-process CPU acceptance of real reduced SDXL networks, not DTK kernels."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from tests.checkpoint_assertions import assert_checkpoint_value_exact
from tests.e2e.test_distributed_training import _launch
from tests.unit.test_sdxl_family import tiny_pipeline as sdxl_assets  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "algorithm,length,world_size,cached",
    [
        ("lora", 75, 1, True),
        ("lora", 150, 1, True),
        ("lora", 225, 1, True),
        ("lokr", 150, 1, True),
        ("lokr", 225, 1, True),
        ("lora", 150, 2, False),
    ],
)
def test_frozen_dual_clip_cold_resume(request, tmp_path, algorithm, length, world_size, cached):
    assets_root = request.getfixturevalue("sdxl_assets")
    images = tmp_path / "images"
    images.mkdir()
    caption = "cat " * (length - 1) + "dog"
    for i, size in enumerate([(64, 64), (96, 64), (64, 96), (80, 64)] * 2):
        Image.new("RGB", size, (30 + 20 * i, 40, 100)).save(images / f"{i}.png")
        (images / f"{i}.txt").write_text(caption)
        mask = Image.new("L", size, 255)
        mask.paste(0, (0, 0, 8, size[1]))
        mask.save(images / f"{i}.mask.png")
    cfg = {
        "model": {
            "family": "sdxl",
            "dit_path": str(assets_root),
            "dtype": "fp32",
            "sdxl_max_token_length": length,
        },
        "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": False},
        "adapter": {"algo": algorithm, "rank": 2, "alpha": 2, "preset": "attn-only"},
        "dataset": {
            "sources": [{"path": str(images)}],
            "resolutions": [64],
            "bucket_step": 16,
            "batch_size": 1,
            "num_workers": 0,
            "image_fit": "pad",
            "bucket_no_upscale": True,
            "cache_latents": cached,
            "cache_dir": str(tmp_path / "cache"),
            "text_encoding": "cached",
            "masked_loss": True,
        },
        "objective": {"timestep_sampling": "uniform", "weighting": "none"},
        "memory": {"activation_checkpointing": "none", "offload_text_encoder": False},
        "loop": {"max_steps": 4, "mixed_precision": "no", "seed": 19, "deterministic": True},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "sampling": {
            "enabled": True,
            "every_steps": 2,
            "every_epochs": None,
            "width": 64,
            "height": 64,
            "prompts": [{"prompt": caption, "steps": 2, "cfg": 1}],
        },
        "checkpoint": {
            "output_dir": str(tmp_path / "reference"),
            "name": "main",
            "save_state_every_steps": 2,
            "save_every_epochs": None,
            "save_dtype": "fp32",
        },
    }
    config_path = tmp_path / "config.json"

    def launch():
        config_path.write_text(json.dumps(cfg))
        if world_size == 2:
            _launch(config_path)
        else:
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(k, None)
            result = subprocess.run(
                [sys.executable, "-m", "ypuddin.cli", "train", str(config_path), "--device", "cpu"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, result.stdout + result.stderr

    launch()
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    cfg["checkpoint"].update(output_dir=str(resumed), resume=str(reference / "state-2"))
    launch()
    a, b = [load_file(p / "state-4/training.safetensors") for p in (reference, resumed)]
    assert a and a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    earlier = load_file(reference / "state-2/training.safetensors")
    assert any(not torch.equal(a[k], earlier[k]) for k in a)
    for component in ("optimizer", "rng", "scheduler"):
        expected, actual = [
            torch.load(p / f"state-4/{component}.pt", map_location="cpu", weights_only=False)
            for p in (reference, resumed)
        ]
        assert_checkpoint_value_exact(actual, expected, component)
    meta = [json.loads((p / "state-4/state.json").read_text()) for p in (reference, resumed)]
    for field in ("progress", "sampler"):
        assert_checkpoint_value_exact(meta[1][field], meta[0][field], field)
    events = [
        [json.loads(s) for s in (p / "events.jsonl").read_text().splitlines()] for p in (reference, resumed)
    ]
    previews = [
        next(Path(e["path"]) for e in es if e["type"] == "sample.saved" and e["step"] == 4) for es in events
    ]
    assert previews[0].read_bytes() == previews[1].read_bytes()
    if world_size == 2:
        for directory in (reference, resumed):
            reports = [json.loads((directory / f"rank-{i}.json").read_text()) for i in (0, 1)]
            assert all(r["text_frozen"] and r["frozen_unchanged"] and r["adapter_updated"] for r in reports)
            assert set(reports[0]["samples"]).isdisjoint(reports[1]["samples"])
