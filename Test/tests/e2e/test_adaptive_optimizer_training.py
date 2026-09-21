"""Synthetic CPU training verifies the optimizer/preview/checkpoint integration."""

import json
from pathlib import Path

import pytest
import torch

from ypuddin.config import TrainConfig
from ypuddin.optim import is_schedule_free
from ypuddin.train import Trainer


@pytest.mark.parametrize(
    "kind,param_dtype,schedule_free",
    [
        ("automagic", "fp32", False),
        ("automagic", "bf16", False),
        ("prodigy", "fp32", False),
        ("prodigy_plus_sf", "fp32", True),
        ("prodigy_plus_sf", "fp32", False),
    ],
)
def test_adaptive_optimizer_accumulation_preview_and_resume(
    image_dataset, tmp_path, kind, param_dtype, schedule_free
):
    if kind == "prodigy":
        pytest.importorskip("prodigyopt")
    elif kind == "prodigy_plus_sf":
        pytest.importorskip("prodigyplus")
    optimizer = {"type": kind, "lr": 1e-6 if kind == "automagic" else 1.0, "grad_clip_norm": 0}
    if kind == "prodigy_plus_sf":
        # Legacy args must retain their meaning through typed-field migration.
        optimizer["args"] = {"use_schedulefree": schedule_free}
    config = TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolutions": [64],
                "bucket_step": 16,
                "batch_size": 2,
                "num_workers": 0,
            },
            "adapter": {
                "algo": "lokr",
                "rank": 4,
                "alpha": 4,
                "preset": "attn-mlp",
                "param_dtype": param_dtype,
            },
            "optimizer": optimizer,
            "scheduler": {
                "type": "constant" if kind == "automagic" or schedule_free else "cosine",
                "warmup_steps": 0,
            },
            "loop": {"max_steps": 4, "grad_accum": 2, "mixed_precision": "no", "seed": 7, "log_every": 1},
            "checkpoint": {
                "output_dir": str(tmp_path / "reference"),
                "name": "adaptive",
                "save_every_epochs": None,
                "save_state_every_steps": 2,
            },
            "validation": {"enabled": False},
            "sampling": {
                "enabled": True,
                "every_steps": 2,
                "every_epochs": None,
                "width": 64,
                "height": 64,
                "prompts": [{"prompt": "test image", "width": 64, "height": 64, "steps": 2}],
            },
        }
    )
    reference = Trainer(config, device="cpu")
    assert reference.run() == "finished"
    assert is_schedule_free(config.optimizer) == schedule_free
    assert (reference.scheduler is None) == (kind == "automagic" or schedule_free)
    config = config.model_copy(deep=True)
    config.checkpoint.output_dir = str(tmp_path / "resumed")
    config.checkpoint.resume = str(reference.run_dir / "state-2")
    resumed = Trainer(config, device="cpu")
    assert resumed.run() == "finished"
    for key, expected in reference.adapters.training_state_dict().items():
        torch.testing.assert_close(resumed.adapters.training_state_dict()[key], expected, rtol=0, atol=0)
    from Test.tests.checkpoint_assertions import assert_checkpoint_value_exact

    assert_checkpoint_value_exact(
        resumed.optimizer.state_dict(), reference.optimizer.state_dict(), "optimizer"
    )

    def events(run_dir):
        return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]

    records = events(reference.run_dir)
    steps = [event for event in records if event["type"] == "step"]
    assert [event["step"] for event in steps] == [1, 2, 3, 4]
    assert all(0 < lr < 1 for event in steps for lr in event["lr"].values())
    original_preview = [event for event in records if event["type"] == "sample.saved"][-1]
    resumed_preview = [event for event in events(resumed.run_dir) if event["type"] == "sample.saved"][-1]
    assert Path(original_preview["path"]).read_bytes() == Path(resumed_preview["path"]).read_bytes()
