"""Real Toy sampling retains optimizer loss even when metric logging is sparse."""

import json
from pathlib import Path

import pytest

from ypuddin.config import TrainConfig
from ypuddin.train import Trainer


def config(data, output):
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(data)}],
                "resolutions": [64],
                "bucket_step": 16,
                "batch_size": 2,
                "num_workers": 0,
            },
            "adapter": {"rank": 4, "alpha": 4},
            "loop": {
                "epochs": None,
                "max_steps": 3,
                "grad_accum": 2,
                "mixed_precision": "no",
                "log_every": 100,
            },
            "checkpoint": {"output_dir": str(output), "save_every_epochs": None, "save_state_every_steps": 2},
            "sampling": {
                "enabled": True,
                "at_start": True,
                "every_steps": 1,
                "every_epochs": None,
                "width": 64,
                "height": 64,
                "steps": 1,
                "prompts": [{"prompt": "a cat"}],
            },
        }
    )


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sparse_logging_samples_have_real_optimizer_loss_and_initial_has_none(image_dataset, tmp_path):
    trainer = Trainer(config(image_dataset, tmp_path / "run"), device="cpu")
    actual = {}
    optimizer_step = trainer._optimizer_step

    def observe(group_loss, elapsed):
        before = trainer.progress.step
        optimizer_step(group_loss, elapsed)
        if trainer.progress.step > before:
            actual[trainer.progress.step] = group_loss

    trainer._optimizer_step = observe
    assert trainer.run() == "finished"
    saved = events(trainer.run_dir / "events.jsonl")
    assert [event["step"] for event in saved if event["type"] == "step"] == [3]
    samples = [event for event in saved if event["type"] == "sample.saved"]
    assert [event["step"] for event in samples] == [0, 1, 2, 3]
    assert samples[0]["loss"] is None
    for event in samples[1:]:
        assert event["loss"] == actual[event["step"]]
        assert Path(event["path"]).is_file()
    checkpoint = json.loads((trainer.run_dir / "state-2/state.json").read_text())
    assert checkpoint["progress"]["extra"]["train_loss"] == {"step": 2, "loss": actual[2]}


@pytest.mark.parametrize("stored", ["matching", "absent", "wrong-step", "nonfinite", "overflow"])
def test_resume_sample_loss_accepts_only_exact_checkpoint_record(image_dataset, tmp_path, stored):
    cfg = config(image_dataset, tmp_path / "first")
    cfg.sampling.enabled = False
    original = Trainer(cfg, device="cpu")
    assert original.run() == "finished"
    state_path = original.run_dir / "state-2/state.json"
    state = json.loads(state_path.read_text())
    expected = state["progress"]["extra"]["train_loss"]["loss"]
    if stored == "absent":
        del state["progress"]["extra"]["train_loss"]
    elif stored == "wrong-step":
        state["progress"]["extra"]["train_loss"]["step"] = 1
    elif stored == "nonfinite":
        state["progress"]["extra"]["train_loss"]["loss"] = float("nan")
    elif stored == "overflow":
        state["progress"]["extra"]["train_loss"]["loss"] = 10**400
    state_path.write_text(json.dumps(state))
    cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg.checkpoint.resume = str(state_path.parent)
    cfg.loop.max_steps = 3
    resumed = Trainer(cfg, device="cpu")
    try:
        resumed.prepare()
        resumed.sample_images("before-next-step")
        before = [
            event for event in events(resumed.run_dir / "events.jsonl") if event["type"] == "sample.saved"
        ]
        assert before[0]["step"] == 2
        assert before[0]["loss"] == (expected if stored == "matching" else None)
        # A checkpoint made before this feature still gets a fresh exact loss after one step.
        resumed.cfg.sampling.enabled = True
        assert resumed.run() == "finished"
        after = [
            event for event in events(resumed.run_dir / "events.jsonl") if event["type"] == "sample.saved"
        ]
        assert after[-1]["step"] == 3
        assert after[-1]["loss"] == resumed.progress.extra["train_loss"]["loss"]
        assert after[-1]["loss"] is not None
    finally:
        resumed.emitter.close()
        resumed._close_logs()
