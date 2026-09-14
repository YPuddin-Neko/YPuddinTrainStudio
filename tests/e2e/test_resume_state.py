"""Resume actual augmented training across epochs and compare every saved state."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file

from tests.checkpoint_assertions import assert_checkpoint_value_exact
from tests.e2e.test_toy_training import _cfg, _events
from ypuddin.train import Trainer


class BatchAuditTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_inputs = {}

    def compute_loss(self, batch, *args, **kwargs):
        key = self.progress.epoch, self.progress.batch_in_epoch, tuple(batch["index"].tolist())
        self.batch_inputs[key] = (
            tuple(batch["caption"]),
            hashlib.sha256(batch["pixels"].numpy().tobytes()).hexdigest(),
        )
        return super().compute_loss(batch, *args, **kwargs)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("resolution_mode", ["bucket", "native"])
def test_every_checkpoint_component_exact_after_mid_epoch_resume(
    image_dataset, tmp_path, workers, resolution_mode
):
    cfg = _cfg(
        image_dataset,
        tmp_path / "reference",
        dataset={
            "num_workers": workers,
            "resolution_mode": resolution_mode,
            "cache_latents": False,
            "text_encoding": "online",
            "flip": True,
            "caption": {"shuffle": True, "tag_dropout": 0.2, "caption_dropout": 0.1},
        },
        adapter={"module_dropout": 0.2},
        loop={"epochs": None, "max_steps": 5, "grad_accum": 2, "deterministic": True},
        checkpoint={"save_every_epochs": None, "save_state_every_steps": 1},
        validation={"enabled": False},
        sampling={"enabled": False},
    )
    reference = BatchAuditTrainer(cfg, device="cpu")
    assert reference.run() == "finished"
    checkpoint = tmp_path / "reference" / "state-2"
    meta = json.loads((checkpoint / "state.json").read_text())
    assert meta["progress"]["extra"]["deterministic"] is True
    assert 0 < meta["progress"]["batch_in_epoch"] < len(reference.sampler.plan(0))
    cfg2 = cfg.model_copy(deep=True)
    cfg2.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg2.checkpoint.resume = str(checkpoint)
    resumed = BatchAuditTrainer(cfg2, device="cpu")
    assert resumed.run() == "finished"
    assert resumed.progress.epoch >= 1
    assert resumed.batch_inputs
    for key, value in resumed.batch_inputs.items():
        assert value == reference.batch_inputs[key], key
    expected_losses = {
        event["step"]: event["loss"]
        for event in _events(tmp_path / "reference" / "events.jsonl")
        if event["type"] == "step"
    }
    for event in _events(tmp_path / "resumed" / "events.jsonl"):
        if event["type"] == "step":
            assert event["loss"] == expected_losses[event["step"]]
    for step in (3, 4, 5):
        paths = [tmp_path / phase / f"state-{step}" for phase in ("reference", "resumed")]
        expected, actual = [json.loads((path / "state.json").read_text()) for path in paths]
        for key in ("progress", "sampler"):
            assert_checkpoint_value_exact(actual[key], expected[key], key)
        assert expected["sampler"]["epoch"] == expected["progress"]["epoch"]
        assert expected["sampler"]["position"] == expected["progress"]["batch_in_epoch"]
        for component in ("rng", "optimizer", "scheduler"):
            values = [
                torch.load(path / f"{component}.pt", map_location="cpu", weights_only=False)
                for path in paths
            ]
            assert_checkpoint_value_exact(values[1], values[0], component)
        values = [load_file(path / "training.safetensors") for path in paths]
        assert_checkpoint_value_exact(values[1], values[0], "weights")


def test_legacy_checkpoint_can_resume_with_explicit_historical_policy(image_dataset, tmp_path):
    cfg = _cfg(
        image_dataset,
        tmp_path / "legacy",
        loop={"max_steps": 3, "epochs": None, "deterministic": False},
        checkpoint={"save_every_epochs": None, "save_state_every_steps": 1},
        validation={"enabled": False},
        sampling={"enabled": False},
    )
    assert Trainer(cfg, device="cpu").run() == "finished"
    checkpoint = tmp_path / "legacy" / "state-1"
    state_path = checkpoint / "state.json"
    meta = json.loads(state_path.read_text())
    del meta["progress"]["extra"]["deterministic"]
    state_path.write_text(json.dumps(meta))
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.resume = str(checkpoint)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    expected = load_file(str(tmp_path / "legacy" / "state-3" / "training.safetensors"))
    actual = load_file(str(tmp_path / "resumed" / "state-3" / "training.safetensors"))
    assert_checkpoint_value_exact(actual, expected, "legacy_weights")
