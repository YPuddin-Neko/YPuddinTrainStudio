"""`ypuddin smoke` on the toy family: the same path a GPU machine uses to validate Anima."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ypuddin.cli import main


@pytest.mark.parametrize("outcome", ["stopped", "paused"])
@pytest.mark.parametrize("completed_steps", [0, 1])
def test_interrupted_smoke_writes_partial_report_without_sampling_or_exporting(
    tmp_path, monkeypatch, capsys, outcome, completed_steps
):
    instances = []

    class InterruptedTrainer:
        def __init__(self, cfg, *, device, emitter):
            self.device = device
            self.emitter = emitter
            self.event_file = emitter._file
            self.loaded = SimpleNamespace(extra={})
            self.family = SimpleNamespace(spec=SimpleNamespace(name="toy", adapter_prefix="lora_unet"))
            self.adapters = SimpleNamespace(
                layers=["layer"], summary=lambda: {"trainable_params": 1, "by_algo": {"lora": 1}}
            )
            self.progress = SimpleNamespace(total_steps=cfg.loop.max_steps)
            self.prepare = Mock()
            self.sample_images = Mock(side_effect=AssertionError("sampling continued after interruption"))
            self.save_weights = Mock(side_effect=AssertionError("export continued after interruption"))
            instances.append(self)

        def run(self):
            for step in range(completed_steps):
                self.emitter.emit("step", step=step + 1, loss=0.5, grad_norm=0.2, it_s=2)
            return outcome

    monkeypatch.setattr("ypuddin.train.Trainer", InterruptedTrainer)
    out = tmp_path / "interrupted-smoke"
    rc = main(["smoke", "--set", "model.family=toy", "--device", "cpu", "--out", str(out)])
    report = json.loads((out / "smoke-report.json").read_text(encoding="utf-8"))
    assert rc == 1 and report["ok"] is False and report["outcome"] == outcome
    assert report["losses"] == [0.5] * completed_steps
    assert report["total_steps"] == 3
    assert {"prepare", "train"} <= report["timings_s"].keys()
    assert "sample" not in report["timings_s"] and "sample" not in report and "export" not in report
    assert "error" not in report and "traceback" not in report
    checks = {check["name"]: check["ok"] for check in report["checks"]}
    assert checks["training finished"] is False
    assert "sample saved" not in checks and "adapter file round-trips" not in checks
    trainer = instances[0]
    trainer.sample_images.assert_not_called()
    trainer.save_weights.assert_not_called()
    assert trainer.event_file.closed and trainer.emitter._file is None
    assert f"outcome={outcome}" in capsys.readouterr().out


def test_smoke_recipe_keeps_selected_sampling_algorithm(tmp_path):
    from ypuddin.config import TrainConfig
    from ypuddin.tools.smoke import smoke_config

    cfg = TrainConfig.model_validate(
        {"sampling": {"sampler": "er_sde", "scheduler": "simple", "er_sde_order": 2, "er_sde_s_noise": 0.2}}
    )
    tiny = smoke_config(cfg, out=tmp_path / "run", steps=1, resolution=64, sample_size=64, sample_steps=2)
    for name in ("sampler", "scheduler", "er_sde_order", "er_sde_s_noise"):
        assert getattr(tiny.sampling, name) == getattr(cfg.sampling, name)
    assert tiny.sampling.prompts[0].steps == 2


@pytest.mark.parametrize("algo", ["lokr", "lora", "loha", "full"])
def test_smoke_cli_toy_family(tmp_path, capsys, algo):
    out = tmp_path / "smoke"
    rc = main(
        [
            "smoke",
            "--set",
            "model.family=toy",
            "--set",
            "model.dtype=fp32",
            "--set",
            f"adapter.algo={algo}",
            "--set",
            "adapter.rank=4",
            "--set",
            "adapter.alpha=4",
            "--set",
            "loop.mixed_precision=no",
            "--set",
            "dataset.bucket_step=16",
            "--out",
            str(out),
            "--steps",
            "2",
            "--resolution",
            "64",
            "--sample-size",
            "64",
            "--sample-steps",
            "2",
            "--device",
            "cpu",
        ]
    )
    text = capsys.readouterr().out
    assert rc == 0, text
    report = json.loads((out / "smoke-report.json").read_text(encoding="utf-8"))
    assert report["ok"] and report["family"] == "toy" and report["outcome"] == "finished"
    names = {c["name"]: c["ok"] for c in report["checks"]}
    assert names["training finished"] and names["sample saved"] and names["adapter file round-trips"]
    assert len(report["losses"]) == 2 and (out / "data").exists()  # dataset was synthesised
    assert "PASS" in text and "smoke-report.json" in text
    assert report["export"]["tensors"] > 0 and all(
        k.startswith("lora_unet_") for k in report["export"]["example_keys"]
    )
