"""Real AdapterSet / Trainer warm starts must not silently ignore an entire weight file."""

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from ypuddin.config import TrainConfig
from ypuddin.train import Trainer
from ypuddin.train.events import Emitter


@pytest.mark.parametrize("selection", ["none", "one", "all"])
def test_trainer_reports_actual_warm_start_coverage(tmp_path, selection):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 32), "red").save(image_dir / "red.png")
    (image_dir / "red.txt").write_text("a red image")
    config = TrainConfig.model_validate(
        {
            "model": {"family": "toy", "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(image_dir)}],
                "resolutions": [32],
                "bucket_step": 16,
                "num_workers": 0,
            },
            "adapter": {"algo": "lora", "rank": 2, "alpha": 2, "preset": "attn-only"},
            "loop": {"mixed_precision": "no", "max_steps": 1},
            "sampling": {"enabled": False},
            "checkpoint": {"output_dir": str(tmp_path / "source")},
        }
    )
    source = Trainer(config, device="cpu")
    warm = None
    try:
        source.prepare()
        initial = {key: tensor.clone() for key, tensor in source.adapters.export_state()[0].items()}
        with torch.no_grad():
            for layer in source.adapters.layers.values():
                for parameter in layer.adapter.parameters():
                    parameter.fill_(0.125)
        exported = source.adapters.export_state()[0]
        modules = {key.partition(".")[0] for key in exported}
        assert len(modules) > 1
        chosen = min(modules)
        if selection == "none":
            weights = {"different_family_" + key: value for key, value in exported.items()}
        elif selection == "one":
            weights = {key: value for key, value in exported.items() if key.startswith(chosen + ".")}
        else:
            weights = exported
        path = tmp_path / "warm-start.safetensors"
        save_file(weights, path)
        resumed = config.model_copy(deep=True)
        resumed.checkpoint.output_dir = str(tmp_path / "warm")
        resumed.adapter.resume_weights = str(path)
        events = []
        warm = Trainer(resumed, device="cpu", emitter=Emitter(listeners=[events.append]))
        if selection == "none":
            with pytest.raises(ValueError, match="matched no adapted layers"):
                warm.prepare()
            assert not any(event["type"] == "adapters.injected" for event in events)
            return
        warm.prepare()
        actual = warm.adapters.export_state()[0]
        for key, value in actual.items():
            expected = weights.get(key, initial[key])
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        warnings = [event for event in events if event.get("code") == "adapter.partial_warm_start"]
        if selection == "one":
            assert len(warnings) == 1
            assert warnings[0]["matched_layers"] == 1
            assert warnings[0]["missing_layers"] == len(modules) - 1
            assert f"1/{len(modules)}" in warnings[0]["message"]
        else:
            assert not warnings
        assert any(event["type"] == "adapters.injected" for event in events)
    finally:
        for trainer in (source, warm):
            if trainer is not None:
                if trainer._logs is not None:
                    trainer._logs.close()
                trainer.emitter.close()
