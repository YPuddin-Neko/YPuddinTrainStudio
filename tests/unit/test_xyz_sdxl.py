"""XYZ integrates real reduced SDXL UNet, dual CLIP, VAE and saved training adapters."""

import json

import pytest
import torch
from PIL import Image

from tests.unit.test_sdxl_family import tiny_pipeline  # noqa: F401
from ypuddin.config import TrainConfig
from ypuddin.server.xyz import XyzRequest, file_signature
from ypuddin.server.xyz_worker import generate
from ypuddin.train import Trainer


@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_saved_sdxl_adapter_drives_xyz_real_networks(tiny_pipeline, tmp_path, algo):  # noqa: F811
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = tmp_path / "images"
        data.mkdir()
        Image.new("RGB", (64, 64), (120, 55, 33)).save(data / "cat.png")
        (data / "cat.txt").write_text("a cat")
        cfg = TrainConfig.model_validate(
            {
                "model": {"family": "sdxl", "dit_path": str(tiny_pipeline), "dtype": "fp32"},
                "dataset": {
                    "sources": [{"path": str(data)}],
                    "resolutions": [64],
                    "batch_size": 1,
                    "num_workers": 0,
                    "text_encoding": "cached",
                },
                "adapter": {"algo": algo, "rank": 4, "alpha": 4, "preset": "attn-only"},
                "memory": {"activation_checkpointing": "block"},
                "objective": {"timestep_sampling": "uniform", "weighting": "none"},
                "loop": {"max_steps": 1, "mixed_precision": "no"},
                "sampling": {"enabled": False},
                "checkpoint": {"output_dir": str(tmp_path / "train"), "save_on_finish": True},
                "logging": {"tensorboard": False},
            }
        )
        trainer = Trainer(cfg, device="cpu")
        assert trainer.run() == "finished"
        checkpoint = next((tmp_path / "train").glob("*final.safetensors"))
        request = XyzRequest(
            prompt="a cat",
            negative="blurry",
            width=64,
            height=64,
            steps=2,
            cfg=2,
            sampler="euler",
            scheduler="uniform",
            checkpoint_id="trained",
            x={"key": "sampler", "values": ["euler", "heun"]},
            y={"key": "adapter_scale", "values": [0, 1]},
        )
        payload = {
            "model": cfg.model.model_dump(mode="json"),
            "memory": cfg.memory.model_dump(mode="json"),
            "device": "cpu",
            "fingerprint_cache": str(tmp_path / "fingerprints"),
            "xyz": {
                "request": request.model_dump(mode="json"),
                "checkpoints": {
                    "trained": {
                        "path": str(checkpoint),
                        "name": checkpoint.name,
                        "signature": file_signature(checkpoint),
                    }
                },
            },
        }
        generate(payload, tmp_path / "xyz", lambda *a, **k: None, lambda: False)
        manifest = json.loads((tmp_path / "xyz/manifest.json").read_text())
        assert manifest["complete"] and len(manifest["cells"]) == 4
        assert {cell["sampler"] for cell in manifest["cells"]} == {"euler", "heun"}
        assert all(Image.open(tmp_path / "xyz" / cell["file"]).size == (64, 64) for cell in manifest["cells"])
        assert all(cell["checkpoint_id"] == "trained" and cell["shift"] == 1 for cell in manifest["cells"])
        assert len(manifest["grids"]) == 1
    finally:
        torch.set_num_threads(previous)
