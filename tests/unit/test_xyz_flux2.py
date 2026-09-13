"""API-created Klein comparisons keep the loader's swap contract on real networks."""

import json

import torch
from fastapi.testclient import TestClient
from PIL import Image

from tests.unit.test_flux2_family import tiny_root  # noqa: F401
from ypuddin.config import TrainConfig
from ypuddin.server import create_app
from ypuddin.server.db import now
from ypuddin.server.xyz_worker import generate
from ypuddin.train import Trainer


def test_api_klein_swap_checkpoint_survives_real_xyz_load(tiny_root, tmp_path):  # noqa: F811
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = tmp_path / "data"
        data.mkdir()
        Image.new("RGB", (64, 64), (130, 60, 30)).save(data / "cat.png")
        (data / "cat.txt").write_text("a cat")
        cfg = TrainConfig.model_validate(
            {
                "model": {
                    "family": "flux2",
                    "dit_path": str(tiny_root),
                    "dtype": "fp32",
                    "flux2_variant": "klein-base-4b",
                },
                "dataset": {
                    "sources": [{"path": str(data)}],
                    "resolutions": [64],
                    "num_workers": 0,
                    "text_encoding": "cached",
                },
                "adapter": {"algo": "lora", "rank": 2, "alpha": 2, "preset": "attn-only"},
                "memory": {"activation_checkpointing": "block", "blocks_to_swap": 3},
                "loop": {"max_steps": 1, "mixed_precision": "no"},
                "sampling": {"enabled": False},
                "checkpoint": {"output_dir": str(tmp_path / "train"), "save_on_finish": True},
                "logging": {"tensorboard": False},
            }
        )
        trainer = Trainer(cfg, device="cpu")
        assert trainer.run() == "finished"
        checkpoint = next((tmp_path / "train").glob("*final.safetensors"))
        app = create_app(tmp_path / "studio")
        context = app.state.ctx
        context.db.set_kv("queue.settings", {"held": True, "max_concurrent": 1})
        context.db.insert(
            "jobs",
            {
                "id": "source",
                "type": "train",
                "name": "Klein",
                "status": "completed",
                "created_at": now(),
                "run_dir": str(checkpoint.parent),
                "config_json": json.dumps(cfg.to_dict()),
            },
        )
        context.db.insert(
            "artifacts",
            {
                "id": "weights",
                "job_id": "source",
                "kind": "weights",
                "name": checkpoint.name,
                "path": str(checkpoint),
                "step": 1,
                "created_at": now(),
            },
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs/source/xyz",
                json={
                    "prompt": "a cat",
                    "width": 64,
                    "height": 64,
                    "steps": 2,
                    "cfg": 1,
                    "checkpoint_id": "weights",
                    "x": {"key": "adapter_scale", "values": [0, 1]},
                },
            )
            assert response.status_code == 202, response.text
            row = context.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
            payload = json.loads(row["config_json"])
            assert payload["memory"]["activation_checkpointing"] == "block"
            assert payload["memory"]["blocks_to_swap"] == 3
            payload.update(device="cpu", fingerprint_cache=str(tmp_path / "fingerprints"))
            generate(payload, tmp_path / "xyz", lambda *args, **kwargs: None, lambda: False)
        manifest = json.loads((tmp_path / "xyz/manifest.json").read_text())
        assert manifest["complete"] and len(manifest["cells"]) == 2
        for cell in manifest["cells"]:
            with Image.open(tmp_path / "xyz" / cell["file"]) as image:
                assert image.size == (64, 64)
    finally:
        torch.set_num_threads(previous)
