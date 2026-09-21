"""Full product preparation/train/export/sample/cold-resume under real CPU FSDP2."""

import json

import pytest
import torch
from safetensors.torch import load_file

from Test.scripts.verify_frozen_adapter_resume import compare
from Test.tests.e2e.test_distributed_training import _config, _launch


@pytest.mark.parametrize(
    "algorithm,options",
    [
        ("lora", {}),
        ("lokr", {}),
        ("lora", {"init": "scalar", "dropout": 0.1, "rank_dropout": 0.1}),
        ("lokr", {"dora": True, "mode": "merged"}),
    ],
)
def test_fsdp_adapter_product_cold_resume(tmp_path, algorithm, options):
    config = _config(tmp_path, {"mode": "adapter", "train_backbone": True, "train_text_encoder": False})
    config["loop"].update(gpu_count=2, distributed_strategy="fsdp", deterministic=True)
    config["adapter"].update(algo=algorithm, module_dropout=0, **options)
    config["memory"]["activation_checkpointing"] = "none"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    _launch(path, worker="Test.tests.fsdp_adapter_worker")
    reference = tmp_path / "reference"
    config["checkpoint"].update(output_dir=str(tmp_path / "resumed"), resume=str(reference / "state-2"))
    path.write_text(json.dumps(config))
    _launch(path, worker="Test.tests.fsdp_adapter_worker")
    result = compare(reference, tmp_path / "resumed", 4)
    assert result["weights_exact"] and result["states_exact"]
    artifacts = list(reference.glob("*.safetensors"))
    assert artifacts
    state = load_file(reference / "state-4/model.safetensors")
    assert all(".adapter." in key or ".dora." in key for key in state)
    first = load_file(reference / "state-2/model.safetensors")
    assert any(not torch.equal(first[k], v) for k, v in state.items())
