"""The UI must validate the same versioned policy payload the server emits."""

import json
from pathlib import Path

import pytest

from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import resolve_training_compute_config

CASES = json.loads((Path(__file__).parents[2] / "frontend/fixtures/trainingComputePolicies.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda row: row["name"])
def test_frontend_compute_contract_matches_backend(case):
    config = TrainConfig.model_validate(case["config"])
    _, policy = resolve_training_compute_config(config, "cuda", "linux-dtk")
    assert policy == case["policy"]
