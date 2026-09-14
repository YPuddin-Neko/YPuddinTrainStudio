"""A fresh process catches import-order bugs hidden by an already warmed pytest interpreter."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("family", ["anima", "sdxl", "flux2", "krea2", "toy", "krea2_text", "all"])
def test_cold_meta_planning_keeps_lazy_libraries_importable_and_parameters_on_meta(family):
    script = r'''
import sys, torch
from ypuddin.config import ModelConfig
from ypuddin.models import get_family

family = sys.argv[1]
original_parameter = torch.nn.Module.register_parameter
observed = []
def guard_parameter(self, name, value):
    if value is not None and value.numel() > 10000:
        assert value.is_meta, "planner allocated model parameters outside meta"
        observed.append(value.numel())
    return original_parameter(self, name, value)
torch.nn.Module.register_parameter = guard_parameter

if family == "all":
    from ypuddin.models.registry import available
    from ypuddin.server.routes_core import family_info
    # The real /families route orders Anima first and suppresses layer errors.
    # Guard each call so a poisoned import cannot be hidden by an empty list.
    families = available()
    assert families == ["anima", "flux2", "krea2", "sdxl", "toy"]
    for name in families:
        instance = get_family(name)
        original = instance.linear_module_names
        def guard(original=original):
            names = original()
            assert names
            return names
        instance.linear_module_names = guard
        info = family_info(name)
        assert any(preset["layers"] > 0 for preset in info["presets"]), name
elif family == "krea2_text":
    from ypuddin.models.training_parameters import text_parameter_count
    with torch.device("meta"):
        count = text_parameter_count(get_family("krea2"), ModelConfig(family="krea2"))
    assert count > 1000000 and observed
else:
    instance = get_family(family)
    original = instance.meta_backbone
    def guard(config):
        model = original(config)
        assert all(parameter.is_meta for parameter in model.parameters())
        return model
    instance.meta_backbone = guard
    assert instance.linear_module_names()

# A family-info request must not leave Dynamo partially initialized before
# the next Krea memory plan imports its text architecture.
from transformers import Qwen3VLTextModel
assert callable(Qwen3VLTextModel)
assert torch.empty(()).device.type == "cpu", "meta context escaped its scope"
print("COLD_META_IMPORT_OK")
'''
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", script, family], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COLD_META_IMPORT_OK" in result.stdout
