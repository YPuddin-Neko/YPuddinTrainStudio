import pytest
import torch
from safetensors.torch import save_file

from ypuddin.config import ModelConfig, TrainConfig
from ypuddin.config.compute_policy import resolve_training_compute_config
from ypuddin.models.precision import checkpoint_precision, model_load_precision


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.float16, "fp16"),
        (torch.bfloat16, "bf16"),
        (torch.float32, "fp32"),
        (torch.float8_e4m3fn, "bf16"),
    ],
)
def test_reads_matrix_precision_without_loading_tensors(tmp_path, dtype, expected):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros((8, 8), dtype=dtype), "scale": torch.ones(1)}, path)
    assert checkpoint_precision(str(path)) == expected
    cfg = ModelConfig(dit_path=str(path))
    assert cfg.dtype == "auto"
    assert model_load_precision(cfg, "cuda") == expected
    assert model_load_precision(cfg, "mps") == "fp32"
    assert model_load_precision(cfg, "cpu") == "fp32"
    cfg.dtype = "bf16"
    assert model_load_precision(cfg, "cuda") == "bf16"


def test_diffusers_precision_uses_backbone_not_vae(tmp_path):
    for component, dtype in [("transformer", torch.bfloat16), ("vae", torch.float32)]:
        folder = tmp_path / component
        folder.mkdir()
        save_file({"weight": torch.zeros((8, 8), dtype=dtype)}, folder / "model.safetensors")
    assert checkpoint_precision(str(tmp_path)) == "bf16"


def test_follow_model_is_resolved_in_effective_config_before_snapshot(tmp_path):
    path = tmp_path / "fp16.safetensors"
    save_file({"weight": torch.zeros((8, 8), dtype=torch.float16)}, path)
    cfg = TrainConfig(model=ModelConfig(dit_path=str(path)))
    resolved, _ = resolve_training_compute_config(cfg, "cuda", "linux-cuda")
    assert resolved.model.dtype == "fp16"
    assert cfg.model.dtype == "auto"
    again, _ = resolve_training_compute_config(resolved, "cuda", "linux-cuda")
    assert again.to_dict() == resolved.to_dict()


@pytest.mark.parametrize(
    "profile,allowed",
    [
        ("linux-cuda", ["auto", "sdpa", "xformers", "flash_attn"]),
        ("windows-cuda", ["auto", "sdpa", "xformers", "flash_attn"]),
        ("linux-dtk", ["auto", "sdpa", "xformers", "flash_attn"]),
        ("macos-mps", ["auto", "sdpa", "metal_flash"]),
        ("linux-cpu", ["auto", "sdpa"]),
    ],
)
def test_family_attention_options_follow_launch_profile(monkeypatch, profile, allowed):
    from ypuddin import runtime_profiles
    from ypuddin.server import routes_core

    monkeypatch.setattr(runtime_profiles, "current_profile", lambda: profile)
    source = {"attention_backends": ["auto", "sdpa", "xformers", "flash_attn", "metal_flash"]}
    monkeypatch.setattr(routes_core, "family_info", lambda _name: source)
    info = routes_core.runtime_family_info("anima")
    assert info["attention_backends"] == allowed
    assert (
        info["runtime_backend"]
        == {
            "linux-cuda": "cuda",
            "windows-cuda": "cuda",
            "linux-dtk": "hip",
            "macos-mps": "mps",
            "linux-cpu": "cpu",
        }[profile]
    )
    assert source["attention_backends"][-1] == "metal_flash"


def test_directory_config_precision_fallback_without_unpickling_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"torch_dtype":"float16"}')
    assert checkpoint_precision(str(tmp_path)) == "fp16"
