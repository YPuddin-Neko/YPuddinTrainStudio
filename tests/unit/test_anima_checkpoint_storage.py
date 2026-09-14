"""Unsupported prequantized Anima files must fail before scales can be discarded."""

import json

import pytest
import torch
from safetensors.torch import save_file

from ypuddin.config import ModelConfig
from ypuddin.models.anima.checkpoint import check_unquantized_checkpoint
from ypuddin.models.anima.family import AnimaFamily, _read_state_dict, load_dit
from ypuddin.models.anima.text import _load_qwen3


@pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."])
@pytest.mark.parametrize("kind", ["fp8", "weight_scale", "scale_weight", "scaled_fp8"])
def test_prequantized_dit_fails_header_preflight_and_loader_before_reading_weights(tmp_path, monkeypatch, prefix, kind):
    state = {prefix + "linear.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    if kind == "fp8":
        state[prefix + "linear.weight"] = state[prefix + "linear.weight"].to(torch.float8_e4m3fn)
    else:
        key = prefix + ("scaled_fp8" if kind == "scaled_fp8" else f"linear.{kind}")
        state[key] = torch.tensor(3.)
    path = tmp_path / "anima.safetensors"
    save_file(state, str(path))
    before = path.read_bytes()
    monkeypatch.setattr("safetensors.torch.load_file", lambda *a, **kw: pytest.fail("payload was loaded before rejection"))
    cfg = ModelConfig(family="anima", dit_path=str(path), text_encoder_path=str(tmp_path), vae_path=str(tmp_path))
    problems = AnimaFamily().validate_config(cfg)
    assert len(problems) == 1
    assert "model.dit_path" in problems[0] and "暂不支持直接加载现成 FP8" in problems[0]
    with pytest.raises(ValueError, match="暂不支持直接加载现成 FP8"):
        load_dit(path, device="cpu", dtype=torch.bfloat16)
    assert path.read_bytes() == before


@pytest.mark.parametrize("directory", [False, True])
def test_qwen_prequantized_storage_rejected_in_validation_and_direct_load(tmp_path, directory):
    file = tmp_path / "model.safetensors"
    save_file({"model.linear.weight": torch.ones(2, 2, dtype=torch.float8_e4m3fn),
               "model.linear.weight_scale": torch.tensor(0.5)}, str(file))
    path = tmp_path if directory else file
    cfg = ModelConfig(family="anima", dit_path=str(tmp_path / "empty"), text_encoder_path=str(path), vae_path=str(tmp_path))
    problems = AnimaFamily().validate_config(cfg)
    assert any("model.text_encoder_path" in problem and "Anima 文字编码器" in problem for problem in problems)
    with pytest.raises(ValueError, match="Anima 文字编码器.*FP8"):
        _load_qwen3(path, torch.bfloat16, "cpu")


def test_qwen_checks_every_selected_shard_and_rejects_escaping_shards(tmp_path):
    save_file({"part.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(tmp_path / "first.safetensors"))
    save_file({"part.weight": torch.ones(2, 2, dtype=torch.float8_e5m2)}, str(tmp_path / "second.safetensors"))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}))
    with pytest.raises(ValueError, match="FP8"):
        check_unquantized_checkpoint(tmp_path, "Anima 文字编码器")
    index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="目录之外"):
        check_unquantized_checkpoint(tmp_path, "Anima 文字编码器")


def test_real_precision_and_normalization_scales_are_preserved_and_unrelated_files_ignored(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"linear.weight": torch.tensor([[1., 2.]], dtype=torch.bfloat16), "norm.scale": torch.tensor(2.)}, str(path))
    save_file({"linear.weight": torch.ones(2, 2, dtype=torch.float8_e4m3fn)}, str(tmp_path / "unrelated.safetensors"))
    check_unquantized_checkpoint(tmp_path, "Anima 文字编码器")
    state = _read_state_dict(path, torch.float32)
    torch.testing.assert_close(state["linear.weight"], torch.tensor([[1., 2.]]))
    assert state["norm.scale"].item() == 2
