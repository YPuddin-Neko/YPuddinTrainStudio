"""CPU/meta checks for actionable Klein extension failures; no vendor wheels or GPU required."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("diffusers")

from diffusers import Flux2Transformer2DModel
from diffusers.models import attention_dispatch

from ypuddin.config import MemoryConfig, ModelConfig
from ypuddin.models.flux2 import family as flux2

TINY_CONFIG = dict(
    in_channels=128,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=24,
    timestep_guidance_channels=8,
    axes_dims_rope=(2, 2, 2, 2),
    mlp_ratio=2,
    guidance_embeds=False,
)


@pytest.fixture
def model(monkeypatch):
    registry = attention_dispatch._AttentionBackendRegistry
    monkeypatch.setattr(registry, "_active_backend", registry._active_backend)
    with torch.device("meta"):
        return Flux2Transformer2DModel(**TINY_CONFIG)


def vendor_flash_without_public_interface(monkeypatch):
    # Reproduce the actual vendor wheel's usable public function but absent
    # public deterministic entrypoint required by the model-local DTK path.
    monkeypatch.setattr(torch.version, "hip", "6.3.26093")
    monkeypatch.setattr(attention_dispatch, "_CAN_USE_FLASH_ATTN", False)
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_func=lambda: None))


def test_dtk_flash_missing_interface_gives_sdpa_action_and_retains_cause(monkeypatch, model):
    vendor_flash_without_public_interface(monkeypatch)
    with pytest.raises(RuntimeError, match="未满足接口或运行时要求") as caught:
        flux2._set_attention_backend(model, "flash_attn", "cuda:0")
    message = str(caught.value)
    assert "SDPA" in message and "厂商构建" in message and "Klein" in message
    assert "pip install" not in message
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "deterministic backward" in str(caught.value.__cause__)
    assert all(parameter.is_meta for parameter in model.parameters())


@pytest.mark.parametrize("attention", ["auto", "sdpa"])
def test_dtk_native_selection_survives_incompatible_optional_flash(monkeypatch, model, attention):
    vendor_flash_without_public_interface(monkeypatch)
    flux2._set_attention_backend(model, attention, "cpu")
    active, _ = attention_dispatch._AttentionBackendRegistry.get_active_backend()
    assert active == attention_dispatch.AttentionBackendName.NATIVE


def test_non_dtk_flash_error_does_not_claim_vendor_interface_failure(monkeypatch, model):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(attention_dispatch, "_CAN_USE_FLASH_ATTN", False)
    with pytest.raises(RuntimeError, match="Select SDPA") as caught:
        flux2._set_attention_backend(model, "flash_attn", "cuda:0")
    assert "DTK" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_cpu_explicit_flash_fails_before_backend_selection(model):
    with pytest.raises(ValueError, match="CUDA / HIP GPU.*Select SDPA"):
        flux2._set_attention_backend(model, "flash_attn", "cpu")


def test_native_runtime_error_is_preserved_without_relabeling():
    original = RuntimeError("unrelated native backend failure")
    model = SimpleNamespace(set_attention_backend=Mock(side_effect=original))
    with pytest.raises(RuntimeError) as caught:
        flux2._set_attention_backend(model, "sdpa", "cpu")
    assert caught.value is original


def test_load_rejects_incompatible_flash_before_cache_or_weight_materialization(monkeypatch, tmp_path):
    vendor_flash_without_public_interface(monkeypatch)
    family = flux2.Flux2Family()
    monkeypatch.setattr(family, "validate_config", lambda _: [])
    monkeypatch.setattr(family, "_paths", lambda _: (tmp_path,) * 5)
    monkeypatch.setattr(flux2, "transformer_config", lambda _: TINY_CONFIG)
    monkeypatch.setattr(flux2, "resolve_variant", lambda *args: "klein-base-4b")
    text = SimpleNamespace(hidden_size=24, weight_elements=1, encode=Mock(), encode_for_cache=Mock())
    monkeypatch.setattr(flux2, "Flux2Text", lambda *args, **kwargs: text)
    latent = Mock()
    load_weights = Mock()
    monkeypatch.setattr(flux2, "Flux2Latent", latent)
    monkeypatch.setattr(flux2, "load_transformer", load_weights)
    with pytest.raises(RuntimeError, match="未满足接口或运行时要求"):
        family.load(
            ModelConfig(family="flux2", dit_path=str(tmp_path), attention="flash_attn"),
            MemoryConfig(),
            device="cuda:0",
            dtype=torch.float16,
        )
    text.encode.assert_not_called()
    text.encode_for_cache.assert_not_called()
    latent.assert_not_called()
    load_weights.assert_not_called()
