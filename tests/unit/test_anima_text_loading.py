"""Real small Qwen3 decoder coverage for the single-file Anima loader."""

from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file

transformers = pytest.importorskip("transformers")

from ypuddin.models.anima.text import _load_qwen3  # noqa: E402


@pytest.fixture
def qwen_checkpoint(monkeypatch):
    config = transformers.Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    torch.manual_seed(123)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    state = {f"model.{k}": v.detach().clone() for k, v in model.model.state_dict().items()}
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: deepcopy(config))
    return config, state


@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("stored_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("requested_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_meta_loader_matches_initialized_decoder(
    tmp_path, qwen_checkpoint, bare, stored_dtype, requested_dtype
):
    config, state = qwen_checkpoint
    state = {k: v.to(stored_dtype) for k, v in state.items()}
    path = tmp_path / "encoder.safetensors"
    save_file({k.removeprefix("model.") if bare else k: v for k, v in state.items()}, path)

    reference = transformers.AutoModelForCausalLM.from_config(deepcopy(config), dtype=requested_dtype)
    reference.load_state_dict(state, strict=False)
    reference = reference.model.eval().requires_grad_(False)
    actual = _load_qwen3(path, requested_dtype, "cpu")

    assert not actual.training
    assert all(
        not p.requires_grad and p.dtype == requested_dtype and not p.is_meta for p in actual.parameters()
    )
    assert all(not b.is_meta for b in actual.buffers())
    assert actual.rotary_emb.inv_freq.dtype == torch.float32
    assert actual.rotary_emb.original_inv_freq.dtype == torch.float32
    tokens = torch.tensor([[1, 7, 2, 0], [1, 3, 4, 5]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    with torch.no_grad():
        expected = reference(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
        observed = actual(input_ids=tokens, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_complete_checkpoint_constructs_only_meta_parameters(tmp_path, qwen_checkpoint, monkeypatch):
    _, state = qwen_checkpoint
    path = tmp_path / "encoder.safetensors"
    save_file(state, path)
    original = transformers.AutoModelForCausalLM.from_config
    constructed = []

    def observe(*args, **kwargs):
        model = original(*args, **kwargs)
        constructed.append(all(p.is_meta for p in model.parameters()))
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", observe)
    _load_qwen3(path, torch.float32, "cpu")
    assert constructed == [True]


@pytest.mark.parametrize("bare", [False, True])
def test_tied_head_can_supply_embedding_alias(tmp_path, qwen_checkpoint, bare):
    _, state = qwen_checkpoint
    expected_embedding = state.pop("model.embed_tokens.weight")
    state["lm_head.weight"] = expected_embedding
    path = tmp_path / "head_alias.safetensors"
    save_file({k.removeprefix("model.") if bare else k: v for k, v in state.items()}, path)
    actual = _load_qwen3(path, torch.float32, "cpu")
    torch.testing.assert_close(actual.embed_tokens.weight, expected_embedding, rtol=0, atol=0)


def test_unused_head_does_not_overwrite_decoder_embedding(tmp_path, qwen_checkpoint):
    _, state = qwen_checkpoint
    state["lm_head.weight"] = torch.full_like(state["model.embed_tokens.weight"], 9.0)
    path = tmp_path / "separate_head.safetensors"
    save_file(state, path)
    actual = _load_qwen3(path, torch.float32, "cpu")
    torch.testing.assert_close(actual.embed_tokens.weight, state["model.embed_tokens.weight"], rtol=0, atol=0)


def test_untied_head_cannot_supply_missing_decoder_embedding(tmp_path, qwen_checkpoint):
    config, state = qwen_checkpoint
    config.tie_word_embeddings = False
    state["lm_head.weight"] = state.pop("model.embed_tokens.weight")
    path = tmp_path / "untied_head.safetensors"
    save_file(state, path)
    with pytest.raises(ValueError, match="missing required encoder weights: embed_tokens.weight"):
        _load_qwen3(path, torch.float32, "cpu")


@pytest.mark.parametrize("missing", ["model.norm.weight", "model.embed_tokens.weight"])
def test_missing_decoder_weight_fails_before_use(tmp_path, qwen_checkpoint, missing):
    _, state = qwen_checkpoint
    del state[missing]
    path = tmp_path / "incomplete.safetensors"
    save_file(state, path)
    with pytest.raises(
        ValueError, match="missing required encoder weights: .*" + missing.removeprefix("model.")
    ):
        _load_qwen3(path, torch.bfloat16, "cpu")


def test_unused_checkpoint_keys_remain_compatible(tmp_path, qwen_checkpoint, caplog):
    _, state = qwen_checkpoint
    state["unused.weight"] = torch.ones(1)
    path = tmp_path / "extra.safetensors"
    save_file(state, path)
    _load_qwen3(path, torch.bfloat16, "cpu")
    assert "1 unexpected keys" in caplog.text


def test_wrong_shape_remains_an_error(tmp_path, qwen_checkpoint):
    _, state = qwen_checkpoint
    state["model.norm.weight"] = torch.ones(31)
    path = tmp_path / "wrong_shape.safetensors"
    save_file(state, path)
    with pytest.raises(RuntimeError, match="size mismatch for norm.weight"):
        _load_qwen3(path, torch.bfloat16, "cpu")
