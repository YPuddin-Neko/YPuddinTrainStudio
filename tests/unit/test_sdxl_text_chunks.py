"""Real dual CLIP chunk boundaries, differentiable conditioning and cache compatibility."""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from ypuddin.config import ModelConfig, TrainConfig, config_hash
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.sdxl.loading import ASSETS, config_asset
from ypuddin.models.sdxl.text import SDXLText


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def clip_weights(tmp_path_factory):
    root = tmp_path_factory.mktemp("sdxl-chunk-clips")
    torch.manual_seed(198)
    common = dict(vocab_size=49408, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77)
    CLIPTextModel(CLIPTextConfig(hidden_size=8, intermediate_size=16, **common)).save_pretrained(root / "l")
    CLIPTextModelWithProjection(
        CLIPTextConfig(hidden_size=12, intermediate_size=24, projection_dim=6, **common)
    ).save_pretrained(root / "g")
    return root


def make_text(root, length=75):
    return SDXLText(
        root / "l",
        root / "g",
        (ASSETS / "tokenizer", ASSETS / "tokenizer_2"),
        device="cpu",
        dtype=torch.float32,
        max_token_length=length,
    )


@pytest.mark.parametrize("length", [150, 225])
def test_real_tokenizers_content_boundaries_empty_padding_and_truncation(clip_weights, length):
    text = make_text(clip_weights, length)
    text._ensure()
    counts = [0, 1, 74, 75, 76, 149, 150, 151, 224, 225, 226]
    captions = ["cat " * count for count in counts]
    for tokenizer in text.tokenizers:
        word = tokenizer("cat", add_special_tokens=False).input_ids
        assert len(word) == 1  # make the token boundary, rather than word count, explicit
        chunks = text._tokenize(tokenizer, captions).reshape(len(captions), length // 75, 77)
        expected = []
        for count in counts:
            caption_chunks = []
            for start in range(0, length, 75):
                content_count = min(75, max(0, min(length, count) - start))
                caption_chunks.append(
                    [tokenizer.bos_token_id]
                    + word * content_count
                    + [tokenizer.eos_token_id]
                    + [tokenizer.pad_token_id] * (75 - content_count)
                )
            expected.append(caption_chunks)
        assert torch.equal(chunks, torch.tensor(expected))
    assert text.tokenizers[0].pad_token_id == text.tokenizers[0].eos_token_id
    assert text.tokenizers[1].pad_token_id == 0 != text.tokenizers[1].eos_token_id


@pytest.mark.parametrize("length", [150, 225])
def test_clip_g_real_zero_token_survives_chunk_start_and_end_in_mixed_batch(clip_weights, length):
    text = make_text(clip_weights, length)
    text._ensure()
    tokenizer = text.tokenizers[1]
    assert tokenizer("!", add_special_tokens=False).input_ids == [tokenizer.pad_token_id] == [0]
    captions = ["", "!", "cat", "! " * (length + 1)]
    for start in range(0, length, 75):
        captions.extend(["cat " * start + "! " + "dog " * (length - start), "cat " * (start + 74) + "! dog"])
    chunks = text._tokenize(tokenizer, captions).reshape(len(captions), length // 75, 77)
    for caption, actual in zip(captions, chunks, strict=True):
        # Derive padding from the length of the unpadded content, independently
        # of token values. A real zero token must survive at either boundary.
        content = tokenizer(caption, add_special_tokens=False).input_ids[:length]
        expected = []
        for start in range(0, length, 75):
            part = content[start : start + 75]
            expected.append(
                [tokenizer.bos_token_id]
                + part
                + [tokenizer.eos_token_id]
                + [tokenizer.pad_token_id] * (75 - len(part))
            )
        assert torch.equal(actual, torch.tensor(expected)), caption


def test_default_tokenization_keeps_original_75_path_with_real_zero_content(clip_weights):
    text = make_text(clip_weights)
    text._ensure()
    captions = ["", "!", "! " * 76, "cat " * 74 + "! dog", "cat"]
    for tokenizer in text.tokenizers:
        original = tokenizer(
            captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        assert torch.equal(text._tokenize(tokenizer, captions), original)


@pytest.mark.parametrize("length", [75, 150, 225])
def test_dual_hidden_stitch_first_chunk_pooled_and_cache_roundtrip(clip_weights, length):
    text = make_text(clip_weights, length)
    captions = ["cat " * 76 + "dog " * 160, ""]
    actual = text.encode(captions, "cpu")
    expected_hidden = []
    for tokenizer, model in zip(text.tokenizers, text.models, strict=True):
        ids = text._tokenize(tokenizer, captions)
        with torch.no_grad():
            output = model(ids, output_hidden_states=True)
        h = output.hidden_states[-2].reshape(2, length // 75, 77, -1)
        if length == 75:
            expected_hidden.append(h[:, 0])
        else:
            pieces = [h[:, 0, :1]] + [h[:, i, 1:76] for i in range(length // 75)] + [h[:, -1, -1:]]
            expected_hidden.append(torch.cat(pieces, dim=1))
        if hasattr(output, "text_embeds"):
            expected_pool = output.text_embeds.reshape(2, length // 75, -1)[:, 0]
    torch.testing.assert_close(actual["embeds"], torch.cat(expected_hidden, dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(actual["pooled"], expected_pool, rtol=0, atol=0)
    assert actual["embeds"].shape == (2, length + 2, 20)
    assert actual["pooled"].shape == (2, 6)
    assert actual["embeds"][1, -1].abs().sum() > 0
    assert not actual["embeds"].requires_grad
    cache = text.encode_for_cache(captions)
    text.unload()
    for restored in (text.cond_from_cache(cache, "cpu"), text.encode(captions, "cpu")):
        for key in actual.tensors:
            torch.testing.assert_close(actual[key], restored[key], rtol=0, atol=0)


@pytest.mark.parametrize("length", [150, 225])
def test_late_chunk_tokens_change_conditioning_but_not_first_chunk_pool(clip_weights, length):
    text = make_text(clip_weights, length)
    prefix = "cat " * (length - 1)
    cond = text.encode([prefix + "dog", prefix + "bird"], "cpu")
    # The last supported token contributes in both CLIPs; early chunks and pooled
    # CLIP-G stay unchanged. Nothing beyond the selected budget is encoded.
    assert not torch.equal(cond["embeds"][0, length, :8], cond["embeds"][1, length, :8])
    assert not torch.equal(cond["embeds"][0, length, 8:], cond["embeds"][1, length, 8:])
    torch.testing.assert_close(cond["embeds"][0, :76], cond["embeds"][1, :76], rtol=0, atol=0)
    torch.testing.assert_close(cond["pooled"][0], cond["pooled"][1], rtol=0, atol=0)
    truncated = text.encode([prefix + "dog cat", prefix + "dog bird"], "cpu")
    for value in truncated.tensors.values():
        torch.testing.assert_close(value[0], value[1], rtol=0, atol=0)


@pytest.mark.parametrize("length", [150, 225])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_online_gradients_reach_both_clips_last_chunk_and_projected_pool(clip_weights, length, checkpointed):
    text = make_text(clip_weights, length)
    text.training_enabled = True
    for model in text.trainable_modules().values():
        model.requires_grad_(True).train()
        if checkpointed:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    caption = "cat " * (length - 1) + "dog"
    cond = text.encode([caption], "cpu")
    assert all(t.requires_grad for t in cond.tensors.values())
    loss = cond["embeds"][:, -2].square().sum() + cond["pooled"].square().sum()
    loss.backward()
    for tokenizer, model in zip(text.tokenizers, text.models, strict=True):
        dog = tokenizer("dog", add_special_tokens=False).input_ids[0]
        clip = getattr(model, "text_model", model)  # Transformers 4 and 5 module layouts
        grad = clip.embeddings.token_embedding.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad[dog].abs().sum() > 0
        assert clip.encoder.layers[0].self_attn.k_proj.weight.grad.abs().sum() > 0
    assert text.models[1].text_projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        assert not text.encode([caption], "cpu")["embeds"].requires_grad


def test_default_cache_identity_unchanged_long_lengths_isolated(clip_weights):
    texts = [make_text(clip_weights, length) for length in (75, 150, 225)]
    default = texts[0]
    legacy_identity = content_fingerprint(
        [
            *default.paths,
            *default.tokenizer_paths,
            *(config_asset(p, c) for p, c in zip(default.paths, default.components, strict=True)),
        ],
        namespace="sdxl-dual-clip-penultimate-pooled-v1:dtype=torch.float32",
    )
    assert default.fingerprint == legacy_identity
    assert len({text.fingerprint for text in texts}) == 3
    for source in texts:
        cache = source.encode_for_cache(["cat"])
        for target in texts:
            if source is target:
                continue
            with pytest.raises(ValueError, match="incompatible CLIP conditioning dimensions"):
                target.cond_from_cache(cache, "cpu")


def test_token_budget_defaults_preserve_legacy_config_hash_without_mutation():
    cfg = TrainConfig(model=ModelConfig(family="sdxl"))
    legacy = cfg.to_dict()
    del legacy["model"]["sdxl_max_token_length"]
    restored = TrainConfig.model_validate(legacy)
    assert restored.model.sdxl_max_token_length == 75
    before = copy.deepcopy(legacy)
    assert config_hash(cfg) == config_hash(restored) == config_hash(legacy)
    assert legacy == before
    for length in (150, 225):
        changed = cfg.model_copy(deep=True)
        changed.model.sdxl_max_token_length = length
        assert config_hash(changed) != config_hash(legacy)
    for invalid in (0, 76, 152, 300):
        with pytest.raises(ValueError, match="sdxl_max_token_length"):
            ModelConfig(family="sdxl", sdxl_max_token_length=invalid)
