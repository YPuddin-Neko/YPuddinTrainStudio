"""Frozen encoding arithmetic, cache identity, and exception cleanup contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ypuddin.config.compute_policy import KLEIN9B_TEXT_COMPUTE_ID
from ypuddin.data.cache import TextCache
from ypuddin.models.flux2.text import Flux2Text
from ypuddin.models.flux2.text_compute import frozen_text_compute


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_native_rounding_and_frozen_parameters_are_preserved(bias, noncontiguous):
    torch.manual_seed(2)
    model = nn.Linear(7, 5, bias=bias).to(torch.bfloat16).requires_grad_(False)
    x = torch.randn(3, 4, 7).to(torch.bfloat16)
    if noncontiguous:
        x = x.transpose(0, 1)
    parameters = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        expected = torch.nn.functional.linear(
            x.float(), model.weight.float(), model.bias.float() if bias else None
        ).to(torch.bfloat16)
        if noncontiguous and bias:
            # Native noncontiguous Linear rounds matmul before its separate bias add.
            expected = (x.float() @ model.weight.float().T).to(torch.bfloat16) + model.bias
        with frozen_text_compute(model, KLEIN9B_TEXT_COMPUTE_ID):
            actual = model(x)
        assert torch.equal(actual, expected) and actual.dtype == torch.bfloat16
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in parameters.items())
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_operator_scope_restored_after_exception():
    model = nn.Linear(3, 2).to(torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        native = model(torch.ones(1, 3, dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match="abort"):
            with frozen_text_compute(model, KLEIN9B_TEXT_COMPUTE_ID):
                model(torch.ones(1, 3, dtype=torch.bfloat16))
                raise RuntimeError("abort")
        assert torch.equal(native, model(torch.ones(1, 3, dtype=torch.bfloat16)))
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_policy_cannot_enable_text_training_or_unknown_math():
    model = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="frozen"):
        with frozen_text_compute(model, KLEIN9B_TEXT_COMPUTE_ID):
            pass
    with torch.no_grad(), pytest.raises(ValueError, match="frozen"):
        with frozen_text_compute(model, KLEIN9B_TEXT_COMPUTE_ID):
            pass
    with pytest.raises(ValueError, match="Unsupported"):
        with frozen_text_compute(model, "unknown"):
            pass


def pipeline(**changes):
    values = dict(
        variant="klein-base-9b",
        dtype=torch.bfloat16,
        model=None,
        compute_implementation=None,
        fingerprint="actual-weight-hash",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_native_and_stable_cache_keys_cannot_overlap():
    text = pipeline()
    old = TextCache.key("caption", text.fingerprint)
    Flux2Text.configure_compute(text, KLEIN9B_TEXT_COMPUTE_ID)
    assert TextCache.key("caption", text.fingerprint) != old
    assert text.compute_implementation == KLEIN9B_TEXT_COMPUTE_ID
    with pytest.raises(ValueError):
        Flux2Text.configure_compute(text, KLEIN9B_TEXT_COMPUTE_ID)


@pytest.mark.parametrize(
    "changes",
    [
        dict(variant="klein-base-4b"),
        dict(dtype=torch.float32),
        dict(model=object()),
        dict(training_enabled=True),
    ],
)
def test_invalid_cache_policy_is_rejected_without_mutation(changes):
    text = pipeline(**changes)
    with pytest.raises(ValueError):
        Flux2Text.configure_compute(text, KLEIN9B_TEXT_COMPUTE_ID)
    assert text.fingerprint == "actual-weight-hash" and text.compute_implementation is None
