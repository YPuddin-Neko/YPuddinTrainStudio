"""Preserve the task's RNG when isolating a formerly all-visible GPU worker."""

from unittest.mock import Mock

import pytest
import torch

from ypuddin.train.state import capture_rng, restore_rng


@pytest.fixture
def cuda_states(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    snapshot = capture_rng()
    saved = [torch.tensor([11], dtype=torch.uint8), torch.tensor([29], dtype=torch.uint8)]
    snapshot["cuda"] = saved
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: saved)
    set_all, set_one = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_one)
    monkeypatch.delenv("YPUDDIN_LEGACY_CUDA_RNG_INDEX", raising=False)
    return snapshot, set_all, set_one


def test_capture_records_actual_training_device_instead_of_default(cuda_states):
    snapshot = capture_rng(device=torch.device("cuda:1"))
    assert snapshot["cuda_device"] == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_isolated_worker_restores_original_card_rng(cuda_states, monkeypatch, legacy):
    snapshot, set_all, set_one = cuda_states
    if legacy:
        monkeypatch.setenv("YPUDDIN_LEGACY_CUDA_RNG_INDEX", "1")
    else:
        snapshot["cuda_device"] = 1
    restore_rng(snapshot, device=torch.device("cuda:0"))
    set_all.assert_not_called()
    assert set_one.call_args.args[0] is snapshot["cuda"][1]
    assert set_one.call_args.kwargs == {"device": 0}


def test_ambiguous_legacy_state_is_rejected_before_rng_changes(cuda_states):
    snapshot, set_all, set_one = cuda_states
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="原任务恢复"):
        restore_rng(snapshot, device=torch.device("cuda:0"))
    assert torch.equal(before, torch.get_rng_state())
    set_all.assert_not_called()
    set_one.assert_not_called()


def test_new_isolated_state_keeps_exact_rng_and_ignores_legacy_hint(cuda_states, monkeypatch):
    snapshot, set_all, set_one = cuda_states
    snapshot["cuda"] = snapshot["cuda"][:1]
    snapshot["cuda_device"] = 0
    monkeypatch.setenv("YPUDDIN_LEGACY_CUDA_RNG_INDEX", "1")
    restore_rng(snapshot, device=torch.device("cuda:0"))
    assert set_all.call_args.args[0] is snapshot["cuda"]
    set_one.assert_not_called()
