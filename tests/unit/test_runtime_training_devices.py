from types import SimpleNamespace

import pytest

from ypuddin.server.environment import runtime_info


@pytest.mark.parametrize(
    "cuda, count, distributed, nccl",
    [(True, 2, True, True), (False, 8, True, False), (False, 0, False, False)],
)
def test_runtime_distinguishes_pytorch_build_capabilities_from_trainer_support(
    monkeypatch, cuda, count, distributed, nccl
):
    import torch

    from ypuddin.server import hardware

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(
        torch,
        "distributed",
        SimpleNamespace(is_available=lambda: distributed, is_nccl_available=lambda: nccl),
    )
    monkeypatch.setattr(hardware, "gpu_info", lambda **kwargs: [{"name": "Driver-visible GPU"}] * 2)
    result = runtime_info()
    assert result["cuda_device_count"] == (count if cuda else 0)
    assert result["distributed_available"] == distributed
    assert result["nccl_available"] == nccl
    assert result["multi_gpu_training"] is False
    assert result["training_device_policy"] == "single_device"
    # Driver inventory must not turn CPU Torch into CUDA-capable Torch.
    assert len(result["gpus"]) == 2
    assert result["cuda_available"] is cuda
