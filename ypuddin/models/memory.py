"""Release unloaded encoder weights at cache and sampling stage boundaries."""

import gc

import torch


def release_model_memory(device: torch.device) -> None:
    """Call after dropping model references, never from the per-batch hot path.

    Model callbacks can retain their module in a Python reference cycle. Clearing
    the allocator first cannot release those live tensors: collect the cycles
    before returning unused blocks to CUDA/HIP or MPS.
    """
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
