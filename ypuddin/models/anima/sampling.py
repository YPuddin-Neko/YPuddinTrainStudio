"""Anima inference conditioning and memory-bounded CFG batches."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from ypuddin.models.base import TextCond


class SamplingContext(TextCond):
    """LLM-adapter output prepared for one sampling invocation."""


def _is_sharded(model) -> bool:
    if not hasattr(model, "_get_fsdp_state"):
        return False
    from torch.distributed.fsdp import FSDPModule

    return isinstance(model, FSDPModule)


@contextmanager
def _preparation_forward(model):
    if not _is_sharded(model):
        yield model._preprocess_text_embeds
        return
    from torch.distributed.fsdp import register_fsdp_forward_method

    name = "_preprocess_text_embeds"
    previous = model.__dict__.get(name)
    register_fsdp_forward_method(model, name)
    try:
        yield getattr(model, name)
    finally:
        if previous is None:
            delattr(model, name)
        else:
            setattr(model, name, previous)


def _shard_reserve(model) -> int:
    """Allow root weights and two prefetched child groups to be gathered."""
    from torch.distributed.tensor import DTensor

    groups = {}

    def visit(module, owner):
        if _is_sharded(module):
            owner = id(module)
        for p in module.parameters(recurse=False):
            if isinstance(p, DTensor):
                missing = max(0, p.numel() - p.to_local().numel()) * p.element_size()
                groups[owner] = groups.get(owner, 0) + missing
        for child in module.children():
            visit(child, owner)

    visit(model, id(model))
    root = groups.pop(id(model), 0)
    return root + sum(sorted(groups.values(), reverse=True)[:2])


def _free_memory(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        stats = torch.cuda.memory_stats(device)
        reusable = stats.get("reserved_bytes.all.current", 0) - stats.get("active_bytes.all.current", 0)
        return int(free + max(0, reusable))
    if device.type == "mps":
        return max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory())
    import psutil

    return int(psutil.virtual_memory().available)


def _memory_required(shape, loaded, dtype: torch.dtype) -> float:
    # ComfyUI's Anima activation estimate; model weights are already resident.
    width = loaded.extra["dit_config"]["model_channels"]
    factor = width / 2048 * 0.95 * (1.4 if dtype == torch.float16 else 1.0)
    mode = getattr(loaded.backbone, "attn_mode", "torch")
    efficient = (
        loaded.device.type == "cuda"
        and not getattr(torch.version, "hip", None)
        and dtype in (torch.float16, torch.bfloat16)
        and (
            mode in ("xformers", "flash_attn", "sage", "sageattn")
            or (
                mode in ("torch", "sdpa", None)
                and torch.backends.cuda.flash_sdp_enabled()
                and torch.cuda.get_device_capability(loaded.device)[0] >= 8
            )
        )
    )
    per_pixel = torch.empty((), dtype=dtype).element_size() * 0.01 if efficient else 0.15
    return shape[0] * math.prod(shape[2:]) * per_pixel * factor * 1024**2


class AnimaPredictionPair:
    """Return positive/negative predictions, executing compatible contexts negative-first."""

    def __init__(self, loaded, forward, cond, uncond, *, reserve_memory=0, check=None):
        from .vendor.attention import sampling_attention

        self.loaded, self.forward = loaded, forward
        self.sharded = _is_sharded(loaded.backbone)
        self.reserve_memory = reserve_memory + (_shard_reserve(loaded.backbone) if self.sharded else 0)
        dtype = loaded.dtype if loaded.device.type != "cpu" else torch.float32

        def prepare(condition):
            if check is not None:
                check()
            with sampling_attention(True), _preparation_forward(loaded.backbone) as preprocess:
                context = preprocess(
                    condition["embeds"].to(device=loaded.device, dtype=dtype),
                    condition["t5_ids"].to(loaded.device),
                    condition["t5_mask"].to(loaded.device),
                    condition["attn_mask"].to(loaded.device),
                )
            return SamplingContext({"embeds": context})

        # Keep the adapter's batch/precision and masks separate from DiT CFG batching.
        self.positive, self.negative = prepare(cond), prepare(uncond)

    def __call__(self, x, t):
        positive, negative = self.positive["embeds"], self.negative["embeds"]
        compatible = (
            positive.shape == negative.shape
            and positive.device == negative.device
            and positive.dtype == negative.dtype
        )
        shape = (2 * x.shape[0], *x.shape[1:])
        available = max(0, _free_memory(x.device) - self.reserve_memory)
        fits = _memory_required(shape, self.loaded, positive.dtype) * 1.5 < available
        decision = 0 if not compatible else (2 if fits else 1)
        if self.sharded:
            import torch.distributed as dist

            # Every rank must execute the same number and order of FSDP forwards.
            common = torch.tensor(decision, device=x.device, dtype=torch.int32)
            dist.all_reduce(common, op=dist.ReduceOp.MIN)
            decision = int(common.item())
        if decision == 0:
            return self.forward(x, t, self.positive), self.forward(x, t, self.negative)
        if decision == 2:
            context = SamplingContext({"embeds": torch.cat((negative, positive))})
            result = self.forward(torch.cat((x, x)), torch.cat((t, t)), context)
            neg, pos = result.chunk(2)
            return pos, neg
        negative_result = self.forward(x, t, self.negative)
        return self.forward(x, t, self.positive), negative_result
