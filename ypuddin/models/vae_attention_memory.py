"""Allocator preparation for large, changing HIP VAE attention workspaces."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)
_GIB = 1 << 30


class VaeMathMemoryGuard:
    def __init__(self) -> None:
        self._previous: tuple[torch.device, int] | None = None

    def prepare(self, q: torch.Tensor, k: torch.Tensor) -> None:
        # Math SDPA uses FP32 intermediates for half inputs. Include its three
        # score buffers and mask; reserved capacity is not live tensor storage.
        scores = q.shape[0] * q.shape[1] * q.shape[-2] * k.shape[-2]
        workspace = scores * (3 * max(4, q.element_size()) + 1)
        current = (q.device, workspace)
        previous, self._previous = self._previous, current
        if workspace < _GIB or (previous is not None and previous[0] == q.device and workspace <= previous[1]):
            return

        free, _ = torch.cuda.mem_get_info(q.device)
        if free >= workspace:
            return
        stats = torch.cuda.memory_stats(q.device)
        reserved = stats.get("reserved_bytes.all.current")
        allocated = stats.get("allocated_bytes.all.current")
        if reserved is None or allocated is None:
            return
        unused = max(0, reserved - allocated)
        # Both figures are bounds: the workspace may be smaller on vendor builds,
        # and split cache blocks may not all be releasable. A partial reclaim can
        # still help; avoid synchronizing just to return a negligible cache.
        if unused < min(_GIB, workspace - free):
            return
        # empty_cache can release this process's pools on more than one device.
        # Keep normal reuse for equal/smaller workspaces instead of clearing on
        # every image. Live tensors and the subsequent SDPA call stay unchanged.
        with torch.cuda.device(q.device):
            torch.cuda.empty_cache()
        log.debug(
            "VAE attention cache release on %s: workspace %.2f GiB, free %.2f GiB, unused cache %.2f GiB",
            q.device, workspace / _GIB, free / _GIB, unused / _GIB,
        )
