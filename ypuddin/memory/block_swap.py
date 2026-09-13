"""Block swapping: keep the frozen weights of the last N transformer blocks in (pinned) host memory
and stream them to the accelerator just in time, on both the forward and the backward pass.

Design notes (lessons from AnimaLoraStudio / sd-scripts / diffusion-pipe):
* Tensors are moved by rebinding ``param.data`` in place — module objects are never replaced, so
  adapter wrappers, hooks and optimizer references stay valid.
* Trainable parameters are never swapped (their gradients must live on the device).
* Hooks are installed for forward-pre, forward-post, backward-pre and backward-post. Forward-only
  hooks silently corrupt gradients when activation checkpointing recomputes a block.
* On CUDA copies happen on a dedicated stream with events; elsewhere they are synchronous.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)


def _swappable_tensors(block: nn.Module) -> list[Tensor]:
    """Frozen parameters and buffers of a block (adapter parameters with grad are excluded)."""
    out: list[Tensor] = []
    for p in block.parameters():
        if not p.requires_grad:
            out.append(p)
    for b in block.buffers():
        out.append(b)
    return out


def _make_host_masters(tensors: list[Tensor], *, pin_memory: bool) -> list[Tensor]:
    """Pack a block's frozen tensors into one independent CPU pool per dtype.

    PyTorch rounds each pinned allocation up to a power of two. Grouping a block
    avoids rounding every projection separately, without casting FP8 weights or
    their scales. The final pools are allocated directly; blocking copies make
    CUDA sources ready before rebinding. Views are contiguous, retain each input's
    shape and values (including strided inputs), and never overlap.
    """
    groups: dict[torch.dtype, list[tuple[int, Tensor]]] = {}
    for index, tensor in enumerate(tensors):
        groups.setdefault(tensor.dtype, []).append((index, tensor))
    masters: dict[int, Tensor] = {}
    for dtype, group in groups.items():
        pool = torch.empty(sum(t.numel() for _, t in group), dtype=dtype, device="cpu", pin_memory=pin_memory)
        offset = 0
        for index, tensor in group:
            count = tensor.numel()
            host = pool.narrow(0, offset, count).view(tensor.shape)
            host.copy_(tensor.detach())
            masters[index] = host
            offset += count
    return [masters[index] for index in range(len(tensors))]


class BlockSwapper:
    def __init__(
        self,
        blocks: Iterable[nn.Module],
        num_swap: int,
        device: torch.device | str,
        *,
        pin_memory: bool = True,
        prefetch: bool = True,
    ):
        self.blocks = list(blocks)
        self.device = torch.device(device)
        n = min(int(num_swap), len(self.blocks))
        self.swapped_idx = list(range(len(self.blocks) - n, len(self.blocks)))
        self.pin = bool(pin_memory) and self.device.type == "cuda"
        self.prefetch = prefetch and self.device.type == "cuda"
        self._masters: dict[int, list[Tensor]] = {}  # block index -> host copies (same order as _tensors)
        self._tensors: dict[int, list[Tensor]] = {}
        self._resident: set[int] = set()
        self._deferred: set[int] = set()
        self._events: dict[int, torch.cuda.Event] = {}
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.forward_only = False
        had_cuda_sources = False
        for i in self.swapped_idx:
            tensors = _swappable_tensors(self.blocks[i])
            had_cuda_sources |= any(t.device.type == "cuda" for t in tensors)
            masters = _make_host_masters(tensors, pin_memory=self.pin)
            for t, host in zip(tensors, masters, strict=True):
                t.data = host  # block lives on the host until fetched
            self._tensors[i] = tensors
            self._masters[i] = masters
        if had_cuda_sources and self.device.type == "cuda":
            # Host copies above are blocking. Release the now-unused CUDA
            # allocations from temporary full-model staging before training;
            # live tensors and stream-dependent allocations remain protected
            # by PyTorch's allocator. CPU-loaded models do not need this flush.
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.bytes_per_block = (
            sum(t.numel() * t.element_size() for t in self._masters[self.swapped_idx[0]])
            if self.swapped_idx
            else 0
        )
        self._install()

    # ----------------------------------------------------------------- moves
    def _fetch(self, i: int) -> None:
        if i not in self._tensors or i in self._resident:
            return
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                for t, host in zip(self._tensors[i], self._masters[i], strict=True):
                    t.data = host.to(self.device, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
                self._events[i] = ev
        else:
            for t, host in zip(self._tensors[i], self._masters[i], strict=True):
                t.data = host.to(self.device)
        self._resident.add(i)

    def _wait(self, i: int) -> None:
        ev = self._events.pop(i, None)
        if ev is not None:
            torch.cuda.current_stream(self.device).wait_event(ev)

    def _release(self, i: int) -> None:
        if i not in self._resident:
            return
        if self._stream is not None:
            # make sure compute on the current stream finished before the device copies are dropped
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
        for t, host in zip(self._tensors[i], self._masters[i], strict=True):
            t.data = host
        self._resident.discard(i)

    def ensure(self, i: int) -> None:
        self._fetch(i)
        self._wait(i)

    # ----------------------------------------------------------------- hooks
    def _install(self) -> None:
        for i in self.swapped_idx:
            block = self.blocks[i]
            self._handles.append(block.register_forward_pre_hook(self._make_fwd_pre(i)))
            self._handles.append(block.register_forward_hook(self._make_fwd_post(i)))
            self._handles.append(block.register_full_backward_pre_hook(self._make_bwd_pre(i)))
            self._handles.append(block.register_full_backward_hook(self._make_bwd_post(i)))

    def _make_fwd_pre(self, i: int):
        def hook(_m, _inputs):
            self.ensure(i)
            if self.prefetch and i + 1 in self._tensors:
                self._fetch(i + 1)

        return hook

    def _make_fwd_post(self, i: int):
        def hook(_m, _inputs, _output):
            if self.forward_only or not torch.is_grad_enabled():
                self._release(i)
            # during training the block stays resident until its backward finished (or is evicted
            # by the next fetch when memory is tight — see release_all_but)
            elif len(self._resident) > 2:
                self._release(i)

        return hook

    def _make_bwd_pre(self, i: int):
        def hook(_m, _grad_output):
            self.ensure(i)
            if self.prefetch and i - 1 in self._tensors:
                self._fetch(i - 1)

        return hook

    def _make_bwd_post(self, i: int):
        def hook(_m, grad_input, _grad_output):
            # When no block input requires grad, PyTorch fires this hook *before* the block's own
            # backward runs (at output-grad time). Releasing then would corrupt the backward, so
            # defer to release_all(), which the trainer calls after backward().
            if grad_input and any(g is not None for g in grad_input):
                self._release(i)
            else:
                self._deferred.add(i)

        return hook

    # ----------------------------------------------------------------- lifecycle
    def move_model_to_device(self, model: nn.Module) -> None:
        """Move trainable tensors and resident modules without first uploading swapped frozen blocks.

        Rebinding data preserves the Tensor objects recorded by the hooks, including buffers.
        Using model.to() here would upload all weights and replace buffer objects.
        """
        owned = {id(t) for values in self._tensors.values() for t in values}
        for tensor in list(model.parameters()) + list(model.buffers()):
            if id(tensor) not in owned:
                tensor.data = tensor.data.to(self.device)

    def set_forward_only(self, value: bool) -> None:
        """Inference mode: release every block right after its forward."""
        self.forward_only = value

    def release_all(self) -> None:
        """Release every resident block; call after ``backward()`` (handles deferred releases)."""
        for i in list(self._resident):
            self._release(i)
        self._deferred.clear()

    def remove(self) -> None:
        """Bring every block back to the device permanently and remove hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for i in self.swapped_idx:
            for t, host in zip(self._tensors[i], self._masters[i], strict=True):
                t.data = host.to(self.device)
        self._resident.clear()
        self._masters.clear()

    def summary(self) -> dict:
        return {
            "swapped_blocks": len(self.swapped_idx),
            "bytes_per_block": self.bytes_per_block,
            "pinned": self.pin,
            "device": str(self.device),
        }
