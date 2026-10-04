"""One attention check per model load, before the model computes anything.

The selected backend runs a tiny call through the family's own attention path,
in the dtype and head dimensions the model computes with, on its device: forward
and backward for training, forward only for inference. When it fails, the
existing SDPA path is checked the same way. Errors during later computation are
not caught here; they fail the job as before.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

log = logging.getLogger(__name__)

SDPA = "sdpa"
_LABELS = {
    "auto": "默认",
    "sdpa": "SDPA",
    "xformers": "xFormers",
    "flash_attn": "FlashAttention 2",
    "metal_flash": "Metal FlashAttention",
}
_DTYPES = {torch.float32: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}
_checking = False


class AttentionCheckError(RuntimeError):
    """The selected attention backend failed its check and no permitted fallback passed."""


@contextmanager
def checking() -> Iterator[None]:
    """While a check runs, model-local shims raise instead of quietly using SDPA."""
    global _checking
    previous, _checking = _checking, True
    try:
        yield
    finally:
        _checking = previous


def is_checking() -> bool:
    return _checking


def backend_label(name: str) -> str:
    if name == "flash_attn" and getattr(torch.version, "hip", None):
        return "FlashAttention 2（DTK）"
    return _LABELS.get(name, name)


def dtype_label(dtype: torch.dtype) -> str:
    return _DTYPES.get(dtype, str(dtype).removeprefix("torch."))


def _sdpa_label(device: torch.device) -> str:
    if device.type == "cuda":
        cuda = torch.backends.cuda
        fused = (getattr(cuda, f"{name}_sdp_enabled", lambda: False)() for name in ("flash", "mem_efficient", "cudnn"))
        if not any(fused) and cuda.math_sdp_enabled():
            return "SDPA（数学实现）"
    return "SDPA"


def _where(device: torch.device) -> str:
    """The rank and GPU in a multi-GPU run, where each rank checks on its own; empty otherwise."""
    try:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError):
        return ""
    if world <= 1:
        return ""
    if device.type == "cuda":
        return f"（rank {rank}，GPU {device.index if device.index is not None else torch.cuda.current_device()}）"
    return f"（rank {rank}）"


def _reason(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip()
    return text if len(text) <= 4000 else text[:4000] + "…"


@contextmanager
def _isolated_rng(device: torch.device) -> Iterator[None]:
    """The check may initialize layers or draw inputs; the run's RNG streams stay where they were."""
    cpu = torch.get_rng_state()
    accelerator = None
    if device.type == "cuda":
        accelerator = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        accelerator = torch.mps.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if device.type == "cuda":
            torch.cuda.set_rng_state(accelerator, device)
        elif device.type == "mps":
            torch.mps.set_rng_state(accelerator)


def _run(run: Callable[[str], None], backend: str, device: torch.device) -> None:
    with _isolated_rng(device), checking():
        run(backend)
        # Asynchronous kernel errors must surface inside the check, not in the first step.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()


def check_attention(
    *,
    selected: str,
    run: Callable[[str], None],
    device: torch.device | str,
    dtype: torch.dtype,
    pinned: str | None = None,
) -> str:
    """Run ``run(selected)``; on failure log it and check SDPA. Returns the backend that passed.

    ``pinned`` names the setting that forbids another implementation (reproducible
    training); the check then stops instead of switching.
    """
    device = torch.device(device)
    where = _where(device)
    try:
        _run(run, selected, device)
    except Exception as error:
        if selected == SDPA:
            raise AttentionCheckError(f"注意力后端检查失败{where}：SDPA 无法运行。原因：{_reason(error)}") from error
        if pinned:
            raise AttentionCheckError(
                f"注意力后端检查失败{where}：{backend_label(selected)} 无法运行。{pinned}，不会改用 SDPA。"
                f"原因：{_reason(error)}"
            ) from error
        try:
            _run(run, SDPA, device)
        except Exception as fallback:
            raise AttentionCheckError(
                f"注意力后端检查失败{where}：{backend_label(selected)} 无法运行（{_reason(error)}）；"
                f"SDPA 也无法运行（{_reason(fallback)}）。"
            ) from fallback
        log.warning(
            "注意力后端：%s 不可用，改用 %s，%s%s。原因：%s",
            backend_label(selected), _sdpa_label(device), dtype_label(dtype), where, _reason(error),
        )
        return SDPA
    log.info(
        "注意力后端：%s，%s%s。",
        _sdpa_label(device) if selected == SDPA else backend_label(selected), dtype_label(dtype), where,
    )
    return selected


def grad_mode(training: bool):
    return torch.enable_grad() if training else torch.no_grad()


__all__ = [
    "AttentionCheckError",
    "SDPA",
    "backend_label",
    "check_attention",
    "checking",
    "dtype_label",
    "grad_mode",
    "is_checking",
]
