"""Signed LoRA objectives; callers backpropagate before leaving each direction's scope."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ypuddin.adapters.components import ComponentAdapterSet
    from ypuddin.adapters.inject import AdapterSet

MultiplierScope = Callable[[float], AbstractContextManager]
Prediction = Callable[[], Tensor]
Loss = Callable[[Tensor, Tensor], Tensor]
T = TypeVar("T")


@dataclass(frozen=True)
class TextTargets:
    positive: Tensor
    negative: Tensor
    anchor: Tensor | None = None


@dataclass(frozen=True)
class DirectionLoss:
    sign: int
    loss: Tensor


@contextmanager
def adapter_multiplier(adapters: AdapterSet | ComponentAdapterSet, value: float) -> Iterator[None]:
    """Restore every layer's original multiplier, including on interrupted backward."""
    original = [(layer, layer.multiplier) for layer in adapters.layers.values()]
    try:
        adapters.set_multiplier(value)
        yield
    finally:
        for layer, multiplier in original:
            layer.multiplier = multiplier


def text_targets(
    neutral: Prediction,
    positive: Prediction,
    negative: Prediction,
    *,
    multiplier_scope: MultiplierScope,
    guidance: float,
    anchor: Prediction | None = None,
) -> TextTargets:
    """Callbacks predict the same noised latent/timestep with different frozen text conditions."""
    with multiplier_scope(0.0), torch.no_grad():
        base = neutral().detach().float()
        pos = positive().detach().float()
        neg = negative().detach().float()
        if base.shape != pos.shape or base.shape != neg.shape:
            raise ValueError("slider teacher predictions must have matching shapes")
        delta = guidance * (pos - neg)
        anchor_target = anchor().detach().float() if anchor is not None else None
        return TextTargets(base + delta, base - delta, anchor_target)


def paired_inputs(
    positive: Tensor,
    negative: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    *,
    prepare: Callable[[Tensor, Tensor, Tensor], T],
) -> tuple[T, T]:
    """Prepare both poles with the supplied shared noise/time and the family's own target rule."""
    if positive.shape != negative.shape or positive.shape != noise.shape:
        raise ValueError("slider image pairs and their noise must have matching shapes")
    with torch.no_grad():
        return prepare(positive, noise, timesteps), prepare(negative, noise, timesteps)


@contextmanager
def direction_loss(
    multiplier_scope: MultiplierScope,
    sign: int,
    forward: Prediction,
    target: Tensor,
    *,
    loss_fn: Loss,
    anchor_forward: Prediction | None = None,
    anchor_target: Tensor | None = None,
    anchor_weight: float = 1.0,
    weight: float = 0.5,
) -> Iterator[DirectionLoss]:
    """Yield an already weighted loss; the caller must backward inside this ``with`` block.

    ``loss_fn`` keeps the family's reduction, timestep weighting and mask. The scope stays
    active through checkpoint recomputation, and each pole builds a separate graph.
    """
    if sign not in (-1, 1):
        raise ValueError("slider direction must be +1 or -1")
    if (anchor_forward is None) != (anchor_target is None):
        raise ValueError("slider anchor needs both a prediction and a frozen target")
    with multiplier_scope(float(sign)):
        loss = loss_fn(forward(), target.detach())
        if anchor_forward is not None and anchor_weight != 0:
            loss = loss + anchor_weight * loss_fn(anchor_forward(), anchor_target.detach())
        if loss.numel() != 1:
            raise ValueError("slider loss must be reduced to one value")
        yield DirectionLoss(sign=sign, loss=weight * loss)
