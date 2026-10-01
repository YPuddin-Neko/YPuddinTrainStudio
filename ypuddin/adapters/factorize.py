"""Integer factorization used by LoKr to split a dimension into two Kronecker factors.

Semantics are kept identical to LyCORIS so community presets (``factor=8`` etc.) mean
the same thing here:

* ``factor == -1``: the divisor pair closest to ``sqrt(dim)`` (minimal ``m + n``).
* ``factor > 0`` and ``factor | dim``: ``(min(factor, dim // factor), max(...))``.
* ``factor > 0`` otherwise: the largest divisor ``<= factor`` (a warning is emitted).

The smaller value is always returned first and is the side given to ``W1``.
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if factor == 0 or factor < -1:
        raise ValueError("factor must be -1 or a positive integer")
    if dimension == 1:
        return 1, 1
    if factor > 0 and dimension % factor == 0:
        m, n = factor, dimension // factor
        return (m, n) if m <= n else (n, m)

    limit = dimension if factor < 0 else factor
    if factor > 0:
        log.warning(
            "factor %d does not divide %d; using the largest divisor <= %d", factor, dimension, factor
        )
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > limit:
            break
        m, n = new_m, new_n
    if m > n:
        m, n = n, m
    if m == 1 and dimension > 1:
        log.warning(
            "dimension %d has no useful factorization (prime?); LoKr degenerates to a scaled LoRA", dimension
        )
    return m, n


def is_balanced(dimension: int, pair: tuple[int, int]) -> bool:
    return pair == factorization(dimension, -1)


def sqrt_hint(dimension: int) -> int:
    return int(math.isqrt(dimension))
