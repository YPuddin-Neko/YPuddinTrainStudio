"""Model-family registry. Families register lazily so heavy imports only happen on use."""

from __future__ import annotations

from collections.abc import Callable

from .base import ModelFamily

_FACTORIES: dict[str, Callable[[], ModelFamily]] = {}
_INSTANCES: dict[str, ModelFamily] = {}


def register(name: str, factory: Callable[[], ModelFamily]) -> None:
    _FACTORIES[name] = factory


def get_family(name: str) -> ModelFamily:
    if name not in _INSTANCES:
        if name not in _FACTORIES:
            _autoload(name)
        if name not in _FACTORIES:
            raise KeyError(f"unknown model family {name!r}; known: {sorted(_FACTORIES)}")
        _INSTANCES[name] = _FACTORIES[name]()
    return _INSTANCES[name]


def available() -> list[str]:
    for n in ("toy", "anima", "krea2"):
        if n not in _FACTORIES:
            try:
                _autoload(n)
            except ImportError:
                pass
    return sorted(_FACTORIES)


def _autoload(name: str) -> None:
    if name == "toy":
        from . import toy  # noqa: F401
    elif name == "anima":
        from . import anima  # noqa: F401
    elif name == "krea2":
        from . import krea2  # noqa: F401
