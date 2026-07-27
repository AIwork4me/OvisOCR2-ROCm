"""Runtime backend registry.

The registry is the single place that maps a backend NAME to a RuntimeBackend.
``select`` raises on an unknown backend — no silent fallback (D5).
"""
from __future__ import annotations

from .base import PageResult, RuntimeBackend, SmokeBackend


def _vllm_factory() -> RuntimeBackend:
    # imported lazily so the registry stays runtime-free at import time
    from .vllm_inprocess import VLLMBackend
    return VLLMBackend()


_BACKENDS = {
    "smoke": SmokeBackend,
    "vllm": _vllm_factory,
}


def available() -> list[str]:
    return sorted(_BACKENDS)


def select(name: str) -> RuntimeBackend:
    """Return a fresh RuntimeBackend for ``name``. Raise on unknown backend."""
    if name not in _BACKENDS:
        raise SystemExit(
            f"unknown backend {name!r}; available: {available()}. No silent fallback.")
    entry = _BACKENDS[name]
    if callable(entry) and not isinstance(entry, type):
        return entry()
    return entry()  # type: ignore[operator]


__all__ = ["RuntimeBackend", "SmokeBackend", "PageResult", "select", "available"]
