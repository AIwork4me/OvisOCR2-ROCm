"""Backward-compat re-export — config now lives in ``ovisocr2_rocm.config``.

Existing callers/tests that ``import adapter.adapter_config`` keep working; the
single source of truth is :mod:`ovisocr2_rocm.config`.
"""
from __future__ import annotations

from ovisocr2_rocm.config import (  # noqa: F401
    ENV_LOCAL,
    VALID_BACKENDS,
    VALID_PLATFORMS,
    Config,
    config_snapshot,
    load_env_local,
    resolve,
    validate,
)
