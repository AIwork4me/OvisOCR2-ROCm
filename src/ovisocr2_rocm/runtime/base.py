"""Runtime backend abstraction.

A ``RuntimeBackend`` turns a list of page images into predictions. The contract:

  * ``load(cfg, prompt)`` — do heavy imports + model load HERE (lazy). Must not
    run at module import time.
  * ``infer(images, cfg) -> list[PageResult]`` — never raises per-page; a failed
    page is returned with ``error`` set. The pipeline records it and continues
    (R2). On a whole-batch failure the backend bisects to single pages.
  * ``actual_dtype()`` / ``env()`` — report what ACTUALLY ran (detected), never
    what was requested.

``SmokeBackend`` is a no-GPU placeholder (backend=="smoke") so the full standard
CLI + conformance profiles run in CI without torch/vllm/GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


@dataclass
class PageResult:
    image: Path
    text: str | None          # None on failure
    seconds: float | None     # honest only for single-page batches; else None
    error: str | None         # None on success


class RuntimeBackend:
    """Abstract base. ``name`` is the backend recorded in run_stats / cli_result."""
    name: str = "abstract"

    def load(self, cfg: Config, prompt: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def infer(self, images: list[Path], cfg: Config) -> list[PageResult]:  # pragma: no cover
        raise NotImplementedError

    def actual_dtype(self) -> str | None:
        return None

    def env(self) -> dict:
        return {}

    def close(self) -> None:
        """Release any held resources (model, engine). Best-effort."""


class SmokeBackend(RuntimeBackend):
    """No-GPU placeholder: writes a deterministic stub ``.md`` per page.

    Used by CI and the standard-CLI conformance profiles. Never imports torch or
    vllm, so it is importable everywhere.
    """

    name = "smoke"

    def load(self, cfg: Config, prompt: str) -> None:
        return None

    def actual_dtype(self) -> str | None:
        return None

    def env(self) -> dict:
        return {"backend": "smoke", "gpu": None, "gfx_arch": None,
                "torch_version": None, "hip_version": None, "vllm_version": None,
                "dtype": None}

    def infer(self, images: list[Path], cfg: Config) -> list[PageResult]:
        out: list[PageResult] = []
        for i in images:
            text = f"# {i.stem}\n\n(smoke output — backend=smoke)\n"
            out.append(PageResult(image=i, text=text, seconds=None, error=None))
        return out

    def close(self) -> None:
        return None
