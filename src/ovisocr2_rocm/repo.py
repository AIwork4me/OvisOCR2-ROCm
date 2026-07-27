"""Locate the model-repo root robustly.

The standard CLI is invoked three ways — as a console script from the repo root,
as ``python -m ovisocr2_rocm.cli`` from the repo root, and as a subprocess by the
conformance profiles (``sys.executable <repo>/.../cli.py`` with CWD = repo root).
In all of those, locating ``rocmdoc.yaml`` by walking up from CWD (then from the
package's own location) is correct and does not depend on install layout.
"""
from __future__ import annotations

from pathlib import Path

_MARKER = "rocmdoc.yaml"


def find_repo_root(start: Path | None = None) -> Path:
    """Return the directory containing ``rocmdoc.yaml``.

    Searches CWD upward first, then the package file's location upward, then
    falls back to the src-layout repo root (``parents[1]`` of this package dir).
    """
    def has_marker(p: Path) -> bool:
        return (p / _MARKER).exists()

    candidates: list[Path] = []
    cwd = Path(start) if start else Path.cwd()
    candidates.extend([cwd, *cwd.parents])
    here = Path(__file__).resolve().parent  # .../src/ovisocr2_rocm
    candidates.extend([here, *here.parents])
    for p in candidates:
        if has_marker(p):
            return p
    # fallback: src layout -> package dir's grandparent is the repo root
    return here.parents[1]
