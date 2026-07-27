"""Output writers: atomic per-page ``.md``, run_stats, and the cli_result view.

One UTF-8 ``<stem>.md`` per page image (R3). ``write_run_stats`` emits the exact
``_run_stats.json`` shape the engine / cli_bridge / conformance profiles read.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with open(tmp, "ab") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_unlink(path: Path) -> None:
    """Best-effort remove; never raise."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


@dataclass
class PageStatus:
    image: str
    status: str
    error: str = ""
    seconds: float | None = None
    attempts: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # keep the JSON tidy: drop empty error, null seconds only when absent
        return {k: v for k, v in d.items() if not (k == "error" and not v)}


def build_run_stats(
    *,
    count: int,
    ok: int,
    fail: int,
    fallback: int,
    limit_pages: int | None,
    stats: list[PageStatus],
    engine: str,
    efficiency: dict[str, Any] | None = None,
) -> dict:
    out: dict[str, Any] = {
        "schema_version": 1,
        "count": count,
        "ok": ok,
        "fail": fail,
        "fallback": fallback,
        "limit_pages": limit_pages,
        "engine": engine,
        "stats": [s.to_dict() for s in stats],
    }
    if efficiency:
        out["efficiency"] = efficiency
    return out


def write_run_stats(path: Path, run_stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(run_stats, ensure_ascii=False, indent=2))


def build_cli_result(
    *,
    status: str,
    backend: str,
    stats: list[PageStatus],
    count: int,
    ok: int,
    failed: int,
    skipped: int,
    output_dir: str,
    full_set: bool,
) -> dict:
    """Build the ``cli_result`` JSON object (matches the central $def)."""
    pages = []
    for s in stats:
        st = s.status
        mapped = "ok" if st == "ok" else ("failed" if st.startswith("failed") else "skipped")
        pages.append({
            "image": s.image,
            "status": mapped,
            "error": s.error,
            "seconds": s.seconds if s.seconds is not None else 0.0,
        })
    return {
        "schema_version": 1,
        "status": status,
        "backend": backend,
        "engine": backend,
        "page_count": count,
        "ok": ok,
        "failed": failed,
        "skipped": skipped,
        "output_dir": output_dir,
        "full_set": full_set,
        "pages": pages,
    }
