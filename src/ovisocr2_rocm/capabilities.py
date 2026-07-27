"""Capabilities declaration, read from the static ``rocmdoc.yaml`` (ADR-0009).

Declares supported platforms/backends/interfaces. Must agree with the manifest;
the standard CLI never imports a model runtime to answer ``capabilities``.
"""
from __future__ import annotations

from pathlib import Path

from .repo import find_repo_root


def read_manifest(repo: Path | None = None) -> dict | None:
    repo = Path(repo) if repo else find_repo_root()
    mf = repo / "rocmdoc.yaml"
    if not mf.exists():
        return None
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a dev dep; degrade gracefully
        return None
    try:
        return yaml.safe_load(mf.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return None


def capabilities(repo: Path | None = None) -> dict:
    """Return the cli_capabilities JSON object (matches the central $def)."""
    mf = read_manifest(repo)
    if mf is None:
        return {"platforms": [], "interfaces": [],
                "warning": "no rocmdoc.yaml — cannot declare capabilities"}
    plats = []
    for impl in mf.get("implementations") or []:
        if impl.get("status", "supported") in ("supported", "experimental"):
            plats.append({
                "platform": impl.get("platform"),
                "backend": impl.get("backend", ""),
                "precision": impl.get("precision", ""),
                "interface": impl.get("interface", "adapter-script"),
            })
    return {"platforms": plats, "interfaces": mf.get("interfaces", [])}
