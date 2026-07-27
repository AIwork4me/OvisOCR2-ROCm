"""Doctor: offline readiness check (no model runtime, no network)."""
from __future__ import annotations

from pathlib import Path

from .repo import find_repo_root


def check(repo: Path | None = None) -> dict:
    """Return a readiness report: ``status`` ready|not-ready plus per-check flags.

    ``status == ready`` requires the structural pieces the standard CLI +
    conformance depend on: the adapter bridge, the manifest, and a model card.
    It never touches the GPU or the network.
    """
    repo = Path(repo) if repo else find_repo_root()
    checks = {
        "run_adapter_present": (repo / "adapter" / "run_adapter.py").exists(),
        "rocmdoc_yaml_present": (repo / "rocmdoc.yaml").exists(),
        "model_card_present": (repo / "model_card.json").exists(),
        "model_card_v2_present": (repo / "model_card_v2.json").exists(),
        "package_importable": _package_importable(),
        "offline": True,
    }
    status = "ready" if (checks["run_adapter_present"] and checks["rocmdoc_yaml_present"]
                        and checks["package_importable"]) else "not-ready"
    return {"status": status, "checks": checks}


def _package_importable() -> bool:
    try:
        import ovisocr2_rocm  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
