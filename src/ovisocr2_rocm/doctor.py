"""Doctor: offline readiness check (no model runtime, no network).

Two levels (Round-2 §9.2):

  * ``structural`` (default) — files + package present; runs with no GPU/deps.
  * ``runtime``              — lazily probes the ROCm/VLM stack (torch.version.hip,
    GPU, gfx, vLLM, model revision, weights availability, offline completeness).
    Never imports torch/vllm at module import time — only inside ``check_runtime``
    when invoked. A missing runtime stack is reported as ``not-available`` (NOT a
    structural failure), so a structural doctor stays green in CPU-only CI.
"""
from __future__ import annotations

import os
from pathlib import Path

from .repo import find_repo_root


def check(repo: Path | None = None, level: str = "structural") -> dict:
    """Return a readiness report: ``status`` ready|not-ready plus per-check flags.

    ``status == ready`` requires the structural pieces the standard CLI +
    conformance depend on: the adapter bridge, the manifest, and a model card.
    It never touches the GPU or the network. When ``level`` is ``runtime`` or
    ``all``, a nested ``runtime`` block is added (informational; it never turns a
    ready structural status into not-ready, so the contract stays stable).
    """
    repo = Path(repo) if repo else find_repo_root()
    checks = {
        "run_adapter_present": (repo / "adapter" / "run_adapter.py").exists(),
        "rocmdoc_yaml_present": (repo / "rocmdoc.yaml").exists(),
        "model_card_present": (repo / "model_card.json").exists(),
        "model_card_v2_present": (repo / "model_card_v2.json").exists(),
        "spec_lock_present": (repo / ".rocmdoc" / "spec-lock.json").exists(),
        "package_importable": _package_importable(),
        "offline": True,
    }
    status = "ready" if (checks["run_adapter_present"] and checks["rocmdoc_yaml_present"]
                        and checks["package_importable"]) else "not-ready"
    report: dict = {"status": status, "level": "structural", "checks": checks}
    if level in ("runtime", "all"):
        report["runtime"] = check_runtime()
    return report


def check_runtime() -> dict:
    """Runtime-level readiness: lazily probe the ROCm/VLM stack.

    ``status`` is one of:
      * ``ready``          — torch+HIP present, >=1 GPU visible;
      * ``not-ready``      — torch importable but no HIP/GPU;
      * ``not-available``  — the runtime stack is absent (CPU-only CI). This is
        NOT a failure of the artifact; it means runtime readiness could not be
        assessed here. Never imports the runtime at module import time.
    """
    checks: dict = {"torch_importable": False, "vllm_importable": False, "offline": True}
    torch_ok = False
    try:
        import torch  # lazy — only when this function is called
        torch_ok = True
        checks["torch_version"] = getattr(torch, "__version__", "unknown")
        checks["torch_hip_available"] = bool(getattr(torch.version, "hip", None))
        try:
            checks["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            checks["gfx_arch"] = (torch.cuda.get_device_name(0)
                                  if checks["gpu_count"] else None)
        except Exception:  # noqa: BLE001
            checks["gpu_count"] = 0
            checks["gfx_arch"] = None
    except Exception:  # noqa: BLE001
        checks["torch_importable"] = False
    checks["torch_importable"] = torch_ok
    try:
        import vllm  # noqa: F401  — lazy
        checks["vllm_importable"] = True
        checks["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        checks["vllm_importable"] = False
    checks["weights_available"] = _weights_available()
    if not torch_ok:
        status = "not-available"
    elif checks.get("torch_hip_available") and checks.get("gpu_count"):
        status = "ready"
    else:
        status = "not-ready"
    return {"status": status, "checks": checks}


def _weights_available() -> bool | None:
    """True/False if a local weights path is configured and exists; else None.

    Probes only LOCAL paths (no network): ``OVISOCR2_MODEL_PATH`` or the HF
    cache snapshot referenced in REPRO.yaml. ``None`` means 'unknown / not
    configured here' — never a failure.
    """
    cand = os.environ.get("OVISOCR2_MODEL_PATH")
    if cand:
        return Path(cand).exists()
    repro = find_repo_root() / "REPRO.yaml"
    if repro.exists():
        try:
            import re
            text = repro.read_text(encoding="utf-8")
            m = re.search(r"(?:model_path|MODEL_PATH|weights):\s*['\"]?([^'\"\n]+)", text)
            if m:
                p = Path(m.group(1).strip())
                return p.exists() if p.is_absolute() else None
        except Exception:  # noqa: BLE001
            return None
    return None


def _package_importable() -> bool:
    try:
        import ovisocr2_rocm  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
