"""Provenance + environment fingerprinting.

``collect_env`` is a best-effort runtime fingerprint (every field None if the
dependency is absent) — it imports torch/vllm ONLY when called, never at module
import. The dtype it records is DETECTED from the loaded model, never a config
parameter (no fabricated bf16).
"""
from __future__ import annotations

import subprocess
from contextlib import suppress
from pathlib import Path


def _repo_root() -> Path:
    from .repo import find_repo_root
    return find_repo_root()


def collect_env() -> dict:
    """Best-effort runtime fingerprint; every field None if unavailable."""
    out = {
        "output_tokens": None, "tokens_per_second": None,
        "max_memory_allocated_mb": None, "max_memory_reserved_mb": None,
        "gpu": None, "gfx_arch": None, "torch_version": None, "hip_version": None,
        "vllm_version": None, "transformers_version": None,
        "git_commit": None, "git_dirty": None, "weights_revision": None,
        "dtype": None,
    }
    with suppress(Exception):
        import torch

        out["torch_version"] = torch.__version__
        out["hip_version"] = getattr(torch.version, "hip", None)
        if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            out["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            out["max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
            out["gpu"] = torch.cuda.get_device_name(0)
            with suppress(Exception):
                # gfx arch from the HIP device name (e.g. "gfx1100")
                import re
                m = re.search(r"gfx[0-9a-f]+", out["gpu"].lower())
                if m:
                    out["gfx_arch"] = m.group(0)
    for modname, key in (("vllm", "vllm_version"), ("transformers", "transformers_version")):
        with suppress(Exception):
            out[key] = __import__(modname).__version__
    with suppress(Exception):
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False, cwd=_repo_root())
        out["git_commit"] = rev.stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               check=False, cwd=_repo_root())
        out["git_dirty"] = bool(dirty.stdout.strip())
    return out


def detect_dtype(llm) -> str | None:
    """Read the ACTUAL dtype off a loaded vLLM model; None if not discoverable.

    Never fabricated: if we cannot read it from the model config, we return None
    rather than guessing bf16.
    """
    with suppress(Exception):
        cfg = getattr(llm, "llm_engine", None)
        model = getattr(cfg, "model_config", None) if cfg else None
        mc = getattr(model, "hf_config", None) if model else None
        dt = getattr(mc, "torch_dtype", None)
        if dt is None and model is not None:
            dt = getattr(model, "dtype", None)
        if dt is not None:
            return str(dt).replace("torch.", "")
    return None


def run_fingerprint(*, prompt_sha256: str, config_snapshot: dict) -> str:
    """Deterministic fingerprint of everything that changes a page's output.

    Two runs with the same fingerprint produce byte-identical ``.md`` for a given
    page, so a stored output is safe to RESUME. A fingerprint mismatch means the
    config changed (weights/model/params/prompt) and stored outputs are STALE —
    they must be re-run, never silently reused (D5: safe resume).
    """
    import hashlib
    import json as _json
    # only the output-affecting fields — provenance-only fields are excluded
    affecting = {
        "backend": config_snapshot.get("backend"),
        "weights_dir": config_snapshot.get("weights_dir"),
        "max_tokens": config_snapshot.get("max_tokens"),
        "temperature": config_snapshot.get("temperature"),
        "min_pixels": config_snapshot.get("min_pixels"),
        "max_pixels": config_snapshot.get("max_pixels"),
        "gdn_prefill_backend": config_snapshot.get("gdn_prefill_backend"),
        "prompt_sha256": prompt_sha256,
    }
    blob = _json.dumps(affecting, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
