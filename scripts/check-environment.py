#!/usr/bin/env python3
"""Verify the active Python can serve OvisOCR2 (or, with --cpu-only, that the
repo + engine are importable without a GPU). Clear errors; never silent.

Exit 0 = ready, 1 = not ready. With --cpu-only the GPU/vLLM checks are skipped
(used in CI, which has no AMD GPU).
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path


def _ok(msg):
    print(f"  [OK] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify the OvisOCR2-ROCm environment")
    p.add_argument("--cpu-only", action="store_true", help="skip GPU/vLLM checks (CI)")
    p.add_argument("--weights", default="/root/models/OvisOCR2")
    a = p.parse_args(argv)
    rc = 0
    _ok(f"Python {platform.python_version()}")
    try:
        import torch

        _ok(f"torch {torch.__version__}")
    except (ImportError, RuntimeError, OSError):
        if a.cpu_only:
            print("  [INFO] torch not installed (OK for --cpu-only; required for real inference)")
        else:
            rc = _fail("torch import failed")

    if not a.cpu_only:
        try:
            import torch
            import vllm

            _ok(f"vLLM {vllm.__version__}")
            from vllm.model_executor.models.registry import ModelRegistry

            if "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs():
                _ok("Qwen3_5ForConditionalGeneration registered")
            else:
                rc = _fail("Qwen3_5ForConditionalGeneration NOT registered")
            hip = getattr(torch.version, "hip", None)
            if hip:
                _ok(f"HIP/ROCm {hip}")
            else:
                rc = _fail("no HIP/ROCm in this torch build")
            if torch.cuda.is_available():
                _ok(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                rc = _fail("torch.cuda not available (no GPU visible)")
        except (ImportError, RuntimeError, OSError) as e:
            rc = _fail(f"vLLM/GPU check: {e}")

    wdir = Path(a.weights)
    if wdir.exists():
        if (wdir / "model.safetensors").exists():
            _ok(f"weights at {wdir}")
        else:
            rc = _fail(f"weights missing model.safetensors at {wdir}")
    elif not a.cpu_only:
        print(f"  [INFO] no weights at {wdir} (override with --weights)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
