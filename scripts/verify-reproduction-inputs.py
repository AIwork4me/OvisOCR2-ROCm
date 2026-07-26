#!/usr/bin/env python3
"""Verify REPRO.yaml reproduction inputs.

Confirms the weights sha256 against ``model.safetensors`` when present locally
(the hash target is documented here, in the code, so it is unambiguous), and
prints the pinned weights/dataset revisions + expected engine. Exits 1 only on a
hash mismatch when the weights are present; missing weights is informational.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
# REPRO.yaml `weights.sha256` is the sha256 of this single file:
WEIGHTS_HASH_TARGET = "model.safetensors"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify REPRO.yaml reproduction inputs")
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--weights", default="/root/models/OvisOCR2")
    a = p.parse_args(argv)
    rc = 0
    repro = yaml.safe_load((Path(a.repo) / "REPRO.yaml").read_text())
    print(f"[verify] model={repro['model_id']} overall={repro['overall']} "
          f"engine_image={repro['environment']['image']}")
    print(f"  weights.sha256 target file: {WEIGHTS_HASH_TARGET}")
    target = Path(a.weights) / WEIGHTS_HASH_TARGET
    if target.exists():
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h == repro["weights"]["sha256"]:
            print(f"  [OK] sha256({WEIGHTS_HASH_TARGET}) == REPRO ({h[:12]}...)")
        else:
            want = repro["weights"]["sha256"][:12]
            print(f"  [FAIL] sha256 mismatch: got {h[:12]}... want {want}...")
            rc = 1
    else:
        print(f"  [INFO] {target} not present locally -- cannot verify weights hash")
    print(f"  weights revision: {repro['weights']['revision']}")
    print(f"  dataset revision: {repro['dataset']['revision']}  "
          f"gt_sha256: {repro['dataset']['gt_sha256']}")
    print(f"  expected engine:  {repro['environment']['image']}")
    print(f"  REPRO.git_commit: {repro['git_commit']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
