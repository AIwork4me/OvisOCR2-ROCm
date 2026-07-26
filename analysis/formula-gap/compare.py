#!/usr/bin/env python3
"""Compare two per-formula CDM runs to attribute the formula-CDM gap.

Expects two JSON files, each a list of objects with a stable ``key`` (e.g.
"image:formula_id") and a numeric ``cdm``. Prints the mean CDM of each run, the
number of formulas where the candidate regresses vs the baseline, and the 10
largest per-formula deltas.

This is a SKELETON for the A/B described in README.md. No result files are
fabricated — fill in real per-formula CDM paths when re-running.

Example:
    python compare.py --baseline run_0.22.1.json --candidate run_0.19.0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    return {str(r["key"]): float(r["cdm"]) for r in data}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare two per-formula CDM runs")
    p.add_argument("--baseline", required=True, help="JSON [{key, cdm}, ...] (e.g. vLLM 0.22.1)")
    p.add_argument("--candidate", required=True, help="JSON [{key, cdm}, ...] (e.g. vLLM 0.19.0)")
    a = p.parse_args(argv)

    base = _load(a.baseline)
    cand = _load(a.candidate)
    keys = sorted(set(base) & set(cand))
    if not keys:
        print("no overlapping keys between baseline and candidate", file=sys.stderr)
        return 1

    deltas = [(k, cand[k] - base[k]) for k in keys]
    regressed = [k for k, d in deltas if d < 0]
    mean_b = sum(base[k] for k in keys) / len(keys)
    mean_c = sum(cand[k] for k in keys) / len(keys)
    print(f"compared {len(keys)} formulas")
    print(f"baseline  mean CDM = {mean_b:.4f}")
    print(f"candidate mean CDM = {mean_c:.4f}")
    print(f"regressed (candidate < baseline): {len(regressed)}")
    for k, d in sorted(deltas, key=lambda x: x[1])[:10]:
        print(f"  {k}: delta={d:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
