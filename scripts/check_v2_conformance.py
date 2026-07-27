#!/usr/bin/env python3
"""v2 conformance gate — the CI-only source of truth for this model repo.

Runs everything CI needs WITHOUT a GPU (smoke backend + structural checks):
  1. .rocmdoc/spec-lock.json pins a 40-char central commit (not main).
  2. model_card.json (v1), model_card_v2.json, rocmdoc.yaml all schema-valid.
  3. model_card_v2 invariants (result_id uniqueness, assurance, derived platforms).
  4. structural conformance (check_repo).
  5. behavioral profiles base + runtime-core + benchmark-omnidocbench-v16 (smoke).
  6. README results blocks are in sync with model_card_v2.json (no handwritten drift).

Exit 0 = conformant; 1 = not. Reuses the pinned central engine for all checks.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from omnidocbench_rocm.conformance import check_repo
from omnidocbench_rocm.conformance_profiles import check_profile
from omnidocbench_rocm.model_card_v2 import validate_card_v2
from omnidocbench_rocm.schema import validate_artifact

REPO = Path(__file__).resolve().parents[1]
PROBLEMS: list[str] = []


def _problem(msg: str) -> None:
    PROBLEMS.append(msg)


def check_spec_lock() -> None:
    p = REPO / ".rocmdoc" / "spec-lock.json"
    if not p.exists():
        _problem(".rocmdoc/spec-lock.json missing")
        return
    d = json.loads(p.read_text())
    if d.get("repo") != "AIwork4me/OmniDocBench-ROCm":
        _problem(f"spec-lock repo wrong: {d.get('repo')!r}")
    commit = d.get("commit", "")
    if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit.lower()):
        _problem(f"spec-lock commit is not a 40-hex SHA: {commit!r}")


def _validate(name: str, obj) -> None:
    try:
        validate_artifact(name, obj)
    except Exception as e:  # noqa: BLE001
        _problem(f"{name} invalid: {e}")


def check_schemas() -> None:
    v1 = json.loads((REPO / "model_card.json").read_text())
    _validate("model_card", v1)
    v2 = json.loads((REPO / "model_card_v2.json").read_text())
    _validate("model_card_v2", v2)
    for m in validate_card_v2(v2):
        _problem(f"model_card_v2 invariant: {m}")
    mf = yaml.safe_load((REPO / "rocmdoc.yaml").read_text())
    _validate("rocmdoc_manifest", mf)
    # result-alignment: every result's platform/backend must be declared supported
    declared = {(i.get("platform"), i.get("backend")) for i in mf.get("implementations") or []
                if i.get("status") in ("supported", "experimental")}
    for r in v2.get("results", []):
        cov = r.get("coverage") or {}
        impl = r.get("implementation") or {}
        key = (cov.get("platform"), impl.get("backend"))
        if r.get("status") == "valid" and key not in declared:
            _problem(f"result {r.get('result_id')} claims {key} not declared supported in rocmdoc.yaml")


def check_structural() -> None:
    r = check_repo(REPO)
    for f in r.failures:
        _problem(f"check_repo: {f}")


def check_profiles() -> None:
    cli = shutil.which("ovisocr2-rocm")
    if not cli:
        _problem("ovisocr2-rocm console script not on PATH (pip install -e . ?)")
        return
    for prof in ("base", "runtime-core"):
        rep = check_profile(prof, cli_path=cli)
        for f in rep.failures:
            _problem(f"profile {prof}: {f}")
    with tempfile.TemporaryDirectory() as t:
        from PIL import Image
        imgs = Path(t) / "imgs"
        imgs.mkdir()
        for i in range(2):
            Image.new("RGB", (16, 16)).save(imgs / f"p{i+1}.png")
        rep = check_profile("benchmark-omnidocbench-v16", cli_path=cli, img_dir=imgs,
                            out_dir=Path(t) / "out", requested_backend="smoke")
        for f in rep.failures:
            _problem(f"profile benchmark-omnidocbench-v16(smoke): {f}")


def check_readme_drift() -> None:
    for args in (["README.md", "--lang", "en"], ["README.zh-CN.md", "--lang", "zh"]):
        import subprocess
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "generate_readme_results.py"),
                            *args, "--check"], capture_output=True, text=True)
        if r.returncode != 0:
            _problem(f"README drift: {' '.join(args)} -> {r.stderr.strip()}")


def main() -> int:
    check_spec_lock()
    check_schemas()
    check_structural()
    check_profiles()
    check_readme_drift()
    if PROBLEMS:
        print("NON-CONFORMANT (v2):", file=sys.stderr)
        for p in PROBLEMS:
            print("  -", p, file=sys.stderr)
        return 1
    print("CONFORMANT (v2) — spec-lock, schemas, invariants, profiles, README all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
