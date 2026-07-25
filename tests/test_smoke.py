"""Smoke test: the adapter's no-GPU `smoke` backend satisfies the contract.

Runs `adapter/run_adapter.py --backend smoke` as a subprocess (exactly how the
engine invokes it) over a tiny image dir and asserts the contract outputs:
one `<stem>.md` per image + a schema-valid `_run_stats.json`. No GPU required —
this is the CI gate.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "adapter" / "run_adapter.py"
DEMO = REPO / "examples" / "demo.png"


def test_smoke_backend_writes_contract_outputs(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    out_dir = tmp_path / "out"
    # Need a real image file for IMG_EXT matching; copy the shipped demo.
    img_dir.joinpath("page.png").write_bytes(DEMO.read_bytes())

    r = subprocess.run(
        [
            sys.executable,
            str(ADAPTER),
            "--img-dir",
            str(img_dir),
            "--out-dir",
            str(out_dir),
            "--platform",
            "linux-rocm",
            "--backend",
            "smoke",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, f"adapter failed:\n{r.stderr}"

    # R3: one <stem>.md per image.
    md = out_dir / "page.md"
    assert md.exists() and md.read_text(encoding="utf-8").strip()

    # _run_stats.json schema v1, ok count matches.
    rs = json.loads((out_dir / "_run_stats.json").read_text())
    assert rs["schema_version"] == 1
    assert rs["count"] == 1 and rs["ok"] == 1 and rs["fail"] == 0
    assert rs["engine"] == "smoke"
