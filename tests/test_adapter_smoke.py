"""Smoke-backend contract tests (P0-2/3): no GPU, no vllm import, atomic outputs."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "adapter" / "run_adapter.py"
DEMO = REPO / "examples" / "demo.png"


def _run(args, **kw):
    return subprocess.run([sys.executable, str(ADAPTER), *args], capture_output=True, text=True, check=False, **kw)


def test_smoke_writes_contract_outputs(tmp_path):
    img = tmp_path / "imgs"
    img.mkdir()
    (img / "page.png").write_bytes(DEMO.read_bytes())
    out = tmp_path / "out"
    r = _run(["--img-dir", str(img), "--out-dir", str(out),
              "--platform", "linux-rocm", "--backend", "smoke"])
    assert r.returncode == 0, r.stderr
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["schema_version"] == 1 and rs["count"] == 1 and rs["ok"] == 1
    assert rs["engine"] == "smoke"
    assert (out / "page.md").read_text(encoding="utf-8").strip()


def test_smoke_empty_dir_is_ok(tmp_path):
    img = tmp_path / "empty"
    img.mkdir()
    out = tmp_path / "out"
    r = _run(["--img-dir", str(img), "--out-dir", str(out),
              "--platform", "linux-rocm", "--backend", "smoke"])
    assert r.returncode == 0
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 0 and rs["ok"] == 0


def test_smoke_does_not_import_vllm(tmp_path):
    img = tmp_path / "imgs"
    img.mkdir()
    (img / "page.png").write_bytes(DEMO.read_bytes())
    out = tmp_path / "out"
    probe = (
        "import sys, runpy; "
        f"sys.argv=['x','--img-dir','{img}','--out-dir','{out}',"
        "'--platform','linux-rocm','--backend','smoke']; "
        f"runpy.run_path('{ADAPTER}', run_name='__main__'); "
        'assert "vllm" not in sys.modules, "smoke imported vllm"'
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
