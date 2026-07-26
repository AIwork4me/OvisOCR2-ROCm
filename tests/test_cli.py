"""CLI surface tests (P0-2/3/4): --help, backend validation, combo rejection."""
import subprocess
import sys
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[1] / "adapter" / "run_adapter.py"


def _run(args):
    return subprocess.run([sys.executable, str(ADAPTER), *args], capture_output=True, text=True, check=False)


def test_help_exits_zero():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "--batch-size" in r.stdout and "--num-shards" in r.stdout


def test_invalid_backend_exits_nonzero(tmp_path):
    r = _run(["--img-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"),
              "--platform", "linux-rocm", "--backend", "bogus"])
    assert r.returncode != 0 and "backend" in r.stderr.lower()


def test_windows_hip_vllm_rejected(tmp_path):
    r = _run(["--img-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"),
              "--platform", "windows-hip", "--backend", "vllm"])
    assert r.returncode != 0 and "windows-hip" in r.stderr
