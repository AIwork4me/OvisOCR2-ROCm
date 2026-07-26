"""Repro/env script tests (P0-1): --help, REPRO structure, check-environment CLI."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_verify_repro_inputs_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify-reproduction-inputs.py"), "--help"],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0


def test_repro_yaml_sha256_field_present():
    import yaml

    repro = yaml.safe_load((REPO / "REPRO.yaml").read_text())
    sha = repro["weights"]["sha256"]
    # 64-char hex; the file it hashes (model.safetensors) is documented in
    # scripts/verify-reproduction-inputs.py and confirmed in the T9 verification.
    assert isinstance(sha, str) and len(sha) == 64
    assert repro["dataset"]["gt_sha256"]
    assert repro["git_commit"]


def test_check_env_cpu_only_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check-environment.py"), "--help"],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "--cpu-only" in r.stdout


def test_install_deps_uses_unified_venv_default():
    # The setup script must default to the same venv the docs/results use.
    text = (REPO / "adapter" / "setup" / "00-install-deps.sh").read_text()
    assert "/root/venvs/vllm-0221b" in text
    assert "bash <(curl" not in text
