"""Standard CLI contract tests (ADR-0011) — no GPU, fake/real backends only.

Runs the real console script + the thin adapter as subprocesses to assert the
contract surface (pure JSON, schema-valid outputs, exit codes, no silent
fallback, page conservation, adapter==CLI single pipeline).
"""
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from omnidocbench_rocm.schema import validate_artifact
from PIL import Image

import ovisocr2_rocm.cli as cli
import ovisocr2_rocm.pipeline as pipeline

REPO = Path(__file__).resolve().parents[1]
CLI = shutil.which("ovisocr2-rocm")


def _imgs(tmp, names):
    d = tmp / "imgs"
    d.mkdir(exist_ok=True)
    for n in names:
        Image.new("RGB", (16, 16)).save(d / n)
    return d


def _run(args, **kw):
    return subprocess.run([sys.executable, CLI, *args] if CLI else [sys.executable, "-m", "ovisocr2_rocm.cli", *args],
                          capture_output=True, text=True, **kw)


def test_cli_json_contract(tmp_path):
    """version/capabilities/doctor emit PURE JSON that validates against the $defs."""
    for cmd, art in [("version", "cli_version"), ("capabilities", "cli_capabilities")]:
        r = _run([cmd, "--json"])
        assert r.returncode == 0, r.stderr
        obj = json.loads(r.stdout)             # pure JSON (raises if logs mixed in)
        validate_artifact(art, obj)            # schema-valid
    r = _run(["doctor", "--json"])
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["status"] in ("ready", "not-ready")


def test_parse_smoke_conserves_pages(tmp_path):
    d = _imgs(tmp_path, ["p1.png", "p2.png", "p3.png"])
    out = tmp_path / "out"
    r = _run(["parse", "--img-dir", str(d), "--out-dir", str(out),
              "--platform", "linux-rocm", "--backend", "smoke", "--json"])
    assert r.returncode == 0, r.stderr
    obj = json.loads(r.stdout)
    validate_artifact("cli_result", obj)
    assert obj["status"] == "ok" and obj["page_count"] == 3 and obj["ok"] == 3
    assert len(list(out.glob("*.md"))) == 3
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 3  # page conservation: no page lost


@pytest.mark.parametrize("args,needle", [
    (["--backend", "bogus"], "invalid backend"),
    (["--backend", "vllm", "--server-url", "http://x:8000"], "server_url is not supported"),
])
def test_no_silent_fallback(tmp_path, args, needle):
    d = _imgs(tmp_path, ["p.png"])
    r = _run(["parse", "--img-dir", str(d), "--out-dir", str(tmp_path / "out"),
              "--platform", "linux-rocm", *args, "--json"])
    assert r.returncode == 2                      # USAGE — never silently proceeds
    assert needle in r.stdout                     # explicit error to stdout JSON


def test_adapter_bridge_uses_same_pipeline(tmp_path):
    """The thin adapter subprocess produces the same _run_stats the CLI does
    (single pipeline, no second inference implementation)."""
    d = _imgs(tmp_path, ["a.png", "b.png"])
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(REPO / "adapter" / "run_adapter.py"),
                        "--img-dir", str(d), "--out-dir", str(out),
                        "--platform", "linux-rocm", "--backend", "smoke"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 2 and rs["ok"] == 2 and rs["engine"] == "smoke"


def _ns(**kw):
    base = dict(img_dir=".", out_dir=".", platform="linux-rocm", backend="smoke",
                benchmark="omnidocbench-v16", server_url=None, api_model_name=None,
                resume=None, skip_existing=None, batch_size=None, num_shards=None,
                shard_index=None, limit_pages=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_cli_exit_code_mapping(monkeypatch, tmp_path):
    """status ok/partial/failed -> exit 0/1/5 (partial is never a crash)."""
    (tmp_path / "imgs").mkdir()
    for st, exp in [("ok", cli.EXIT_OK), ("partial", cli.EXIT_PARTIAL),
                    ("failed", cli.EXIT_FATAL)]:
        monkeypatch.setattr(pipeline, "run_pipeline",
                            lambda *a, _st=st, **k: {"schema_version": 1, "status": _st,
                                                    "pages": [], "backend": "smoke"})
        assert cli._cmd_parse(_ns(img_dir=str(tmp_path / "imgs"), out_dir=str(tmp_path / "o"))) == exp


def test_partial_success_status(tmp_path, fake_vllm):
    """One page failing -> status partial + failed recorded (R2: run continues)."""
    d = _imgs(tmp_path, ["good.png"])
    (d / "bad.png").write_bytes(b"not an image")
    fake_vllm()
    result = pipeline.run_pipeline(d, tmp_path / "out", platform="linux-rocm",
                                   cli={"backend": "vllm", "weights_dir": "fake"})
    assert result["status"] == "partial"
    assert result["ok"] == 1 and result["failed"] == 1
