"""vLLM-backend failure/recovery + D5 invariants via the fake_vllm fixture (CPU).

Exercises the REAL :class:`ovisocr2_rocm.runtime.vllm_inprocess.VLLMBackend`
(not a mock pipeline) so the strict-zip, no-silent-fallback and resume logic are
covered on CPU. Mirrors the legacy scenarios; ``skip-existing`` is replaced by a
fingerprint-safe resume test (D5).
"""
import json

import pytest
from PIL import Image

from ovisocr2_rocm.pipeline import run_pipeline


def _make_imgs(tmp_path, names):
    d = tmp_path / "imgs"
    d.mkdir()
    for n in names:
        Image.new("RGB", (8, 8), "white").save(d / n)
    return d


def _rs(out):
    return json.loads((out / "_run_stats.json").read_text())


def test_bad_image_does_not_abort_run(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["good.png"])
    (d / "bad.png").write_bytes(b"not an image")
    fake_vllm()
    run_pipeline(d, tmp_path / "out", platform="linux-rocm",
                 cli={"backend": "vllm", "weights_dir": "fake"})
    rs = _rs(tmp_path / "out")
    assert rs["count"] == 2 and rs["ok"] == 1 and rs["fail"] == 1


def test_empty_output_recorded_as_failed(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["p.png"])
    fake_vllm(texts=["   "])
    run_pipeline(d, tmp_path / "out", platform="linux-rocm",
                 cli={"backend": "vllm", "weights_dir": "fake"})
    rs = _rs(tmp_path / "out")
    assert rs["fail"] == 1 and rs["ok"] == 0


def test_stem_collision_rejected(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["page.png"])
    (d / "page.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # same stem, different ext
    fake_vllm()
    with pytest.raises(SystemExit):
        run_pipeline(d, tmp_path / "out", platform="linux-rocm",
                     cli={"backend": "vllm", "weights_dir": "fake"})


def test_atomic_write(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["p.png"])
    fake_vllm(texts=["# p\n\nhello"])
    run_pipeline(d, tmp_path / "out", platform="linux-rocm",
                 cli={"backend": "vllm", "weights_dir": "fake"})
    assert (tmp_path / "out" / "p.md").read_text() == "# p\n\nhello"
    assert not (tmp_path / "out" / "p.md.tmp").exists()


def test_batch_failure_localizes_page(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["a.png", "b.png"])
    fake_vllm(batch_fail=True)
    run_pipeline(d, tmp_path / "out", platform="linux-rocm",
                 cli={"backend": "vllm", "weights_dir": "fake", "batch_size": 8})
    rs = _rs(tmp_path / "out")
    assert rs["fail"] == 2 and rs["ok"] == 0


def test_resume_is_fingerprint_safe(tmp_path, fake_vllm):
    """D5: resume reuses outputs only under a matching fingerprint; a config
    change (max_tokens) invalidates it and forces a re-run (no stale reuse)."""
    d = _make_imgs(tmp_path, ["a.png", "b.png"])
    out = tmp_path / "out"
    fake_vllm()
    # 1st run: generate both, write fingerprint
    run_pipeline(d, out, platform="linux-rocm",
                 cli={"backend": "vllm", "weights_dir": "fake"})
    assert _rs(out)["ok"] == 2
    # 2nd run, resume, SAME config -> both skipped, nothing regenerated
    run_pipeline(d, out, platform="linux-rocm",
                 cli={"backend": "vllm", "weights_dir": "fake", "resume": True})
    perf2 = json.loads((out / "_performance.json").read_text())
    assert perf2["generated_pages"] == 0 and perf2["skipped_pages"] == 2
    # 3rd run, resume, CHANGED config -> fingerprint mismatch -> re-run both
    run_pipeline(d, out, platform="linux-rocm",
                 cli={"backend": "vllm", "weights_dir": "fake", "resume": True, "max_tokens": 9999})
    perf3 = json.loads((out / "_performance.json").read_text())
    assert perf3["generated_pages"] == 2 and perf3["skipped_pages"] == 0


def test_stale_markdown_removed_on_failure(tmp_path, fake_vllm):
    """D5: when a page fails on re-run, a previously-good .md is deleted so it is
    not mistaken for a success."""
    d = _make_imgs(tmp_path, ["a.png"])
    out = tmp_path / "out"
    fake_vllm(texts=["good output"])
    run_pipeline(d, out, platform="linux-rocm", cli={"backend": "vllm", "weights_dir": "f"})
    assert (out / "a.md").exists()
    # now make the same page fail, without resume (fresh run)
    fake_vllm(batch_fail=True)
    run_pipeline(d, out, platform="linux-rocm", cli={"backend": "vllm", "weights_dir": "f"})
    assert not (out / "a.md").exists()  # stale success cleaned up
    assert _rs(out)["fail"] == 1
