"""vllm-backend failure/recovery tests (P0-3/6) via the fake_vllm fixture (CPU)."""
import json

import pytest
from PIL import Image

import adapter.run_adapter as R


def _make_imgs(tmp_path, names):
    d = tmp_path / "imgs"
    d.mkdir()
    for n in names:
        Image.new("RGB", (8, 8), "white").save(d / n)
    return d


def test_bad_image_does_not_abort_run(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["good.png"])
    (d / "bad.png").write_bytes(b"not an image")
    fake_vllm()
    R.run_adapter(d, tmp_path / "out", platform="linux-rocm",
                  config={"backend": "vllm", "weights_dir": "fake"})
    rs = json.loads((tmp_path / "out" / "_run_stats.json").read_text())
    assert rs["count"] == 2 and rs["ok"] == 1 and rs["fail"] == 1


def test_empty_output_recorded_as_failed(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["p.png"])
    fake_vllm(texts=["   "])
    R.run_adapter(d, tmp_path / "out", platform="linux-rocm",
                  config={"backend": "vllm", "weights_dir": "fake"})
    rs = json.loads((tmp_path / "out" / "_run_stats.json").read_text())
    assert rs["fail"] == 1 and rs["ok"] == 0


def test_skip_existing_excluded_from_throughput(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["a.png", "b.png"])
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.md").write_text("existing")  # pre-existing -> skipped
    fake_vllm()
    R.run_adapter(d, out, platform="linux-rocm",
                  config={"backend": "vllm", "weights_dir": "fake", "skip_existing": True})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["ok"] == 2  # a (skipped) + b (generated), both status=ok for conformance
    skipped = next(s for s in rs["stats"] if s["image"] == "a.png")
    assert skipped["seconds"] is None and skipped["attempts"] == 0
    perf = json.loads((out / "_performance.json").read_text())
    assert perf["generated_pages"] == 1 and perf["skipped_pages"] == 1


def test_stem_collision_rejected(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["page.png"])
    (d / "page.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # same stem, different ext
    fake_vllm()
    with pytest.raises(SystemExit):
        R.run_adapter(d, tmp_path / "out", platform="linux-rocm",
                      config={"backend": "vllm", "weights_dir": "fake"})


def test_atomic_write(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["p.png"])
    fake_vllm(texts=["# p\n\nhello"])
    R.run_adapter(d, tmp_path / "out", platform="linux-rocm",
                  config={"backend": "vllm", "weights_dir": "fake"})
    assert (tmp_path / "out" / "p.md").read_text() == "# p\n\nhello"
    assert not (tmp_path / "out" / "p.md.tmp").exists()


def test_batch_failure_localizes_page(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["a.png", "b.png"])
    fake_vllm(batch_fail=True)
    R.run_adapter(d, tmp_path / "out", platform="linux-rocm",
                  config={"backend": "vllm", "weights_dir": "fake", "batch_size": 8})
    rs = json.loads((tmp_path / "out" / "_run_stats.json").read_text())
    assert rs["fail"] == 2 and rs["ok"] == 0
