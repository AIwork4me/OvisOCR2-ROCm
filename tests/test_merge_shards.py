"""merge-shards validation tests (P0-4): coverage, missing, conflict, stable order.

Runs the single pipeline for two shards, then exercises ``scripts/merge-shards.py``
as a subprocess (the merge lives in ``ovisocr2_rocm.sharding``).
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image

from ovisocr2_rocm.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[1]
MERGE = REPO / "scripts" / "merge-shards.py"


def _imgs(tmp, names):
    d = tmp / "imgs"
    d.mkdir(exist_ok=True)
    for n in names:
        Image.new("RGB", (8, 8)).save(d / n)
    return d


def _run_two_shards(tmp, fake_vllm, names):
    fake_vllm()
    d = _imgs(tmp, names)
    for idx in (0, 1):
        run_pipeline(d, tmp / "preds", platform="linux-rocm",
                     cli={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "f",
                          "num_shards": 2, "shard_index": idx})
    return d


def _merge(tmp, input_root, expected_dir):
    return subprocess.run(
        [sys.executable, str(MERGE), "--input-root", str(input_root),
         "--out-dir", str(tmp / "merged"), "--expected-images", str(expected_dir)],
        capture_output=True, text=True, check=False)


def test_two_shards_merge_clean(tmp_path, fake_vllm):
    names = [f"p_{i:02d}.png" for i in range(6)]
    d = _run_two_shards(tmp_path, fake_vllm, names)
    r = _merge(tmp_path, tmp_path / "preds", d)
    assert r.returncode == 0, r.stderr
    merged = sorted((tmp_path / "merged").glob("*.md"))
    assert len(merged) == 6
    assert [p.stem for p in merged] == sorted(p.stem for p in d.glob("*.png"))


def test_merge_detects_missing_page(tmp_path, fake_vllm):
    names = [f"p_{i:02d}.png" for i in range(4)]
    d = _run_two_shards(tmp_path, fake_vllm, names)
    Image.new("RGB", (8, 8)).save(d / "p_04.png")  # expected but never generated
    r = _merge(tmp_path, tmp_path / "preds", d)
    assert r.returncode != 0 and "p_04" in r.stderr and "missing" in r.stderr.lower()


def test_merge_detects_conflict(tmp_path, fake_vllm):
    names = [f"p_{i:02d}.png" for i in range(4)]
    d = _run_two_shards(tmp_path, fake_vllm, names)
    # p_00 lives in shard 0; inject a different-bytes p_00 into shard 1 -> conflict
    (tmp_path / "preds" / "shard-00001-of-00002" / "p_00.md").write_text("DIFFERENT")
    r = _merge(tmp_path, tmp_path / "preds", d)
    assert r.returncode != 0 and "conflict" in r.stderr.lower()
