#!/usr/bin/env python3
"""Merge deterministic per-shard adapter outputs into one directory, with validation.

Checks: no missing pages, no unexpected pages, no content conflicts (never
last-write-wins), no failed pages in shards. Exits 1 on any problem.

Usage:
    python scripts/merge-shards.py \\
        --input-root predictions/ovisocr2 \\
        --out-dir predictions/ovisocr2-merged \\
        --expected-images "$DATASET/images"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main(argv=None) -> int:
    from adapter.sharding import IMG_EXT, merge_shards

    p = argparse.ArgumentParser(description="Merge sharded adapter outputs with validation")
    p.add_argument("--input-root", required=True,
                   help="dir holding shard-*/ subdirs (or a single output dir)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--expected-images", required=True,
                   help="dir of expected input images (or a text file listing them)")
    a = p.parse_args(argv)

    root = Path(a.expected_images)
    if root.is_dir():
        expected = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXT)
    else:
        expected = sorted(Path(x.strip()) for x in root.read_text().splitlines() if x.strip())

    rep = merge_shards(Path(a.input_root), expected, Path(a.out_dir))
    if rep.ok:
        print(f"[merge] OK: {rep.page_count} pages -> {a.out_dir}")
        return 0
    print("[merge] NON-CONFORMANT:", file=sys.stderr)
    for e in rep.errors:
        print(f"  - {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
