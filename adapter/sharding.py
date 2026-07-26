"""Deterministic sharding + shard-merge validation (shared by adapter + CLI)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1 (got {num_shards})")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"need 0 <= shard_index < num_shards (got {shard_index}/{num_shards})")


def select_shard(images: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    """Deterministic slice: ``sorted(images)[shard_index::num_shards]``."""
    validate_shard_args(num_shards, shard_index)
    return sorted(images)[shard_index::num_shards]


def shard_dir(out_dir: Path, num_shards: int, shard_index: int) -> Path:
    """Multi-shard runs write to a per-shard subdir; single-shard writes to out_dir."""
    if num_shards == 1:
        return Path(out_dir)
    return Path(out_dir) / f"shard-{shard_index:05d}-of-{num_shards:05d}"


@dataclass
class MergeReport:
    ok: bool = True
    merged_dir: Path | None = None
    page_count: int = 0
    missing: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    failed_pages: list[str] = field(default_factory=list)
    shard_configs: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str):
        self.errors.append(msg)
        self.ok = False


def merge_shards(input_root: Path, expected_images: list[Path], out_dir: Path) -> MergeReport:
    """Merge per-shard outputs into ``out_dir``. Never last-write-wins on conflict.

    Validates: no missing pages, no unexpected pages, no content conflicts, no
    failed pages in shards. Returns a MergeReport (``.ok`` False on any problem).
    """
    input_root = Path(input_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = MergeReport(merged_dir=out_dir)

    shard_dirs = sorted(
        p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("shard-"))
    if not shard_dirs:  # fall back to single-dir layout
        shard_dirs = [input_root]

    expected_stems = sorted(p.stem for p in expected_images)
    seen: dict[str, Path] = {}
    for sd in shard_dirs:
        stats_p = sd / "_run_stats.json"
        if stats_p.exists():
            try:
                rs = json.loads(stats_p.read_text(encoding="utf-8"))
                rep.shard_configs.append(
                    {k: rs.get(k) for k in ("engine", "count", "ok", "fail", "fallback")})
                for pg in rs.get("stats", []):
                    if str(pg.get("status", "")).startswith("failed"):
                        rep.failed_pages.append(pg.get("image", "?"))
            except Exception as e:  # noqa: BLE001 -- report, don't crash merge
                rep.fail(f"unreadable {stats_p}: {e}")
        for md in sorted(sd.glob("*.md")):
            stem = md.stem
            content = md.read_bytes()
            if stem in seen:
                if seen[stem].read_bytes() != content:
                    rep.conflicts.append(stem)
                else:
                    rep.duplicates.append(stem)
                continue
            seen[stem] = md
            target = out_dir / md.name
            tmp = target.with_suffix(".md.tmp")
            tmp.write_bytes(content)
            tmp.replace(target)
            rep.page_count += 1

    got = sorted(seen)
    rep.missing = sorted(set(expected_stems) - set(got))
    extra = sorted(set(got) - set(expected_stems))
    if rep.missing:
        rep.fail(f"missing pages: {rep.missing[:10]}{' ...' if len(rep.missing) > 10 else ''}")
    if extra:
        rep.fail(f"unexpected pages not in expected set: {extra[:10]}")
    if rep.conflicts:
        rep.fail(f"content conflicts (different bytes, same stem): {rep.conflicts}")
    if rep.failed_pages:
        rep.fail(f"failed pages present in shards: {rep.failed_pages[:10]}")
    return rep
