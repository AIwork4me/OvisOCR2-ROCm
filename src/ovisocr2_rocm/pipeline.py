"""The single inference pipeline (shared by adapter + standard CLI).

Both ``adapter/run_adapter.py`` and ``ovisocr2_rocm.cli`` reach the model through
:func:`run_pipeline` — there is no second inference implementation.

Guarantees (R1/R2/R3 + D5):
  * R1 — heavy runtime (vllm/torch/PIL) is imported only inside the backend's
    ``load``/``infer``, never when this module is imported.
  * R2 — per-page failure is recorded; the run continues and never raises.
  * R3 — one ``<stem>.md`` per page, written atomically.
  * D5-resume — resume is FINGERPRINT-gated: a stored ``.md`` is reused only when
    the output-affecting config (weights/params/prompt) matches the stored
    fingerprint. A mismatch re-runs the page (never silently reuses stale output).
  * D5-stale — when a page's inference fails, any pre-existing ``<stem>.md`` is
    deleted, so a stale good-looking output is never mistaken for a success.
  * page-conservation — every selected image gets a status; ``count`` equals the
    selected image count; failed pages never disappear.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import outputs, sharding
from .outputs import PageStatus
from .runtime import select as select_backend

IMG_EXT = sharding.IMG_EXT


def load_prompt() -> str:
    p = Path(__file__).resolve().parent / "resources" / "prompts" / "ovisocr2-v1.txt"
    return p.read_text(encoding="utf-8")


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _read_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fingerprint")
    except Exception:  # noqa: BLE001
        return None


def _stem_collision_check(imgs: list[Path]) -> None:
    seen: dict[str, Path] = {}
    for i in imgs:
        if i.stem in seen:
            raise SystemExit(
                f"stem collision: {i.name} and {seen[i.stem].name} both map to {i.stem}.md; "
                "refusing to overwrite")
        seen[i.stem] = i


def run_pipeline(img_dir, out_dir, *, platform: str, cli: dict | None = None) -> dict:
    """Run inference over ``img_dir`` -> ``out_dir``. Returns the cli_result dict.

    ``cli`` carries the override values (None when unset), merged with platform.
    Writes ``_run_stats.json`` (+ ``_performance.json`` / ``_pages.jsonl`` for the
    vllm path) into the shard target dir. Never raises per-page (R2); hard config
    errors raise before any inference.
    """
    from .config import config_snapshot, resolve
    from .provenance import run_fingerprint

    cli = cli or {}
    cfg = resolve({**cli, "platform": platform})
    out_dir = Path(out_dir)
    target_dir = sharding.shard_dir(out_dir, cfg.num_shards, cfg.shard_index)
    target_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if cfg.limit_pages is not None:
        all_imgs = all_imgs[: cfg.limit_pages]
    imgs = sharding.select_shard(all_imgs, cfg.num_shards, cfg.shard_index)

    backend = select_backend(cfg.backend)

    if cfg.backend == "vllm" and not imgs:
        raise SystemExit(
            "vllm backend received an empty image set — a real eval with 0 pages is a "
            "configuration error (smoke may return 0 pages; vllm may not).")

    _stem_collision_check(imgs)

    prompt = load_prompt()
    fp = run_fingerprint(prompt_sha256=prompt_sha256(prompt), config_snapshot=config_snapshot(cfg))
    stored_fp = _read_fingerprint(target_dir / "_fingerprint.json")
    resume = bool(cfg.resume or cfg.skip_existing)
    # a stored output is reusable only under a matching fingerprint
    resumable = resume and (stored_fp == fp)

    stats: list[PageStatus] = []
    pages_jsonl: list[dict] = []
    generated = skipped = failed = 0
    infer_wall = 0.0
    started = datetime.now(UTC)

    if cfg.backend == "smoke":
        backend.load(cfg, prompt)
        for i in imgs:
            pr = backend.infer([i], cfg)[0]
            outputs.atomic_write_text(target_dir / f"{i.stem}.md", pr.text or "")
            stats.append(PageStatus(i.name, "ok", seconds=None, attempts=0))
            pages_jsonl.append({"image": i.name, "status": "ok", "shard_index": cfg.shard_index})
            generated += 1
    else:
        backend.load(cfg, prompt)
        model_load_seconds = (datetime.now(UTC) - started).total_seconds()

        todo: list[Path] = []
        for i in imgs:
            target = target_dir / f"{i.stem}.md"
            if resumable and target.exists() and target.read_text(encoding="utf-8").strip():
                stats.append(PageStatus(i.name, "ok", seconds=None, attempts=0))
                pages_jsonl.append({"image": i.name, "status": "skipped", "shard_index": cfg.shard_index})
                skipped += 1
            else:
                # D5-stale: clear any stale output before re-running this page
                if target.exists():
                    outputs.safe_unlink(target)
                todo.append(i)

        bt0 = time.time()
        results = backend.infer(todo, cfg)
        for i, pr in zip(todo, results, strict=True):
            if pr.error is not None:
                # D5-stale: delete any lingering .md so it isn't counted as success
                outputs.safe_unlink(target_dir / f"{i.stem}.md")
                stats.append(PageStatus(i.name, f"failed: {pr.error}", error=pr.error))
                pages_jsonl.append({"image": i.name, "status": "failed",
                                    "shard_index": cfg.shard_index, "error": pr.error})
                failed += 1
            elif not (pr.text or "").strip():
                outputs.safe_unlink(target_dir / f"{i.stem}.md")
                stats.append(PageStatus(i.name, "failed: empty prediction", error="empty prediction"))
                pages_jsonl.append({"image": i.name, "status": "failed",
                                    "shard_index": cfg.shard_index, "error": "empty prediction"})
                failed += 1
            else:
                outputs.atomic_write_text(target_dir / f"{i.stem}.md", pr.text)
                stats.append(PageStatus(i.name, "ok", seconds=pr.seconds, attempts=1))
                pages_jsonl.append({"image": i.name, "status": "generated",
                                    "shard_index": cfg.shard_index, "seconds": pr.seconds,
                                    "output_bytes": len(pr.text.encode("utf-8"))})
                generated += 1
            print(f"[ovisocr2] {generated + skipped}/{len(imgs)} done "
                  f"(skipped {skipped}, failed {failed})", file=sys.stderr)
        infer_wall = time.time() - bt0

        _write_sidecars(target_dir, cfg, started, model_load_seconds, infer_wall,
                        generated, skipped, failed, platform, pages_jsonl,
                        backend.env())

    # persist the fingerprint for future safe-resume
    (target_dir / "_fingerprint.json").write_text(
        json.dumps({"fingerprint": fp, "schema_version": 1}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    backend.close()

    count = len(imgs)
    ok = sum(1 for s in stats if s.status == "ok")
    fail = sum(1 for s in stats if s.status.startswith("failed"))
    fallback = sum(1 for s in stats if s.status.startswith("fallback"))
    env = backend.env() if cfg.backend != "smoke" else backend.env()
    efficiency = None
    if cfg.backend != "smoke" and generated:
        # latency is null unless honestly measurable (single-page batches); we do
        # not synthesize a number. peak_vram/gpu come from the detected env.
        efficiency = {
            "latency_s_per_page": None,
            "peak_vram_mb": env.get("max_memory_reserved_mb"),
            "gpu": env.get("gpu"),
        }

    run_stats = outputs.build_run_stats(
        count=count, ok=ok, fail=fail, fallback=fallback,
        limit_pages=cfg.limit_pages, stats=stats,
        engine=cfg.backend, efficiency=efficiency,
    )
    outputs.write_run_stats(target_dir / "_run_stats.json", run_stats)

    if fail > 0 or fallback > 0:
        status = "partial"
    elif count == 0:
        status = "failed"
    else:
        status = "ok"

    return outputs.build_cli_result(
        status=status, backend=cfg.backend, stats=stats, count=count,
        ok=ok, failed=fail, skipped=fallback, output_dir=str(out_dir),
        full_set=cfg.limit_pages in (None, 0),
    )


def _write_sidecars(target_dir, cfg, started, model_load_seconds, infer_wall,
                    generated, skipped, failed, platform, pages_jsonl, env):
    completed = datetime.now(UTC)
    total = (completed - started).total_seconds()
    from .config import config_snapshot
    perf = {
        "schema_version": 2,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_wall_seconds": round(infer_wall, 3),
        "total_wall_seconds": round(total, 3),
        "generated_pages": generated,
        "skipped_pages": skipped,
        "failed_pages": failed,
        "limit_pages": cfg.limit_pages,
        "mean_latency_s_per_page": round(infer_wall / generated, 3) if generated else None,
        "backend": cfg.backend,
        "platform": platform,
        "num_shards": cfg.num_shards,
        "shard_index": cfg.shard_index,
        **env,
        "config_snapshot": config_snapshot(cfg),
    }
    outputs.atomic_write_text(target_dir / "_performance.json",
                              json.dumps(perf, ensure_ascii=False, indent=2))
    outputs.atomic_write_text(
        target_dir / "_pages.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pages_jsonl),
    )
