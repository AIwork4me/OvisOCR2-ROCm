"""OvisOCR2-ROCm adapter — implements the omnidocbench-rocm contract.

Contract notes
--------------
* The engine invokes this as a subprocess and consumes only
  ``out_dir/<image_stem>.md`` + ``_run_stats.json`` (R1).
* Per-page failures are caught and recorded; a real run never raises (R2).
  Hard config errors (bad backend, stem collision, empty input on vllm) raise
  up front, before any inference.
* One UTF-8 ``.md`` per page image, named by stem (R3); written atomically.
* ``backend == "smoke"`` is a no-GPU placeholder (CI gate) and never imports vllm.
* ``--skip-existing`` resumes: existing non-empty ``.md`` are skipped (counted as
  ``ok`` with ``seconds=null`` so they don't pollute throughput; the generated vs
  skipped split is recorded in the sidecar).
* Multi-shard runs write per-shard subdirs ``shard-NNNNN-of-NNNNN/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from omnidocbench_rocm.types import PageStatus, RunSummary

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATFORMS = ("linux-rocm", "windows-hip")

# ---- Official OvisOCR2 prompt (upstream model card, verbatim) -----------------
_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    "represent them using an HTML image tag: <"
    'img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)


def _load_helpers():
    """Import config/postprocess/sharding whether run as a package module or a script."""
    try:
        from . import adapter_config, postprocess, sharding
    except ImportError:
        here = Path(__file__).resolve().parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        import adapter_config  # type: ignore
        import postprocess  # type: ignore
        import sharding  # type: ignore
    return adapter_config, postprocess, sharding


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with open(tmp, "ab") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---- vLLM (lazy: keeps `smoke` importable without vllm/GPU) ------------------
_LLM = None
_CHAT = None


def _get_llm(cfg):
    global _LLM, _CHAT
    if _LLM is not None:
        return _LLM, _CHAT
    from vllm import LLM

    weights = cfg.weights_dir or os.environ.get("OVISOCR2_WEIGHTS") or "ATH-MaaS/OvisOCR2"
    kwargs = {
        "model": weights,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "max_model_len": cfg.max_model_len,
        "trust_remote_code": cfg.trust_remote_code,
        "enforce_eager": cfg.enforce_eager,
        "limit_mm_per_prompt": {"image": 1},
    }
    try:
        _LLM = LLM(gdn_prefill_backend=cfg.gdn_prefill_backend, **kwargs)
    except TypeError:
        _LLM = LLM(**kwargs)
    _CHAT = _LLM.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _LLM, _CHAT


def _build_input(chat, image_path, min_px, max_px):
    from PIL import Image

    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    return {
        "prompt": chat,
        "multi_modal_data": {"image": rgb},
        "mm_processor_kwargs": {"images_kwargs": {"min_pixels": min_px, "max_pixels": max_px}},
    }


def _run_batch(llm, chat, sp, postprocess_fn, images, min_px, max_px):
    """Run one batch; on a whole-batch failure, bisect to single pages.

    Returns a list of ``(image, text|None, seconds|None, error|None)``. Never raises.
    ``seconds`` is honest only for single-page batches (``len==1``); otherwise None
    because vLLM schedules a batch internally and per-page latency is not measurable.
    """
    try:
        inputs = [_build_input(chat, i, min_px, max_px) for i in images]
        t0 = time.time()
        outputs = llm.generate(inputs, sp)
        wall = time.time() - t0
    except Exception as e:  # noqa: BLE001 -- whole-batch failure (OOM/bad image/generate error)
        if len(images) == 1:
            return [(images[0], None, None, f"batch failed: {e}")]
        mid = len(images) // 2
        return (_run_batch(llm, chat, sp, postprocess_fn, images[:mid], min_px, max_px)
                + _run_batch(llm, chat, sp, postprocess_fn, images[mid:], min_px, max_px))
    share = wall / len(images) if images else 0.0
    out_list = []
    for i, out in zip(images, outputs, strict=False):
        try:
            text = postprocess_fn(out.outputs[0].text.strip())
            if not text.strip():
                out_list.append((i, None, None, "empty prediction"))
            else:
                out_list.append((i, text, share if len(images) == 1 else None, None))
        except Exception as e:  # noqa: BLE001 -- isolate per-page postprocess failure
            out_list.append((i, None, None, str(e)))
    return out_list


def _collect_env():
    """Best-effort runtime fingerprint; every field None if unavailable."""
    import subprocess

    out = {
        "output_tokens": None, "tokens_per_second": None,
        "max_memory_allocated_mb": None, "max_memory_reserved_mb": None,
        "gpu": None, "gfx_arch": None, "torch_version": None, "hip_version": None,
        "vllm_version": None, "transformers_version": None,
        "git_commit": None, "git_dirty": None, "weights_revision": None,
    }
    with suppress(Exception):
        import torch

        out["torch_version"] = torch.__version__
        out["hip_version"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            out["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            out["max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
            out["gpu"] = torch.cuda.get_device_name(0)
    for modname, key in (("vllm", "vllm_version"), ("transformers", "transformers_version")):
        with suppress(Exception):
            out[key] = __import__(modname).__version__
    with suppress(Exception):
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False, cwd=_repo_root())
        out["git_commit"] = rev.stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               check=False, cwd=_repo_root())
        out["git_dirty"] = bool(dirty.stdout.strip())
    return out


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _write_sidecars(target_dir, cfg, started, model_load_seconds, infer_wall,
                    generated, skipped, failed, platform, pages_jsonl, adapter_config):
    completed = datetime.now(UTC)
    total = (completed - started).total_seconds()
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
        **_collect_env(),
        "config_snapshot": adapter_config.config_snapshot(cfg),
    }
    (target_dir / "_performance.json").write_text(
        json.dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_dir / "_pages.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pages_jsonl),
        encoding="utf-8",
    )


def run_adapter(img_dir, out_dir, *, platform, config) -> dict:
    adapter_config, postprocess, sharding = _load_helpers()
    cfg = adapter_config.resolve({**config, "platform": platform})
    out_dir = Path(out_dir)
    target_dir = sharding.shard_dir(out_dir, cfg.num_shards, cfg.shard_index)
    target_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if cfg.limit_pages is not None:
        all_imgs = all_imgs[: cfg.limit_pages]
    imgs = sharding.select_shard(all_imgs, cfg.num_shards, cfg.shard_index)

    # Stem-collision precheck (hard error before any inference).
    seen: dict[str, Path] = {}
    for i in imgs:
        if i.stem in seen:
            raise SystemExit(
                f"stem collision: {i.name} and {seen[i.stem].name} both map to {i.stem}.md; "
                "refusing to overwrite")
        seen[i.stem] = i

    stats: list[PageStatus] = []
    pages_jsonl = []
    generated = skipped = failed = 0
    infer_wall = 0.0
    started = datetime.now(UTC)

    if cfg.backend == "smoke":
        for i in imgs:
            _atomic_write_text(target_dir / f"{i.stem}.md",
                               f"# {i.stem}\n\n(smoke output — backend=smoke)\n")
            stats.append(PageStatus(i.name, "ok", seconds=None, attempts=0))
            pages_jsonl.append({"image": i.name, "status": "ok", "shard_index": cfg.shard_index})
    else:
        if not imgs:
            raise SystemExit(
                "vllm backend received an empty image set — a real eval with 0 pages is a "
                "configuration error (smoke may return 0 pages; vllm may not).")
        from vllm import SamplingParams  # lazy

        llm, chat = _get_llm(cfg)
        model_load_seconds = (datetime.now(UTC) - started).total_seconds()
        sp = SamplingParams(max_tokens=cfg.max_tokens, temperature=cfg.temperature)
        todo = []
        for i in imgs:
            target = target_dir / f"{i.stem}.md"
            if cfg.skip_existing and target.exists() and target.read_text(encoding="utf-8").strip():
                stats.append(PageStatus(i.name, "ok", seconds=None, attempts=0))
                pages_jsonl.append({"image": i.name, "status": "skipped", "shard_index": cfg.shard_index})
                skipped += 1
            else:
                todo.append(i)
        for start in range(0, len(todo), cfg.batch_size):
            batch = todo[start:start + cfg.batch_size]
            bt0 = time.time()
            for img, text, seconds, error in _run_batch(
                    llm, chat, sp, postprocess.postprocess, batch, cfg.min_pixels, cfg.max_pixels):
                if error is not None:
                    stats.append(PageStatus(img.name, f"failed: {error}", error=error))
                    pages_jsonl.append({"image": img.name, "status": "failed",
                                        "shard_index": cfg.shard_index, "error": error})
                    failed += 1
                else:
                    _atomic_write_text(target_dir / f"{img.stem}.md", text)
                    stats.append(PageStatus(img.name, "ok", seconds=seconds, attempts=1))
                    pages_jsonl.append({"image": img.name, "status": "generated",
                                        "shard_index": cfg.shard_index, "seconds": seconds,
                                        "output_bytes": len(text.encode("utf-8"))})
                    generated += 1
            infer_wall += time.time() - bt0
            done = start + len(batch)
            print(f"[ovisocr2] batch {start // cfg.batch_size + 1}: {done}/{len(todo)} done "
                  f"(skipped {skipped}, failed {failed})", file=sys.stderr)
        _write_sidecars(target_dir, cfg, started, model_load_seconds, infer_wall,
                        generated, skipped, failed, platform, pages_jsonl, adapter_config)

    rs = RunSummary(
        len(imgs),
        sum(1 for s in stats if s.status == "ok"),
        sum(1 for s in stats if s.status.startswith("failed")),
        sum(1 for s in stats if s.status.startswith("fallback")),
        cfg.limit_pages,
        stats,
        engine=cfg.backend,
    )
    rs.write(target_dir / "_run_stats.json")
    return rs.to_run_stats()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OvisOCR2-ROCm OmniDocBench adapter")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default=None, help="smoke | vllm (default: vllm via config/env)")
    p.add_argument("--server-url", default=None)
    p.add_argument("--api-model-name", default=None)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--limit-pages", type=int, default=None)
    a = p.parse_args(argv)
    run_adapter(
        Path(a.img_dir),
        Path(a.out_dir),
        platform=a.platform,
        config={
            "backend": a.backend,
            "server_url": a.server_url,
            "api_model_name": a.api_model_name,
            "skip_existing": a.skip_existing,
            "batch_size": a.batch_size,
            "num_shards": a.num_shards,
            "shard_index": a.shard_index,
            "limit_pages": a.limit_pages,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
