"""OvisOCR2-ROCm adapter — implements the omnidocbench-rocm contract.

OvisOCR2 (ATH-MaaS/OvisOCR2, 0.8B, ``model_type: qwen3_5`` — a Qwen3-VL vision
encoder on a Qwen3-Next GDN hybrid backbone) is served **in-process via vLLM**,
exactly as the upstream model card's ``OvisOCR2Parser`` does. This keeps the
adapter's predictions byte-identical to the official recipe (greedy, temp=0,
max_tokens=16384, 448²–2880² pixels, ``_clean_truncated_repeats`` post-processing,
visual-region tags filtered).

Contract notes
--------------
* The engine invokes this as a subprocess and consumes only
  ``out_dir/<image_stem>.md`` + ``out_dir/_run_stats.json`` (R1).
* Per-page failures are caught and recorded; the run never raises (R2).
* One UTF-8 ``.md`` per page image, named by stem (R3).
* ``backend == "smoke"`` is a no-GPU placeholder so the repo is runnable in CI.
* ``--skip-existing`` resumes: pages whose ``.md`` already exists are skipped
  (recorded as ``ok``) so the full set is never reduced.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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


def _load_adapter_config():
    """Import ``adapter_config`` whether run as a package module or a bare script."""
    try:
        from . import adapter_config  # package-relative import
    except ImportError:
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        import adapter_config  # type: ignore[import-not-found]
    return adapter_config


# ---- Upstream post-processing (upstream model card, verbatim) -----------------
def _clean_truncated_repeats(
    text,
    min_text_len=8000,
    max_period=200,
    min_period=1,
    min_repeat_chars=100,
    min_repeat_times=5,
):
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]
    return text


def _postprocess(text: str) -> str:
    """Drop visual-region <img> tags (card default) then clean repeat tails."""
    text = "\n\n".join(
        b
        for b in text.split("\n\n")
        if not b.strip().startswith('<img src="images/bbox_')
    )
    return _clean_truncated_repeats(text)


# ---- Inference (lazy vLLM import so `smoke` runs without a GPU) ---------------
_LLM = None
_CHAT = None


def _get_llm(cfg: dict):
    """Load vLLM once per process, memoised."""
    global _LLM, _CHAT
    if _LLM is not None:
        return _LLM, _CHAT
    import inspect

    from vllm import LLM  # lazy: keeps `smoke` importable without vllm/GPU

    weights = (
        cfg.get("weights_dir")
        or os.environ.get("OVISOCR2_WEIGHTS")
        or "ATH-MaaS/OvisOCR2"
    )
    kwargs = {
        "model": weights,
        "tensor_parallel_size": int(cfg.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.9)),
        "max_model_len": int(cfg.get("max_model_len", 32768)),
        "trust_remote_code": bool(cfg.get("trust_remote_code", True)),
        "enforce_eager": bool(cfg.get("enforce_eager", True)),
        "limit_mm_per_prompt": {"image": 1},
    }
    # vLLM 0.22+ accepts gdn_prefill_backend; the ROCm 0.19 build does not —
    # pass it only if the running vLLM knows the flag.
    if "gdn_prefill_backend" in inspect.signature(LLM.__init__).parameters:
        kwargs["gdn_prefill_backend"] = cfg.get("gdn_prefill_backend", "triton")
    _LLM = LLM(**kwargs)
    _CHAT = _LLM.get_tokenizer().apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _LLM, _CHAT


def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict) -> dict:
    assert platform in PLATFORMS, f"unknown platform: {platform}"
    adapter_config = _load_adapter_config()
    cfg = {**adapter_config.as_dict(), **config}
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = cfg.get("backend", "smoke")
    skip_existing = bool(cfg.get("skip_existing"))

    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    stats: list[PageStatus] = []

    if backend == "smoke":
        # No-GPU placeholder — one valid .md per page (the CI gate).
        for i in imgs:
            (out_dir / f"{i.stem}.md").write_text(
                f"# {i.stem}\n\n(smoke output — backend=smoke)\n", encoding="utf-8"
            )
            stats.append(PageStatus(i.name, "ok", seconds=0.0, attempts=0))
    else:
        from PIL import Image  # lazy
        from vllm import SamplingParams  # lazy

        llm, chat = _get_llm(cfg)
        sp = SamplingParams(
            max_tokens=int(cfg.get("max_tokens", 16384)), temperature=0.0
        )
        min_px = int(cfg.get("min_pixels", 448 * 448))
        max_px = int(cfg.get("max_pixels", 2880 * 2880))

        todo, skipped = [], 0
        for i in imgs:
            target = out_dir / f"{i.stem}.md"
            if (
                skip_existing
                and target.exists()
                and target.read_text(encoding="utf-8").strip()
            ):
                stats.append(PageStatus(i.name, "ok", seconds=0.0, attempts=0))
                skipped += 1
                continue
            todo.append(i)

        t0 = time.time()
        if todo:
            inputs = [
                {
                    "prompt": chat,
                    "multi_modal_data": {"image": Image.open(i).convert("RGB")},
                    "mm_processor_kwargs": {
                        "images_kwargs": {"min_pixels": min_px, "max_pixels": max_px}
                    },
                }
                for i in todo
            ]
            outputs = llm.generate(inputs, sp)
            for i, out in zip(todo, outputs):
                t_page = time.time()
                try:
                    text = _postprocess(out.outputs[0].text.strip())
                    if not text.strip():
                        raise RuntimeError("empty prediction")
                    (out_dir / f"{i.stem}.md").write_text(text, encoding="utf-8")
                    stats.append(
                        PageStatus(
                            i.name, "ok", seconds=time.time() - t_page, attempts=1
                        )
                    )
                except Exception as e:  # noqa: BLE001 -- contract R2: catch every per-page failure, never raise
                    stats.append(PageStatus(i.name, f"failed: {e}", error=str(e)))
            print(
                f"[ovisocr2] generated {len(todo)} pages in {time.time() - t0:.1f}s "
                f"(skipped {skipped} existing)",
                file=sys.stderr,
            )

    rs = RunSummary(
        len(imgs),
        sum(1 for s in stats if s.status == "ok"),
        sum(1 for s in stats if s.status.startswith("failed")),
        sum(1 for s in stats if s.status.startswith("fallback")),
        cfg.get("limit_pages"),
        stats,
        engine=backend,
    )
    rs.write(out_dir / "_run_stats.json")
    return rs.to_run_stats()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OvisOCR2-ROCm OmniDocBench adapter")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default="smoke")
    p.add_argument("--server-url", default="")  # contract parity; in-process ignores it
    p.add_argument("--api-model-name", default="ovisocr2")
    p.add_argument("--skip-existing", action="store_true")
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
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
