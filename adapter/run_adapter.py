"""Thin adapter-script entrypoint (R1/R2/R3) — delegates to the single pipeline.

The engine / cli_bridge subprocess this and consume only ``out_dir/<stem>.md`` +
``_run_stats.json`` (R1). All inference lives in :mod:`ovisocr2_rocm.pipeline`;
there is NO second inference implementation. ``--skip-existing`` is kept as a
deprecated alias of ``--resume`` (now fingerprint-gated, no longer unsafe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: let `python adapter/run_adapter.py` work from a bare checkout too.
try:
    import ovisocr2_rocm  # noqa: F401
except ImportError:
    _src = Path(__file__).resolve().parents[1] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

PLATFORMS = ("linux-rocm", "windows-hip")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OvisOCR2-ROCm OmniDocBench adapter")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default=None, help="smoke | vllm (default: vllm via config/env)")
    p.add_argument("--server-url", default=None,
                   help="REJECTED for the in-process vllm backend (errors if set)")
    p.add_argument("--api-model-name", default=None)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=None,
                   help="deprecated alias of --resume (now fingerprint-gated)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None,
                   help="fingerprint-safe resume (reuses outputs only when config matches)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--limit-pages", type=int, default=None)
    a = p.parse_args(argv)

    from ovisocr2_rocm.pipeline import run_pipeline

    run_pipeline(
        Path(a.img_dir), Path(a.out_dir), platform=a.platform, cli={
            "backend": a.backend,
            "server_url": a.server_url,
            "api_model_name": a.api_model_name,
            "skip_existing": a.skip_existing,
            "resume": a.resume,
            "batch_size": a.batch_size,
            "num_shards": a.num_shards,
            "shard_index": a.shard_index,
            "limit_pages": a.limit_pages,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
