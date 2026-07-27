"""Standard CLI for OvisOCR2-ROCm (ADR-0011 contract).

Four JSON subcommands with a fixed exit-code scheme. stdout is PURE JSON (logs go
to stderr); the package imports no model runtime, and ``parse`` imports the
pipeline lazily so ``version``/``capabilities``/``doctor`` run with no GPU/deps.

    ovisocr2-rocm version                              # identity (cli_version)
    ovisocr2-rocm capabilities                         # declared platforms/backends
    ovisocr2-rocm doctor                               # readiness (ready|not-ready)
    ovisocr2-rocm parse --img-dir D --out-dir O ...    # -> cli_result + _run_stats.json

Exit codes: 0 OK, 1 PARTIAL, 2 USAGE, 3 BACKEND_MISMATCH, 4 CONTRACT, 5 FATAL.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import __version__
from . import capabilities as _caps
from . import doctor as _doctor

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2
EXIT_BACKEND_MISMATCH = 3
EXIT_CONTRACT = 4
EXIT_FATAL = 5


def _emit(obj: dict, exit_code: int = EXIT_OK) -> int:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()
    return exit_code


def _cmd_version(args) -> int:
    return _emit({"name": "ovisocr2-rocm", "version": __version__,
                  "engine_version": __version__, "schema_version": 1})


def _cmd_capabilities(args) -> int:
    return _emit(_caps.capabilities())


def _cmd_doctor(args) -> int:
    return _emit(_doctor.check())


def _cmd_parse(args) -> int:
    if not Path(args.img_dir).is_dir():
        return _emit({"schema_version": 1, "status": "failed", "pages": [],
                      "error": f"img-dir not found: {args.img_dir}"}, EXIT_USAGE)
    try:
        # lazy: importing the pipeline must not happen at module import time
        from .pipeline import run_pipeline
        result = run_pipeline(
            args.img_dir, args.out_dir, platform=args.platform, cli={
                "backend": args.backend,
                "server_url": args.server_url,
                "api_model_name": args.api_model_name,
                "resume": args.resume,
                "skip_existing": args.skip_existing,
                "batch_size": args.batch_size,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "limit_pages": args.limit_pages,
            },
        )
    except SystemExit as e:
        # config validation / hard preconditions -> USAGE-level failure, never silent
        msg = str(e) or "usage error"
        return _emit({"schema_version": 1, "status": "failed", "pages": [],
                      "error": msg}, EXIT_USAGE)
    except Exception as e:  # noqa: BLE001 -- fatal: never fake success
        traceback.print_exc(file=sys.stderr)
        return _emit({"schema_version": 1, "status": "failed", "pages": [],
                      "error": f"fatal: {e}"}, EXIT_FATAL)

    status = result.get("status")
    if status == "partial":
        return _emit(result, EXIT_PARTIAL)
    if status == "failed":
        return _emit(result, EXIT_FATAL)
    return _emit(result, EXIT_OK)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ovisocr2-rocm",
                                description="OvisOCR2-ROCm standard CLI (ADR-0011).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version").add_argument("--json", dest="json", action="store_true")

    sub.add_parser("capabilities").add_argument("--json", dest="json", action="store_true")

    sub.add_parser("doctor").add_argument("--json", dest="json", action="store_true")

    pr = sub.add_parser("parse")
    pr.add_argument("--img-dir", required=True)
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--platform", default="linux-rocm", choices=("linux-rocm", "windows-hip"))
    pr.add_argument("--backend", default=None, help="smoke | vllm (default: vllm via config/env)")
    pr.add_argument("--benchmark", default="omnidocbench-v16",
                    help="benchmark tag (informational; only omnidocbench-v16 is supported)")
    pr.add_argument("--server-url", default=None,
                    help="REJECTED for the in-process vllm backend (errors if set)")
    pr.add_argument("--api-model-name", default=None)
    pr.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None,
                    help="fingerprint-safe resume (reuses outputs only when config matches)")
    pr.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=None,
                    help="deprecated alias of --resume (now fingerprint-gated)")
    pr.add_argument("--batch-size", type=int, default=None)
    pr.add_argument("--num-shards", type=int, default=None)
    pr.add_argument("--shard-index", type=int, default=None)
    pr.add_argument("--limit-pages", type=int, default=None)
    pr.add_argument("--json", dest="json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd == "version":
        return _cmd_version(a)
    if a.cmd == "capabilities":
        return _cmd_capabilities(a)
    if a.cmd == "doctor":
        return _cmd_doctor(a)
    if a.cmd == "parse":
        return _cmd_parse(a)
    p.print_help(sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
