#!/usr/bin/env bash
# One-command smoke demo for OvisOCR2-ROCm.
# Runs the no-GPU `smoke` backend against examples/demo.png.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/imgs"
cp "$HERE/demo.png" "$TMP/imgs/"
python "$REPO/adapter/run_adapter.py" --img-dir "$TMP/imgs" --out-dir "$TMP/out" \
    --platform linux-rocm --backend smoke
echo "--- demo output ---"
cat "$TMP/out"/*.md
