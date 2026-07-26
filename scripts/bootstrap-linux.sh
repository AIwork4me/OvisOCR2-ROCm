#!/usr/bin/env bash
# OvisOCR2-ROCm — idempotent Linux/ROCm provisioning. Non-destructive. Pinned.
#
# Provisions (1) a qwen3_5-capable vLLM venv and (2) the OvisOCR2 weights at a
# pinned revision. Override locations with the variables below. Re-runnable.
set -euo pipefail

VENV="${VENV:-/root/venvs/vllm-0221b}"
WEIGHTS="${OVISOCR2_WEIGHTS:-/root/models/OvisOCR2}"
INSTALLER_TAG="${INSTALLER_TAG:-v1.0.0}"
INSTALLER_DIR="${INSTALLER_DIR:-/root/src/rocm-vllm-installer}"
WEIGHTS_REVISION="${WEIGHTS_REVISION:-65c619d374b55d4152e85150fc1b003700bc1f0c}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "[bootstrap] VENV=$VENV  WEIGHTS=$WEIGHTS  INSTALLER_TAG=$INSTALLER_TAG"

# 1. qwen3_5-capable vLLM venv (built locally from a pinned installer checkout,
#    NOT curl|bash — the installer needs its patches/ dir).
if [ ! -x "$VENV/bin/python" ]; then
  echo "[bootstrap] building vLLM 0.22.1 ROCm venv at $VENV (~1-2 h)"
  if [ ! -d "$INSTALLER_DIR/.git" ]; then
    git clone https://github.com/AIwork4me/rocm-vllm-installer.git "$INSTALLER_DIR"
  fi
  ( cd "$INSTALLER_DIR" && git fetch --tags && git checkout "$INSTALLER_TAG" \
    && VENV="$VENV" VLLM_VERSION=v0.22.1 bash install.sh )
else
  echo "[bootstrap] venv exists: $VENV"
fi

"$VENV/bin/python" "$REPO/scripts/check-environment.py" --weights "$WEIGHTS"

# 2. Weights at a pinned revision (not "latest").
if [ ! -f "$WEIGHTS/model.safetensors" ]; then
  echo "[bootstrap] pinning weights revision $WEIGHTS_REVISION -> $WEIGHTS"
  "$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("ATH-MaaS/OvisOCR2", revision="$WEIGHTS_REVISION", local_dir="$WEIGHTS")
PY
fi

"$VENV/bin/python" "$REPO/scripts/verify-reproduction-inputs.py" --weights "$WEIGHTS" || true
echo "[bootstrap] done. Run the adapter with: $VENV/bin/python adapter/run_adapter.py ..."
