#!/usr/bin/env bash
# OvisOCR2-ROCm — Linux/ROCm provisioning (thin wrapper over bootstrap-linux.sh).
# Override the venv with: VENV=/your/path bash adapter/setup/00-install-deps.sh
set -euo pipefail
export VENV="${VENV:-/root/venvs/vllm-0221b}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
echo "[00-install-deps] VENV=$VENV -> scripts/bootstrap-linux.sh"
exec bash "$REPO/scripts/bootstrap-linux.sh"
