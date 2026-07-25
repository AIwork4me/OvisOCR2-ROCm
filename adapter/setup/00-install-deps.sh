#!/usr/bin/env bash
# OvisOCR2-ROCm — Linux/ROCm provisioning.
# Idempotent: safe to re-run. Provisions (1) a qwen3_5-capable ROCm vLLM venv and
# (2) the OvisOCR2 weights. The adapter loads weights in-process via vLLM.
set -euo pipefail

VENV="${VENV:-/root/venvs/vllm-build-gfx110x}"
WEIGHTS="${OVISOCR2_WEIGHTS:-/root/models/OvisOCR2}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[00-install-deps] OvisOCR2-ROCm provisioning (linux-rocm, backend=vllm)"

# 1. ROCm vLLM venv with qwen3_5 support -------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[00-install-deps] $VENV not found."
  echo "  Build it once with rocm-vllm-installer (clones vLLM v0.19.0 + ROCm patches):"
  echo "    bash <(curl -sSL https://raw.githubusercontent.com/AIwork4me/rocm-vllm-installer/main/install.sh)"
  echo "  Or set VENV=/path/to/your/qwen3_5-capable vLLM venv."
  exit 1
fi
if "$VENV/bin/python" -c "from vllm.model_executor.models.registry import ModelRegistry as m; \
    assert 'Qwen3_5ForConditionalGeneration' in m.get_supported_archs()" 2>/dev/null; then
  echo "[00-install-deps] OK: $VENV has qwen3_5 support"
else
  echo "[00-install-deps] FAIL: $VENV vLLM lacks Qwen3_5ForConditionalGeneration. Rebuild via rocm-vllm-installer."
  exit 1
fi

# 2. Weights -----------------------------------------------------------------
if [ -f "$WEIGHTS/model.safetensors" ]; then
  echo "[00-install-deps] OK: weights at $WEIGHTS"
else
  echo "[00-install-deps] downloading ATH-MaaS/OvisOCR2 -> $WEIGHTS (~1.7 GB)"
  "$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("ATH-MaaS/OvisOCR2", local_dir="$WEIGHTS")
PY
fi

echo "[00-install-deps] export OVISOCR2_WEIGHTS=$WEIGHTS"
echo "[00-install-deps] use $VENV/bin/python to run the adapter"
echo "[00-install-deps] done"
