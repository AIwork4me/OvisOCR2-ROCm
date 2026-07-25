"""Adapter configuration for OvisOCR2-ROCm.

OvisOCR2 is served **in-process via vLLM** (matching the upstream model card's
``OvisOCR2Parser``). The adapter loads the weights once and runs batched
``generate()`` over the page set. Override any field via ``adapter/setup/.env.local``
or CLI flags; see ``.env.local.example``.
"""

from __future__ import annotations

import os

# Inference backend. ``smoke`` = no-GPU placeholder (CI gate); ``vllm`` = the real
# in-process vLLM/ROCm path used for all published results.
BACKEND = os.environ.get("OVISOCR2_BACKEND", "vllm")

# In-process adapter ignores the server URL (kept for contract parity).
SERVER_URL = ""

# Model name (informational; the in-process adapter loads weights directly).
API_MODEL_NAME = "ovisocr2"

# Weights: a HuggingFace/ModelScope repo id (vLLM auto-downloads) OR a local path.
# Default is the HF repo id; override with OVISOCR2_WEIGHTS for a local checkout
# (e.g. /root/models/OvisOCR2) or a ModelScope mirror.
WEIGHTS_DIR = os.environ.get("OVISOCR2_WEIGHTS", "ATH-MaaS/OvisOCR2")

# Recipe (upstream OvisOCR2 card, verbatim).
MAX_TOKENS = 16384
TEMPERATURE = 0.0
MIN_PIXELS = 448 * 448
MAX_PIXELS = 2880 * 2880
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.9
ENFORCE_EAGER = True  # safest first; set False for speed once kernels are validated
TRUST_REMOTE_CODE = True


def as_dict() -> dict:
    return {
        "backend": BACKEND,
        "server_url": SERVER_URL,
        "api_model_name": API_MODEL_NAME,
        "weights_dir": WEIGHTS_DIR,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "enforce_eager": ENFORCE_EAGER,
        "trust_remote_code": TRUST_REMOTE_CODE,
    }
