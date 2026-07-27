"""OvisOCR2-ROCm — v2-conformant document-parsing adapter for OmniDocBench-ROCm.

Package import MUST stay cheap: it never imports a model runtime
(torch / vllm / paddle / onnxruntime / PIL). Those are imported lazily, only
inside :mod:`ovisocr2_rocm.runtime.vllm_inprocess` methods, so ``version`` /
``capabilities`` / ``doctor`` and the whole module tree import on a CI box with
no GPU and no heavy deps installed.
"""
from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
