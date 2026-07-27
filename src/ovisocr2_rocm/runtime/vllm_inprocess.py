"""In-process vLLM backend for OvisOCR2 (lazy: torch/vllm/PIL imported in methods).

This is the ONLY real inference path. Both ``adapter/run_adapter.py`` and the
standard CLI reach it through :func:`ovisocr2_rocm.pipeline.run_pipeline`, so
there is a single pipeline (no second implementation).

D5 fixes baked in (vs the legacy adapter):
  * ``gdn_prefill_backend`` is NOT silently dropped. We probe the installed
    vLLM's ``LLM`` signature; if the field is requested but unsupported we raise
    a clear error rather than falling back to ``LLM(**kwargs)``.
  * Per-page outputs are zipped STRICTLY against inputs — a length mismatch is an
    explicit failure, never a silent truncation that loses pages.
  * ``actual_dtype()`` reports the dtype DETECTED off the loaded model, never a
    config parameter (no fabricated bf16).
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import PageResult, RuntimeBackend


class VLLMBackend(RuntimeBackend):
    name = "vllm"

    def __init__(self) -> None:
        self._llm = None
        self._chat = None
        self._sp = None
        self._dtype: str | None = None
        self._env: dict = {}

    # -- load (heavy imports happen here, not at module import) ---------------
    def load(self, cfg, prompt: str) -> None:
        import os

        from vllm import LLM, SamplingParams

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

        # D5: probe gdn_prefill_backend support instead of try/except TypeError.
        # A signature "supports" the kwarg if it has a named param OR accepts
        # **kwargs (var-keyword) — both mean passing it will not raise TypeError.
        import inspect

        import vllm

        sig = inspect.signature(LLM.__init__)
        params = sig.parameters
        supports_gdn = ("gdn_prefill_backend" in params) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if cfg.gdn_prefill_backend and not supports_gdn:
            raise RuntimeError(
                f"installed vLLM {vllm.__version__} does not accept gdn_prefill_backend, but the "
                f"config requires '{cfg.gdn_prefill_backend}'. Refusing to silently drop it (D5). "
                "Pin a vLLM that supports it (OvisOCR2 upstream pins 0.22.1), or unset "
                "gdn_prefill_backend explicitly.")
        if supports_gdn:
            self._llm = LLM(gdn_prefill_backend=cfg.gdn_prefill_backend, **kwargs)
        else:
            self._llm = LLM(**kwargs)

        self._sp = SamplingParams(max_tokens=cfg.max_tokens, temperature=cfg.temperature)
        self._chat = self._llm.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # record the ACTUAL dtype (detected, not requested)
        from ..provenance import detect_dtype
        self._dtype = detect_dtype(self._llm)
        from ..provenance import collect_env
        self._env = collect_env()
        if self._dtype:
            self._env["dtype"] = self._dtype

    def actual_dtype(self) -> str | None:
        return self._dtype

    def env(self) -> dict:
        return dict(self._env)

    # -- infer (batch + bisect; never raises per page) -----------------------
    def _build_input(self, image_path: Path, cfg):
        from PIL import Image

        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
        return {
            "prompt": self._chat,
            "multi_modal_data": {"image": rgb},
            "mm_processor_kwargs": {"images_kwargs": {"min_pixels": cfg.min_pixels,
                                                      "max_pixels": cfg.max_pixels}},
        }

    def _run_batch(self, images: list[Path], cfg, postprocess_fn) -> list[PageResult]:
        """Run one batch; on whole-batch failure, bisect to single pages.

        Never raises. ``seconds`` is honest only for single-page batches
        (``len==1``); otherwise None because vLLM schedules a batch internally.
        """
        try:
            from ..postprocess import postprocess as _default_pp  # noqa: F401 (kept for parity)
            inputs = [self._build_input(i, cfg) for i in images]
            t0 = time.time()
            outputs = self._llm.generate(inputs, self._sp)
            wall = time.time() - t0
        except Exception as e:  # noqa: BLE001 -- whole-batch failure (OOM/bad image/generate error)
            if len(images) == 1:
                return [PageResult(images[0], None, None, f"batch failed: {e}")]
            mid = len(images) // 2
            return (self._run_batch(images[:mid], cfg, postprocess_fn)
                    + self._run_batch(images[mid:], cfg, postprocess_fn))

        # D5: STRICT length — a mismatch is an explicit failure, never silent loss.
        if len(outputs) != len(images):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(images)} inputs "
                f"(backend={self.name}); refusing to zip-shorten (would lose pages).")

        share = wall / len(images) if images else 0.0
        out: list[PageResult] = []
        for img, out_obj in zip(images, outputs, strict=True):
            try:
                text = postprocess_fn(out_obj.outputs[0].text.strip())
                if not text.strip():
                    out.append(PageResult(img, None, None, "empty prediction"))
                else:
                    out.append(PageResult(img, text, share if len(images) == 1 else None, None))
            except Exception as e:  # noqa: BLE001 -- isolate per-page postprocess failure
                out.append(PageResult(img, None, None, str(e)))
        return out

    def infer(self, images: list[Path], cfg) -> list[PageResult]:
        from ..postprocess import postprocess
        results: list[PageResult] = []
        for start in range(0, len(images), cfg.batch_size):
            batch = images[start:start + cfg.batch_size]
            results.extend(self._run_batch(batch, cfg, postprocess))
        return results

    def close(self) -> None:
        # vLLM holds the GPU via the engine; best-effort release.
        with __import__("contextlib").suppress(Exception):
            if self._llm is not None:
                del self._llm
                self._llm = None
