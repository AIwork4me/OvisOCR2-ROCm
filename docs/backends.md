# Backends — OvisOCR2-ROCm

## What this repo uses

**vLLM, in-process, on ROCm (linux-rocm).** The adapter
(`adapter/run_adapter.py`) loads OvisOCR2 via `vllm.LLM(...)` inside the adapter
process and runs batched `generate()` over the page set. This **matches the
upstream model card's `OvisOCR2Parser`** recipe byte-for-byte, so the adapter's
predictions are identical to the card's reference usage.

This differs from the zone's server-based VLM adapters (e.g. Unlimited-OCR-ROCm,
which calls a separate vLLM OpenAI server per image). Reasons for in-process here:

1. **Fidelity** — the official card uses in-process `LLM()`; matching it removes
   a variable from the precision-alignment claim.
2. **Simplicity** — no server orchestration; one subprocess does everything.
3. **Server-side batching for free** — `generate()` over the full input list
   lets vLLM schedule batches internally (≈ 220–300 tok/s aggregate), so there is
   no per-image latency penalty from the in-process choice.

## Why a qwen3_5-capable vLLM is required

`config.json` declares `architectures: ["Qwen3_5ForConditionalGeneration"]`. vLLM
routes this to `vllm/model_executor/models/qwen3_5.py`, which only exists from
vLLM v0.19.0 onward (merged after 0.16.0). The hybrid GDN text backbone runs via
`vllm/v1/attention/backends/gdn_attn.py` (Triton/FLA prefill kernel on ROCm). If
your vLLM raises `architectures [...] not supported`, rebuild with
[`rocm-vllm-installer`](https://github.com/AIwork4me/rocm-vllm-installer).

## Backend selection (`config["backend"]`)

| `backend` | What runs | GPU? | Use |
|---|---|---|---|
| `vllm` | In-process vLLM/ROCm, full official recipe | yes | all published results |
| `smoke` | No-GPU placeholder `.md` per page | no | CI / contract verification |

Switch via `adapter/adapter_config.py::BACKEND`, the `--backend` CLI flag, or the
`OVISOCR2_BACKEND` env var.

## Windows (`windows-hip`)

`community-wanted`. No path yet — Qwen3-Next GDN is not in llama.cpp/GGUF, and
ONNX/DirectML coverage of the hybrid linear-attention backbone does not exist.
Not a first-class backend until one lands; see `docs/known-gaps.md`.
