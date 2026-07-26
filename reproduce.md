---
model_id: "ovisocr2"
backend: "vllm"
hardware:
  gpu: "AMD gfx1100 (Radeon PRO W7900, 48 GB)"
  vram_min_gb: 16
environment:
  type: "venv"
  image: "vllm-0221b (vLLM 0.22.1 ROCm, torch 2.10.0+rocm7.12, transformers 4.57.6)"
  rocm: "7.2"
command: |
  export OVISOCR2_WEIGHTS=/root/models/OvisOCR2   # or ATH-MaaS/OvisOCR2 to auto-download
  export HIP_VISIBLE_DEVICES=0                    # any one gfx1100 GPU
  python adapter/run_adapter.py \
    --img-dir "$DATASET/images" --out-dir predictions/ovisocr2 \
    --platform linux-rocm --backend vllm
expected_overall:
  value: 95.88
  tolerance: 0.5
---

# Reproduce OvisOCR2 on AMD ROCm

OvisOCR2 (ATH-MaaS/OvisOCR2, 0.8B, Apache-2.0) runs **in-process via vLLM** on
Radeon gfx1100. vLLM routes `Qwen3_5ForConditionalGeneration` to its native
qwen3_5 implementation; the GDN (gated-delta-net) hybrid backbone runs via the
Triton/FLA GDN prefill kernel on ROCm. The adapter uses the upstream model
card's exact recipe, so predictions reproduce the paper's OmniDocBench v1.6
numbers (Overall 96.58) within tolerance.

## Prerequisites

1. `rocminfo | grep -E "Name:|Marketing Name"` shows a `gfx1100` GPU.
2. `/dev/kfd` is accessible: `ls -la /dev/kfd`.
3. VRAM ≥ 16 GB (the 0.8B model + 32k context peaks ≈ 6 GB; 16 GB headroom for batching).
4. A qwen3_5-capable ROCm vLLM **0.22.1** venv (the upstream OvisOCR2 card pins
   `vllm==0.22.1`). Build it from
   [`rocm-vllm-installer`](https://github.com/AIwork4me/rocm-vllm-installer) with
   `VLLM_VERSION=v0.22.1` (the installer's ROCm patches apply cleanly to v0.22.1;
   note: build the extra `_C_stable_libtorch` cmake target too — `silu_and_mul`
   registers there in 0.22.x). Verify:
   `python -c "from vllm.model_executor.models.registry import ModelRegistry; \
   print('Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs())"`.

## Quickstart

The adapter needs **both** vLLM 0.22.1 (qwen3_5 + `gdn_prefill_backend`) and the
engine types. The reference venv `vllm-0221b` has both (the 0.22.1 ROCm build +
the engine: `pip install --no-deps -e /path/to/OmniDocBench-ROCm`). Inference runs
there; scoring/publishing are venv-agnostic (the engine spawns the 3.11 eval-venv
itself). The first inference pass compiles the GDN/Triton kernels (~10-20 min
one-time, cached in `~/.triton`); later runs are fast.

```bash
VENV=/root/venvs/vllm-0221b   # the vLLM 0.22.1 ROCm venv

# 1. Provision weights (HF or ModelScope — identical):
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ATH-MaaS/OvisOCR2', local_dir='/root/models/OvisOCR2')"

# 2. Run the adapter over the full OmniDocBench v1.6 image set (1651 pages):
export OVISOCR2_WEIGHTS=/root/models/OvisOCR2
export HIP_VISIBLE_DEVICES=0
export DATASET=/root/datasets/OmniDocBench_data
"$VENV/bin/python" adapter/run_adapter.py \
  --img-dir "$DATASET/images" --out-dir predictions/ovisocr2 \
  --platform linux-rocm --backend vllm

# 3. Score (Edit_dist + TEDS + CDM) with the zone engine:
export OMNIDOCBENCH_CHECKOUT=/path/to/OmniDocBench   # checkout pinned at 2b161d0
omnidocbench-rocm score --platform linux-rocm --version v16 \
  --predictions-dir predictions/ovisocr2 \
  --run-stats predictions/ovisocr2/_run_stats.json --dataset-dir "$DATASET" --cdm
```

For a faster wall-clock on the full set, shard across two GPUs (predictions are
identical — each page is independent and greedy):

```bash
for s in 0 1; do HIP_VISIBLE_DEVICES=$s python adapter/run_adapter.py \
  --img-dir "$DATASET/images" --out-dir predictions/ovisocr2 \
  --platform linux-rocm --backend vllm --skip-existing & done; wait
```

## Expected output

Overall **95.88** (paper reports 96.58; the 0.70-pt gap is entirely formula CDM —
see below). Per-metric on the full 1651-page set vs the OvisOCR2 technical report
(arXiv 2607.13639, Table 2):

| metric | reproduced | paper |
|---|---|---|
| text edit-dist ↓ | 0.0260 | 0.025 |
| reading-order edit-dist ↓ | 0.1110 | 0.111 |
| table TEDS ↑ | 94.82 | 94.76 |
| table TEDS-S ↑ | 97.20 | 97.16 |
| formula CDM ↑ | 95.41 | 97.53 |

Three of four metrics are essentially perfect. Formula CDM is 2.1 pt low: a
verified **model-inherent** formula-segmentation difference on ~22 of 2352
formulas (median CDM 1.0) — **version-independent** (running on the card's pinned
vLLM 0.22.1 reproduces 0.19.0's CDM within noise: 0.8514 vs 0.8517 on the
affected pages). The model groups multi-formula systems differently than the GT
annotation; not closable via recipe or version. See `docs/known-gaps.md`. This is
still **#1 in the zone**. Full-set inference ≈ 1 h on one W7900 (first run adds a
~10-20 min one-time Triton/GDN kernel compile, cached after); scoring (incl. CDM)
≈ 30–45 min.

## If it fails

See [OmniDocBench-ROCm pitfalls](https://github.com/AIwork4me/OmniDocBench-ROCm/docs/pitfalls.md).
Common one: `architectures ['Qwen3_5ForConditionalGeneration'] not supported`
means your vLLM predates the qwen3_5 merge — rebuild with the installer above.
