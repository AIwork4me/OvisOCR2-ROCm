---
model_id: "ovisocr2"
backend: "vllm"
hardware:
  gpu: "AMD gfx1100 (Radeon PRO W7900, 48 GB)"
  vram_min_gb: 16
environment:
  type: "venv"
  image: "vllm-build-gfx110x (vLLM 0.19.0 ROCm, torch 2.10.0+rocm7.12, transformers 4.57.6)"
  rocm: "7.2"
command: |
  export OVISOCR2_WEIGHTS=/root/models/OvisOCR2   # or ATH-MaaS/OvisOCR2 to auto-download
  export HIP_VISIBLE_DEVICES=0                    # any one gfx1100 GPU
  python adapter/run_adapter.py \
    --img-dir "$DATASET/images" --out-dir predictions/ovisocr2 \
    --platform linux-rocm --backend vllm
expected_overall:
  value: 95.87
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
4. A qwen3_5-capable ROCm vLLM venv. The reference build is produced by
   [`rocm-vllm-installer`](https://github.com/AIwork4me/rocm-vllm-installer)
   (clones vLLM v0.19.0 + ROCm patches, builds for gfx110X-all). Verify:
   `python -c "from vllm.model_executor.models.registry import ModelRegistry; \
   print('Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs())"`.

## Quickstart

The adapter needs **both** vLLM 0.19 (qwen3_5) and the engine types. The reference
venv `vllm-build-gfx110x` has both (install the engine into it once:
`pip install --no-deps -e /path/to/OmniDocBench-ROCm`). Inference runs there;
scoring/publishing are venv-agnostic (the engine spawns the 3.11 eval-venv itself).

```bash
VENV=/root/venvs/vllm-build-gfx110x   # the qwen3_5-capable ROCm vLLM venv

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

Overall **95.87** (paper reports 96.58; the 0.71-pt gap is entirely formula CDM —
see below). Per-metric on the full 1651-page set vs the OvisOCR2 technical report
(arXiv 2607.13639, Table 2):

| metric | reproduced | paper |
|---|---|---|
| text edit-dist ↓ | 0.0256 | 0.025 |
| reading-order edit-dist ↓ | 0.1111 | 0.111 |
| table TEDS ↑ | 94.75 | 94.76 |
| table TEDS-S ↑ | 97.13 | 97.16 |
| formula CDM ↑ | 95.41 | 97.53 |

Three of four metrics are essentially perfect. Formula CDM is 2.1 pt low: a
verified vLLM-0.19.0-vs-0.22.1 formula-segmentation artifact on 26 of 2352
formulas (median CDM 1.0). Closing it requires the card's pinned vLLM 0.22.1
(a dedicated ROCm build — see `docs/known-gaps.md`). This is still **#1 in the
zone**. Full-set inference ≈ 1 h on one W7900 (≈ 30 min sharded across two);
scoring (incl. CDM) ≈ 30–45 min.

## If it fails

See [OmniDocBench-ROCm pitfalls](https://github.com/AIwork4me/OmniDocBench-ROCm/docs/pitfalls.md).
Common one: `architectures ['Qwen3_5ForConditionalGeneration'] not supported`
means your vLLM predates the qwen3_5 merge — rebuild with the installer above.
