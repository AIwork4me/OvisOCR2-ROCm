# OvisOCR2-ROCm

**OvisOCR2 on AMD Radeon — the first end-to-end model to top OmniDocBench v1.6,
now running natively on ROCm.**

[OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) (ATH-MaaS / Alibaba,
Apache-2.0, 0.8B params) is a compact end-to-end page parser: give it a document
page image and it emits structured Markdown — text, tables, formulas, reading
order — in one pass. It set a new state of the art on
[OmniDocBench v1.6](https://arxiv.org/abs/2607.13639) at **Overall 96.58**, the
first end-to-end model to beat the pipeline methods that previously led the
board. This repo runs it **on AMD ROCm via vLLM** (gfx1100 / Radeon PRO W7900),
with full-set evaluation reproducing the paper's numbers within tolerance.

- **Model:** `ovisocr2` v1.0 — Apache-2.0, no commercial restriction
- **Backend:** vLLM 0.19.0 (ROCm), in-process (matches the upstream model card)
- **Platform:** `linux-rocm` (community) · `windows-hip` (community-wanted)
- **Zone:** [OmniDocBench-ROCm](https://github.com/AIwork4me/OmniDocBench-ROCm)

> **Architecture note.** Despite the "Ovis" name, `config.json` declares
> `model_type: qwen3_5` / `Qwen3_5ForConditionalGeneration` — a Qwen3-VL vision
> encoder on a **Qwen3-Next GDN (gated-delta-net) hybrid** text backbone. vLLM
> routes it to its native `qwen3_5` implementation; the GDN linear-attention
> layers run via the Triton/FLA prefill kernel on ROCm.

## Comparison — OmniDocBench v1.6 (linux-rocm)

| Model | Params | Backend | Overall | Badge |
|---|---|---|---|---|
| **OvisOCR2 (this repo)** | **0.8B** | **vLLM/ROCm** | **96.6** | community |
| PaddleOCR-VL-1.6 | 0.9B | llama.cpp/HIP | 95.77 | community |
| MinerU2.5 | 1.2B | vLLM/ROCm | 95.56 | community |
| HunyuanOCR | 1B | vLLM/ROCm | 93.64 | community |

OvisOCR2 is the **smallest model** in the zone and the **highest-scoring** — and
the first *end-to-end* parser to lead the board. Paper reference: Overall 96.58,
text edit-dist 0.025, formula CDM 97.5, table TEDS 94.8, reading-order 0.111
(arXiv 2607.13639, Table 2). See [`model_card.json`](model_card.json) for the
committed measurement.

## Install

Requires a qwen3_5-capable ROCm vLLM. The reference build is produced by
[`rocm-vllm-installer`](https://github.com/AIwork4me/rocm-vllm-installer)
(clones vLLM v0.19.0 + ROCm patches, builds for gfx110X-all, torch 2.10+rocm7.12).

```bash
# 1. ROCm vLLM venv (one-time; ~1-2 h build):
bash <(curl -sSL https://raw.githubusercontent.com/AIwork4me/rocm-vllm-installer/main/install.sh)
# Verify qwen3_5 is registered:
python -c "from vllm.model_executor.models.registry import ModelRegistry as m; \
  print('Qwen3_5ForConditionalGeneration' in m.get_supported_archs())"   # -> True

# 2. This repo + the engine:
pip install -e ".[dev]"
pip install omnidocbench-rocm        # the engine (omnidocbench-rocm CLI + types)

# 3. Weights (HF or ModelScope — identical):
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ATH-MaaS/OvisOCR2', local_dir='/root/models/OvisOCR2')"
export OVISOCR2_WEIGHTS=/root/models/OvisOCR2
```

GPU: any single gfx1100 with ≥ 16 GB VRAM (the 0.8B model peaks ≈ 6 GB at 32k
context). Verify: `rocminfo | grep gfx1100`.

## Demo

The `smoke` backend needs no GPU — it writes a placeholder `.md` per image so the
contract is verifiable end-to-end in CI:

```bash
bash examples/run_demo.sh
```

To actually parse a page with the model:

```bash
export HIP_VISIBLE_DEVICES=0
mkdir -p /tmp/in /tmp/out && cp examples/demo.png /tmp/in/
python adapter/run_adapter.py --img-dir /tmp/in --out-dir /tmp/out \
  --platform linux-rocm --backend vllm
cat /tmp/out/*.md
```

## Evaluation

Full OmniDocBench v1.6 (1651 pages), Edit_dist + TEDS + CDM:

```bash
export DATASET=/root/datasets/OmniDocBench_data
export OMNIDOCBENCH_CHECKOUT=/path/to/OmniDocBench   # pinned at 2b161d0
make eval-linux        # = omnidocbench-rocm run --stage all ... (infer + score + publish)
```

Or step-by-step (infer → score → publish) — see [`reproduce.md`](reproduce.md).
Eval config: [`eval/configs/omnidocbench_v16.yaml`](eval/configs/omnidocbench_v16.yaml).

## Reproducibility

- **Hardware:** AMD gfx1100 (Radeon PRO W7900, 48 GB) × 4; runs on one.
- **ROCm driver:** 7.2 (torch 2.10.0+rocm7.12).
- **Backend:** vLLM 0.19.0 ROCm (`vllm-build-gfx110x` venv), in-process.
- **Recipe:** official OvisOCR2 card — greedy (temp=0), `max_tokens=16384`,
  pixels 448²–2880², `_clean_truncated_repeats`, visual-region tags filtered.
- **Weights:** `ATH-MaaS/OvisOCR2` — see [`REPRO.yaml`](REPRO.yaml) for revision + sha256.
- Results + provenance: [`results/omnidocbench/v16/linux-rocm/`](results/omnidocbench/v16/linux-rocm/).
  See [`docs/reproducibility.md`](docs/reproducibility.md).

## Known Gaps

- **`windows-hip`:** `community-wanted`. OvisOCR2's Qwen3-Next GDN architecture
  has no GGUF/HIP-SDK serving path yet; Windows is deferred.
- **vLLM version:** built on vLLM 0.19.0 (the card pins 0.22.1). 0.19.0 lacks the
  `gdn_prefill_backend` arg (the default GDN path is used); subset alignment
  confirms outputs match the paper within tolerance.
- **Throughput:** eager mode + a ROCm paged-attention fallback give moderate
  throughput; the full set runs in ≈ 1 h on one GPU (≈ 30 min sharded across two).
  Non-eager/cudagraph tuning is future work.
- **`verified` tier:** not yet — requires a maintainer Docker reproduction (the
  dev env has no Docker). This is a `community` (self-attested, CI-verified)
  entry; see [`docs/known-gaps.md`](docs/known-gaps.md).

## License

Apache-2.0 (weights and code). See [`LICENSE`](LICENSE).
