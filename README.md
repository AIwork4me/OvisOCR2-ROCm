# OvisOCR2-ROCm

[![CI](https://github.com/AIwork4me/OvisOCR2-ROCm/actions/workflows/ci.yml/badge.svg)](https://github.com/AIwork4me/OvisOCR2-ROCm/actions/workflows/ci.yml)

**OvisOCR2 on AMD Radeon — an end-to-end document parser running natively on ROCm.**

[OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) (ATH-MaaS / Alibaba,
Apache-2.0, 0.8B params) is a compact end-to-end page parser: give it a document
page image and it emits structured Markdown — text, tables, formulas, reading
order — in one pass. This repo runs it **on AMD ROCm via vLLM** (gfx1100 / Radeon
PRO W7900) and publishes its OmniDocBench v1.6 measurement under the
[OmniDocBench-ROCm](https://github.com/AIwork4me/OmniDocBench-ROCm) v2 standard
(`rocmdoc.yaml` + `model_card_v2.json` + a standard CLI).

Measured results are **generated** from [`model_card_v2.json`](model_card_v2.json)
into the block below — no score in this README is hand-typed. Cross-model
comparison lives in the central hub, not in this sub-repo.

- **Model:** `ovisocr2` v1.0 — Apache-2.0, no commercial restriction
- **Backend:** vLLM (ROCm), in-process
- **Platform:** `linux-rocm` (supported) · `windows-hip` (unsupported — see Known Gaps)
- **Standard CLI:** `ovisocr2-rocm {version,capabilities,doctor,parse} --json`

> **Architecture note.** Despite the "Ovis" name, `config.json` declares
> `model_type: qwen3_5` / `Qwen3_5ForConditionalGeneration` — a Qwen3-VL vision
> encoder on a **Qwen3-Next GDN (gated-delta-net) hybrid** text backbone. vLLM
> routes it to its native `qwen3_5` implementation; the GDN linear-attention
> layers run via the Triton/FLA prefill kernel on ROCm.

## Results — OmniDocBench v1.6 (linux-rocm)

<!-- BEGIN GENERATED RESULTS -->
<!-- Source: model_card_v2.json — do not edit by hand; run scripts/generate_readme_results.py -->

| result_id | platform | backend | precision | overall | text_edit_dist | reading_order | table_teds % | formula_cdm % | assurance | status |
|---|---|---|---|---|---|---|---|---|---|---|
| ovisocr2__linux-rocm__vllm__bf16__v1-6__7d3d44f37a91 | linux-rocm | vllm | bf16 | 95.88 | 0.0260 | 0.1110 | 94.82 | 95.41 | submitted | valid |

_Last generated from `model_card_v2.json`. Cross-model comparison lives in the [central hub](https://github.com/AIwork4me/OmniDocBench-ROCm), not in this repo._
<!-- END GENERATED RESULTS -->

## Install

Requires a qwen3_5-capable ROCm vLLM. The reference build is produced by
[`rocm-vllm-installer`](https://github.com/AIwork4me/rocm-vllm-installer)
(build vLLM v0.22.1 + ROCm patches for gfx110X-all, torch 2.10+rocm7.12).

```bash
# 1. ROCm vLLM 0.22.1 venv (one-time; ~1-2 h build). Clone the installer locally
#    so its patches/ dir is available (do NOT use curl|bash):
git clone --branch v1.0.0 https://github.com/AIwork4me/rocm-vllm-installer.git
cd rocm-vllm-installer && VENV=/root/venvs/vllm-0221b VLLM_VERSION=v0.22.1 bash install.sh
# Verify qwen3_5 is registered:
/root/venvs/vllm-0221b/bin/python -c "from vllm.model_executor.models.registry import ModelRegistry as m; \
  print('Qwen3_5ForConditionalGeneration' in m.get_supported_archs())"   # -> True

# 2. The engine (omnidocbench-rocm, a sibling GitHub project — not on PyPI; install
#    the pinned commit into the venv) + this repo:
/root/venvs/vllm-0221b/bin/pip install "omnidocbench-rocm @ git+https://github.com/AIwork4me/omnidocbench-rocm.git@c1267cb1104e87bf9f8130875ce2f7da329ddcb4"
pip install -e ".[dev]"

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

Or via the standard CLI (pure-JSON contract):

```bash
ovisocr2-rocm parse --img-dir /tmp/in --out-dir /tmp/out --platform linux-rocm --backend vllm --json
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
- **Backend:** vLLM 0.22.1 ROCm (`vllm-0221b` venv), in-process, `gdn_prefill_backend='triton'`.
- **Recipe:** official OvisOCR2 card — greedy (temp=0), `max_tokens=16384`,
  pixels 448²–2880², `_clean_truncated_repeats`, visual-region tags filtered.
- **Weights:** `ATH-MaaS/OvisOCR2` — see [`REPRO.yaml`](REPRO.yaml) for revision + sha256.
- Results + provenance: [`results/omnidocbench/v16/linux-rocm/`](results/omnidocbench/v16/linux-rocm/).
  See [`docs/reproducibility.md`](docs/reproducibility.md).

## Known Gaps

- **`windows-hip`:** `unsupported` / community-wanted. OvisOCR2's Qwen3-Next GDN
  architecture has no GGUF/HIP-SDK serving path yet; Windows is **not evaluated**
  and carries no result (the 0-page smoke fixture was moved to `tests/fixtures/`).
- **Formula CDM gap (model-inherent):** the gap to the upstream paper's Overall is
  concentrated in formula CDM and is **model-inherent + version-independent** — a
  0.19.0-vs-0.22.1 A/B reproduces the same CDM on both. Not closable via recipe or
  version; see [`docs/known-gaps.md`](docs/known-gaps.md). The headline numbers live
  only in the generated results block above (from `model_card_v2.json`).
- **Throughput:** eager mode + a ROCm paged-attention fallback give moderate
  throughput; the full set is **observed** at ≈ 1 h on one W7900 (≈ 30 min sharded
  across two) — a manual measurement, not CI-derived; per-page latency is not
  recorded in the published bundle. Non-eager/cudagraph tuning is future work.
- **`verified` tier / dtype:** not yet — the published result is `assurance:
  submitted` (self-attested, CI-verified structure + conformance). Promotion to
  higher assurance requires a maintainer Docker reproduction and a GPU-verified
  dtype (not claimed here); see [`docs/known-gaps.md`](docs/known-gaps.md).

## License

Apache-2.0 (weights and code). See [`LICENSE`](LICENSE).
