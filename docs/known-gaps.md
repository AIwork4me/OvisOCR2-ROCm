# Known gaps — OvisOCR2-ROCm

Open items and honest limitations. A `verified` badge would require resolving the
Docker-repro item; the rest are scoping notes, not defects.

## Platform / backend scope

- **`windows-hip` is `community-wanted`.** OvisOCR2's text backbone is Qwen3-Next
  GDN (a hybrid linear-attention architecture). There is no GGUF/HIP-SDK serving
  path for it yet (llama.cpp does not implement qwen3_5/GDN), so Windows is
  deferred until one exists. Linux/ROCm is the only first-class platform here.

- **Formula CDM gap — model-inherent, version-independent (verified).** Overall
  95.88 vs the paper's 96.58; the 0.70-pt gap is **entirely formula CDM** (95.41
  vs 97.53) on ~22 of 2352 formulas (median CDM 1.0). The model groups
  multi-formula systems (e.g. a derivation plus its (1)/(2) cases) as separate
  display blocks, while the OmniDocBench GT annotates them as one — the scorer's
  `split_equation_arrays` then mis-aligns those few formulas. Two rounds of
  systematic debugging ruled out recipe, post-processing, toolchain, and
  **version**: the repo runs the card's pinned vLLM 0.22.1 (`gdn_prefill_backend=
  'triton'`), and on the **full 1651-page set** 0.22.1 reproduces 0.19.0's CDM
  **exactly** (95.41 on both); a 19-page A/B earlier showed 0.8514 vs 0.8517.
  The gap is the model's segmentation vs the GT annotation, not a serving
  artifact — not closable via recipe or version. The repo uses vLLM 0.22.1 (built
  via `rocm-vllm-installer` with `VLLM_VERSION=v0.22.1`; note the extra
  `_C_stable_libtorch` cmake target where `silu_and_mul` registers).

## Throughput

- **Eager mode + paged-attention fallback.** Under eager decode, vLLM warns
  `Cannot use ROCm custom paged attention kernel, falling back to Triton`.
  Aggregate output is ~220–300 tok/s batched — enough for a full 1651-page run
  in ~1 h on one W7900 (~30 min sharded across two), but well short of the
  non-eager/cudagraph ceiling. Enabling compile/cudagraph and/or the ROCm
  custom paged-attn kernel is the main throughput lever left on the table.

## Result maturity

- **`community`, not `verified`.** This is a self-attested, CI-verified entry.
  Promotion to `verified` requires a maintainer Docker reproduction
  (`Dockerfile.repro` + `VERIFIED.yaml` + `check_verified.py`), which needs a
  Docker-capable box (absent in the dev environment). The Docker image would
  need to embed the source-built ROCm vLLM 0.22.1 — non-trivial, hence deferred.

- **Subset table-TEDS observation (resolved).** A 100-page / 42-table subset
  measured table TEDS ~89% vs the paper's 94.8 — confirmed to be small-sample
  noise: the full ~665-table set scores 94.75 (≈ paper 94.76). No action needed.
