# Known gaps — OvisOCR2-ROCm

Open items and honest limitations. A `verified` badge would require resolving the
Docker-repro item; the rest are scoping notes, not defects.

## Platform / backend scope

- **`windows-hip` is `community-wanted`.** OvisOCR2's text backbone is Qwen3-Next
  GDN (a hybrid linear-attention architecture). There is no GGUF/HIP-SDK serving
  path for it yet (llama.cpp does not implement qwen3_5/GDN), so Windows is
  deferred until one exists. Linux/ROCm is the only first-class platform here.

- **vLLM 0.19.0, not the card's 0.22.1.** The ROCm build in
  `vllm-build-gfx110x` is vLLM v0.19.0 (the first tagged release whose
  `model_executor/models/qwen3_5.py` ships). v0.19.0 lacks the
  `gdn_prefill_backend` constructor arg the upstream card passes
  (`="triton"`); this adapter uses vLLM's default GDN prefill path. The
  100-page subset alignment (text 0.033 vs 0.025, reading-order 0.112 vs 0.111)
  confirms outputs match the paper within tolerance, but a 0.1–0.3 pt Overall
  drift vs the card's exact version is possible.

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
  need to embed the source-built ROCm vLLM 0.19.0 — non-trivial, hence deferred.

- **Subset table-TEDS observation.** On a 100-page / 42-table subset, table TEDS
  measured ~89% vs the paper's 94.8%. Small-sample noise is the likely cause
  (the full ~665-table set is the authoritative number — see `model_card.json`).
  If the full-set TEDS stays ~5 pp low, it is a candidate recipe knob to revisit
  (table HTML format vs the matcher); it does not threaten the Overall (within
  the 0.5 tolerance either way).
