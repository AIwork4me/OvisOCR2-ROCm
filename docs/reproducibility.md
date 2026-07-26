# Reproducibility

A score is only meaningful if someone else can reproduce it from the committed
repo. This repo + the engine make that mechanical.

## What gets committed

- `adapter/run_adapter.py` — the exact inference code (in-process vLLM, official
  OvisOCR2 recipe). The `smoke` backend is a no-GPU CI placeholder.
- `adapter/adapter_config.py` — the recipe defaults (sampling, pixels, paths).
- `eval/configs/omnidocbench_v16.yaml` — metrics + dataset revision.
- `model_card.json` — declared hardware, badge, submetrics, artifact pointers.
- `REPRO.yaml` — flat lockfile: weights revision + sha256, venv, ROCm, git commit.
- `reproduce.md` — paste-and-run reproduction entry.
- `results/omnidocbench/v16/linux-rocm/` — published `run_summary.json` +
  `provenance.json` + the 6-artifact bundle.

## What the engine records (provenance)

Every published run produces a schema-validated `provenance.json`:

- `git_commit` — the exact repo state.
- `engine_version` — the `omnidocbench-rocm` version.
- `dataset_revision` — the pinned OmniDocBench revision (`2b161d0`).
- `adapter_command` — the literal subprocess command.
- `backend` — adapter-reported (from `_run_stats.json["engine"]`), not the
  requested flag.
- platform, model_id, page counts, artifact paths.

So a third party checks out that commit, provisions the same ROCm vLLM venv +
dataset revision + weights, re-runs the recorded command, and expects the same
number.

## Determinism notes

OvisOCR2 is run **greedy** (`temperature=0.0`), so outputs are deterministic
given identical (weights, vLLM version, image preprocessing). Known
non-determinism surfaces the analysis explicitly avoids:

- **vLLM version** — the repo runs the upstream card's pinned **vLLM 0.22.1**
  (`gdn_prefill_backend='triton'`). For transparency: an earlier run on vLLM
  0.19.0 produced the same Overall within noise (the CDM gap is model-inherent,
  version-independent — see `docs/known-gaps.md`), so 0.19.0 and 0.22.1 outputs
  are interchangeable for this model.
- **Batching order** — vLLM's continuous batching can in principle affect
  floating-point reduction order; greedy decoding is robust to this in practice.

## Checklist before requesting a `verified` badge

1. `model_card.json.hardware` reflects the actual GPU/VRAM/driver.
2. `results/omnidocbench/v16/linux-rocm/{run_summary,provenance}.json` committed.
3. `omnidocbench-rocm conformance .` → `CONFORMANT`.
4. `Dockerfile.repro` + `VERIFIED.yaml` committed; a maintainer reproduced the
   committed Overall in pinned Docker within ±0.5 (`check_verified.py`).
   (Deferred here — the dev env has no Docker; this is a `community` entry.)
