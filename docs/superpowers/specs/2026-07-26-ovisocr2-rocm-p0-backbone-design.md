# OvisOCR2-ROCm — P0 Engineering Backbone (design spec)

- **Date:** 2026-07-26
- **Status:** Draft — awaiting user review (then → `writing-plans`)
- **Cycle:** 1 of 3 (P0 backbone). P1-2/P1-3/P2 polish and the `OmniDocBench-ROCm` Hub sync are later cycles.
- **Scope guardrails (from user):** no fabricated results/timing/metrics; do not recompute or silently change the published score; no GPU-inference claims without a GPU run; no smoke-described-as-real; never weaken the adapter contract; every behavior change gets a test; docs must match code; fix code first, docs second; don't delete result evidence; small, focused diffs; no commit/push/PR unless asked.

---

## 1. TL;DR

Turn this repo from "an experiment with a great number" into a trustworthy, reproducible, production-shaped eval repo. A colleague's two new commits (`fdef866`, `d00e137`) already **resolved the formula-CDM attribution with a real 0.19.0-vs-0.22.1 A/B** (the gap is model-inherent, version-independent) and **cleaned the published provenance** (real command, real commit, fresh 1651-page run). That removes the biggest reputational risk and most of P0-7/P0-6's provenance work.

What remains is the **engineering backbone**: an install path that actually works, a config layer that doesn't silently fall back to `smoke`, batched/resumable inference, **real** deterministic multi-GPU sharding, a CI gate that runs tests, and honest forward-looking telemetry (the committed bundle still reports `latency_s_per_page: 0.0` because the timer measures postprocessing, not inference).

This cycle delivers exactly that, verified via mocked-vLLM CPU tests + real `smoke` + `conformance`, with **no** GPU-eval claims and **no** change to the published score without explicit sign-off.

---

## 2. Current state (verified this session, with evidence)

### 2.1 Resolved by the colleague's `fdef866`/`d00e137` (done well — keep)
- **Formula-CDM attribution (P0-7 core):** real A/B; 0.22.1 reproduces 0.19.0's CDM (95.41 on both, full 1651-page set). Conclusion correctly rewritten in `docs/known-gaps.md`, `model_card.json.note`, `reproduce.md` as **model-inherent, version-independent**. `recompute_overall(committed metric) == 95.88` ✓.
- **Provenance (P0-6 half):** `provenance.adapter_command` is now the real argv (`/root/venvs/vllm-0221b/bin/python … --backend vllm`), `git_commit` = the real commit (`fdef866`), migration fields (`prediction_source_*`, `migration_type`) now empty → fresh in-repo run, not migrated.
- **`run_stats`** now has 1651 real pages, all `ok`/`attempts=1`.
- **`vllm-0221b` venv is real:** vLLM 0.22.1, `Qwen3_5ForConditionalGeneration registered: True`.

### 2.2 Still live (this cycle's work) — every item confirmed in code
| # | Item | Evidence |
|---|---|---|
| P0-1 | Install path broken; 3 inconsistent venv names; `bash <(curl)` can't fetch `patches/` | `00-install-deps.sh:7` wants `vllm-build-gfx110x` (holds **0.19.0**); docs/results use `vllm-0221b`; installer default is `gfx110x-prod`. `bash <(curl …)` at `README.md:51`, `README.zh-CN.md:33`, `00-install-deps.sh:17` |
| P0-2 | CLI `--backend` defaults to `smoke` and always overrides config; `make eval-linux`/`make demo` → **smoke** | `run_adapter.py:238` default `"smoke"`; merge at `:148`; engine `cli.py:149,211` default `""`, `stages.py:35` forwards only if truthy; `Makefile:6,28` `BACKEND ?=` empty |
| P0-2 | `temperature=0.0` hardcoded; no backend whitelist; `windows-hip+vllm` not rejected; `.env.local` never loaded | `run_adapter.py:169,156`; `adapter_config.py` uses only `os.environ` (docstring `:5` claims `.env.local`) |
| P0-3 | No batching (all pages + all PIL in memory at once); non-atomic writes; skipped→`ok` w/ `seconds=0.0`; no stem-collision guard; no batch failure isolation | `run_adapter.py:189-206,182` |
| P0-4 | No real sharding; `reproduce.md:80-82` is the racy `--skip-existing &` loop; no merge tool | no `--num-shards`/`--shard-index` anywhere |
| P0-5 | CI imports one type — no pytest/ruff/build/conformance; no matrix/permissions/timeout/badge | `.github/workflows/ci.yml`; `pyproject.toml` is 8 lines, `omnidocbench-rocm>=0.2.0` unpinned |
| P0-6 | **Timing bug:** `latency_s_per_page == 0.0` — `t_page` set after `generate()` returns; `ground_truth_sha256 == "not_recorded"`; no telemetry sidecar | `run_adapter.py:201`; `dataset_identity.json`; `run_summary.efficiency` |
| P0-7 leftover | `README.md:15` & `README.zh-CN.md:5` still say *"verified vLLM-version artifact"* — **contradicts** the new conclusion; "exactly/byte-identical" vs "within noise 0.8514 vs 0.8517" wording conflict; `analysis/formula-gap/` skeleton not created | grep hits this session |
| — | `REPRO.yaml.git_commit` stale (`139959c`, real run is `fdef866`) | `REPRO.yaml:31` |

### 2.3 Environment reality (hard limits, stated honestly)
- 4× gfx1100 partitions, 48 GiB each (matches the W7900-class platform), but **shared**: GPU[0]/[2] were ~50 GB occupied by other workloads this session. No exclusive-GPU guarantee.
- The active venv (`/opt/venv`) is vLLM `0.16.1.dev` → `Qwen3_5` **not** registered. The `vllm-0221b` venv **is** model-capable, but the **OmniDocBench dataset dir has 0 images** here, and the box is shared. → A clean full-set re-run is **not** on the menu this cycle.
- `docker`, `conda` absent; `python -m build` works; git reaches GitHub (curl TLS does not).
- **Therefore:** all work is verified via mocked-vLLM CPU tests + real `smoke` + `conformance` + `--help`. We do **not** claim GPU verification, and we do **not** recompute or alter the published 95.88/95.87.

---

## 3. Glossary (shared vocabulary)
- **adapter** — `adapter/run_adapter.py`; one function `run_adapter(img_dir, out_dir, *, platform, config) -> dict`. Invoked as a **subprocess** by the engine; engine never imports it.
- **contract** — the set of obligations the engine's `conformance.py` + `bundle_validator.py` enforce. Adapter must emit `out_dir/<stem>.md` + a schema-valid `_run_stats.json`; the run never raises.
- **backend** — `smoke` (no-GPU placeholder, CI gate) | `vllm` (real in-process ROCm path). No other values after this cycle.
- **run_stats** — `_run_stats.json`: `{schema_version,count,ok,fail,fallback,limit_pages,engine,stats[],efficiency?}`. Per-page `stats[]` entries are `PageStatus(image,status,error,seconds,attempts)` — **fixed shape** (engine reconstructs via `PageStatus(**s)`).
- **efficiency** — optional top-level dict in run_stats; engine's `_derive_efficiency` computes `latency_s_per_page` from ok-page `seconds` and merges adapter-reported `peak_vram_mb`/`gpu`.
- **provenance** — `<save_name>_provenance.json` in the published bundle; records command, commits, engine version, dataset revision, etc.
- **shard** — a deterministic slice of the sorted image set: `images[shard_index::num_shards]`.
- **CDM** — formula-rendering Contestable-Distance Metric; the OmniDocBench formula sub-metric. `Overall = ((1−text_EditDist)*100 + CDM*100 + TEDS*100)/3`.
- **GDN** — gated-delta-net; the Qwen3-Next hybrid linear-attention backbone. vLLM serves it via the Triton/FLA prefill kernel (`gdn_prefill_backend='triton'`).

---

## 4. Decisions (ADRs) — confirm at review

> These are my recommendations. Items marked **[DECISION]** are genuine forks I want your call on before `writing-plans`.

### ADR-1 · Configuration priority & backend validation (P0-2)
**Priority:** CLI explicit > environment (incl. `.env.local`) > `Config` defaults. CLI flags that override config default to `None` (not a value), so an unset flag never clobbers env/config. Resolve into a frozen `@dataclass Config`. Backend is a closed set `{smoke, vllm}`; any other value → `SystemExit` with a clear message (no silent vLLM-branch). `platform=windows-hip` + `backend=vllm` → explicit, non-fallback error ("unsupported; community-wanted"). Load `adapter/setup/.env.local` via `python-dotenv` **only for keys not already in `os.environ`** (shell wins). Wire `temperature` (currently hardcoded `0.0`) and every recipe knob through `Config`. Tests cover all 8 cases (defaults/env/CLI/.env/shell-wins/bad-backend/bad-combo).

**Consequence:** the default effective backend for a real eval becomes `vllm` (so `make eval-linux` with no `BACKEND=` runs the real model), while `smoke` stays an explicit CI choice. This inverts today's dangerous default — **breaking change, intentional, tested**.

### ADR-2 · Results integrity & telemetry home (P0-6) — **[DECISION: 95.88]**
- **Committed bundle:** leave **all** committed result files byte-untouched (also conformance-required: `windows-hip/_run_stats.json` must stay or that dir goes empty → NON-CONFORMANT). Add a short, honest note documenting the historical artifacts' limitations only if not already covered by `docs/known-gaps.md`.
- **Forward-looking telemetry.** Note the **root cause** of the committed `latency_s_per_page: 0.0`: it is the **timing bug** (ADR-4 sets `t_page` after `generate()` returns, so `seconds` ≈ postprocess-only ≈ 0.0004 s) — *not* the skipped-status issue (the colleague's run had zero skipped pages). Fix the timer; populate `efficiency.{peak_vram_mb,gpu}` (engine propagates these); write a **sidecar** `_performance.json` (run-level: `inference_wall_seconds`, `model_load_seconds`, `tokens`, `tokens_per_second`, git commit/dirty, weights revision, torch/HIP/vLLM/transformers versions) + `_pages.jsonl` (per-page: image/status/shard_index/error/output_bytes). Per-page `seconds` in the contract file is set to `None` when not honestly measurable (batched generate) so `_derive_efficiency` omits it rather than emitting a fake `0.0`. Rich fields live only in the sidecar — **never** injected into the contract `stats[]` (would break `PageStatus(**s)`).
- **[DECISION] The published score:** the colleague's 0.22.1 re-run legitimately changed 95.87 → **95.88** (recompute-confirmed, card-pinned version), but your guardrail said "don't modify 95.87." **My recommendation (B): keep 95.88 as primary and document 95.87 (0.19.0) alongside it for transparency** — most honest. Alternatives: (A) 95.88 only; (C) revert to 95.87 and treat the 0.22.1 run as verification-only. I will not touch committed result files regardless; this only governs README/REPRO/model_card prose.

### ADR-3 · Deterministic sharding (P0-4)
New CLI `--num-shards N` / `--shard-index I` (defaults 1/0). Deterministic sort of inputs; assignment `sorted_imgs[shard_index::num_shards]`. Validate `num_shards>=1` and `0<=shard_index<num_shards`. Each shard writes its own subdir `shard-{idx:05d}-of-{n:05d}/` with its own `_run_stats.json`. New `scripts/merge-shards.py --input-root … --out-dir … --expected-images …` validates: no missing pages, no duplicates, no content conflicts (never "last-write-wins"), no failed shards, consistent shard config, page count matches expected; emits a merged `_run_stats.json` and records `num_shards`/`shard_index`/per-shard counts+times/merge-time in provenance/sidecar. `reproduce.md` gets a real copy-pasteable 2-GPU sharded example; the old `--skip-existing &` loop is removed and explicitly noted as **not** sharding.

### ADR-4 · Batch + recovery (P0-3)
New `--batch-size` (default **8**). Load images **per batch** with `with Image.open(p) as im: im.convert("RGB")` (release between batches). Per-batch: preprocess → generate → postprocess → **atomic write** (`*.md.tmp` + `flush`/`os.fsync` + `os.replace`). Per-page exceptions caught (corrupt image, decode, preprocess, generate, empty output, postprocess, disk); one page never kills the run. Whole-batch generate failure → bisection down to single-page, recording the exact failing page. `--skip-existing` skips only if target exists **and** is non-empty **and** readable; **skipped pages keep `status="ok"`** (conformance requires `ok+fail+fallback == count` — a separate "skipped" status would break `bundle_validator`) but carry `seconds=None` + `attempts=0`, so `_derive_efficiency` excludes them from throughput; the generated-vs-skipped distinction is recorded in the sidecar (`_pages.jsonl`) and `_performance.json` (`generated_pages` / `skipped_pages`). Stem-collision precheck (e.g. `page.jpg`+`page.png` → same `page.md`) → hard error before any inference. Empty input dir: `smoke` may return 0 pages; `vllm` → explicit config error (a real eval with 0 pages is misconfigured).

### ADR-5 · Install path unification (P0-1) — **[DECISION: canonical venv name]**
Stop hardcoding a venv name in the repo. The repo requires "any qwen3_5-capable vLLM 0.22.1 venv"; `VENV=/path` overrides everywhere; the documented **reference** build is `vllm-0221b`. `00-install-deps.sh`/README/README.zh-CN/reproduce.md all use the **same** default (`/root/venvs/vllm-0221b`) and the same `VENV=` override. Replace `bash <(curl …)` with **clone installer at a pinned tag (`v1.0.0`) and run locally** so `patches/` is available. New `scripts/check-environment.py` (active-python: torch/ROCm/HIP/GFX/vLLM/qwen3_5-registration/weights-completeness; `--cpu-only` mode for CI) and `scripts/bootstrap-linux.sh` (idempotent, non-destructive, pinned deps). **[DECISION] Is `vllm-0221b` the right canonical default, or should the repo be fully venv-agnostic with no default path?** My recommendation: default `/root/venvs/vllm-0221b` (matches the published results) + `VENV=` override + `check-environment.py` validates whatever is active.

### ADR-6 · CI & packaging (P0-5)
`.github/workflows/ci.yml`: checkout; matrix py3.11/3.12; pip cache; `pip install -e ".[dev]"`; run `pytest -q`, `ruff check .`, `python -m build`, `omnidocbench-rocm conformance .`; run the smoke demo; assert README-referenced paths exist; `python scripts/check-environment.py --cpu-only`; `permissions: {contents: read}`; sensible timeout; **no GPU claims** in name/steps. CI badge added **after** the workflow actually runs green. `pyproject.toml`: add `[build-system]` (hatchling), `description`/`readme`/`license`/`urls`/`classifiers`/`requires-python>=3.11`, `pytest`/`ruff`/`build` config; pin `omnidocbench-rocm==0.3.2` (matches `provenance.engine_version`) — **[DECISION] confirm `==0.3.2` is installable on a plain CPU runner (no hard torch dep that breaks CI).** I'll verify before pinning.

### ADR-7 · Verification bar — **DECIDED (revised 2026-07-26)**
Two layers, each validating only what it actually can:
- **CI (CPU, mandatory, every push/PR):** mocked-vLLM unit/integration tests (config, batching, recovery, sharding, merge, postprocess, CLI, repro-inputs), real `make smoke-test`, real `--backend smoke` run, `ruff check .`, `python -m build`, `omnidocbench-rocm conformance .`, `python scripts/check-environment.py --cpu-only`, and `--help` on every new script. CI has no AMD GPU → it never claims GPU verification.
- **Real-GPU sanity run (manual, this cycle, NOT in CI):** after the mock-tested code lands, run a few real pages on a free GPU partition via the **existing** `vllm-0221b` venv (no rebuild — `Qwen3_5` already registered) to confirm the changed adapter still loads/serves, emits real Markdown, records real seconds/VRAM, and that batching/sharding work on the real model.
- **Honest limits (stated in the report, never faked):** this env has **0 dataset images** → cannot reproduce 95.88 or run full scoring; the box is **shared** → cannot reliably measure throughput (the ≈1 h / ≈30 min claim stays observed/manual); a free partition may be unavailable → if the sanity run can't execute, say so plainly.

---

## 5. Work breakdown (files · tests · acceptance)

> New test files (CPU-only, mocked vLLM): `tests/test_config.py`, `tests/test_adapter_smoke.py`, `tests/test_adapter_failures.py`, `tests/test_sharding.py`, `tests/test_merge_shards.py`, `tests/test_postprocess.py`, `tests/test_cli.py`, `tests/test_repro_inputs.py`. Keep the existing `tests/test_smoke.py`.

- **P0-1** — `adapter/setup/00-install-deps.sh`, `README.md`, `README.zh-CN.md`, `reproduce.md`; **new** `scripts/bootstrap-linux.sh`, `scripts/check-environment.py`, `scripts/verify-reproduction-inputs.py`. *Tests:* `test_repro_inputs.py` (sha256 target = `model.safetensors` — **verified this session**; REPRO/git-commit/dataset-revision/engine-version checks). *Accept:* all install commands use one venv default + `VENV=` override; no `bash <(curl)`; `check-environment.py --cpu-only` exits 0 in CI.
- **P0-2** — `adapter/adapter_config.py` (dataclass `Config` + `.env.local` load + backend/platform validation), `adapter/run_adapter.py` (CLI defaults `None`; resolve; remove `temperature` hardcode). *Tests:* `test_config.py` (8 cases). *Accept:* `make eval-linux` w/o `BACKEND=` → vllm; `--backend bogus` exits non-zero; `windows-hip+vllm` exits non-zero; `.env.local` honored, shell wins.
- **P0-3** — `adapter/run_adapter.py` (batch loop, atomic write, `seconds=None` for skipped, collision guard, bisection, `--batch-size`). *Tests:* `test_adapter_failures.py` (bad image isolated, empty-output=failed, atomic write, skip-existing excludes from throughput, batch→page localization; smoke imports no vLLM). *Accept:* a corrupt image doesn't abort the run; `.md` written atomically; skipped pages carry `seconds=None` (excluded from latency) and are distinguished in the sidecar while keeping `status="ok"` for conformance.
- **P0-4** — `adapter/run_adapter.py` (`--num-shards`/`--shard-index`, shard subdirs); **new** `scripts/merge-shards.py`. *Tests:* `test_sharding.py` (2/3/4/5 shards disjoint + cover-all + deterministic), `test_merge_shards.py` (detects missing/duplicate/conflict; stable order). *Accept:* two shards have zero overlap and full coverage; merge catches a deliberately missing page.
- **P0-5** — `.github/workflows/ci.yml`, `pyproject.toml`; CI badge in READMEs **after** green. *Accept:* CI runs pytest+ruff+build+conformance on push/PR across py3.11/3.12.
- **P0-6** — `adapter/run_adapter.py` (honest timing + `efficiency` + sidecar writers); ensure publish records `ground_truth_sha256` (engine invocation: pass dataset so it hashes `OmniDocBench.json`) — *verify the exact engine flag during planning*. Docs: mark "≈1 h / ≈30 min" as **observed/manual, not from committed stats**. *Accept:* a future run's `run_summary` no longer reports a fake `0.0` latency; sidecar carries real timing+versions.
- **P0-7 (leftovers)** — `README.md:15`, `README.zh-CN.md:5` (replace *"verified vLLM-version artifact"* with the model-inherent/version-independent conclusion already in `known-gaps.md`); reconcile "exactly" vs "within noise" wording across `reproduce.md`/`known-gaps.md`; **new** `analysis/formula-gap/{README.md,compare.py,sample_manifest.example.json,.gitkeep}` capturing the A/B methodology (the 9 locked variables) — **no fabricated 0.22.1 result files**. Also fix `REPRO.yaml.git_commit` → `fdef866`.
- **Makefile / DX** — add `install-dev`, `check` (= ruff+pytest+build+conformance), `smoke-test`, `demo-smoke`, `demo-real`, `eval-linux` (explicit `BACKEND?vllm`), `conformance`, `build`, `clean`; `eval-windows` fails clearly (community-wanted), never silently smokes. Conformance constraint: READMEs must still contain the literal section words `Install`/`Demo`/`Evaluation`/`Reproducibility`/`Known Gaps`.

---

## 6. Test strategy
- **Layer 1 — CPU unit/integration (the CI gate):** mocked vLLM (a fake `LLM`/`SamplingParams` injected for batch/recovery/shard tests; the `smoke` path needs no mock). No GPU, no 1.7 GB download. `tests/test_smoke.py` already drives the real subprocess. Target: fast suite (<30 s), clear failures, every behavior change covered.
- **Layer 2 — real-GPU sanity (manual, once, after Layer 1 is green):** a few real pages on a free partition via `vllm-0221b` to validate the changed adapter on the actual model (load/serve, real Markdown, real timing/VRAM, batching/sharding). Bounded by the honest limits in ADR-7; never claimed as CI/GPU-eval verification.

## 7. Risks & guardrails
- **Breaking change (ADR-1):** default backend flips smoke→vllm for evals. Mitigated by explicit CI `--backend smoke`, tests, and `make demo-smoke`/`demo-real` split.
- **Contract safety:** rich telemetry stays in sidecars; contract `stats[]` shape untouched; **page-status accounting must keep `ok+fail+fallback == count`** (enforced by `bundle_validator._validate_one`) — hence skipped pages stay `status="ok"` with `seconds=None`, not a new status; conformance re-run in CI catches regressions; `windows-hip` smoke artifact preserved (its dir must stay non-empty or conformance fails).
- **Provenance/result provenance:** committed bundle untouched; 95.88 not recomputed by us; `gt_sha256`/timing fixes are forward-looking only.
- **Cross-repo:** engine is `==0.3.2` pinned (verify CPU-installable); Hub sync is out of scope this cycle.

## 8. Out of scope (later cycles)
- P1-2 support matrix, P1-3 test expansion beyond the 8 files, P2 template/CoC/demo-real-example cleanup, `OmniDocBench-ROCm` Hub registry sync, real-GPU/Docker `verified` reproduction.

## 9. Decisions — resolved on review 2026-07-26
1. **[ADR-2] → (B):** 95.88 is the primary published number; 95.87 (vLLM 0.19.0) documented alongside for transparency. Committed result files remain untouched.
2. **[ADR-5]:** canonical default `/root/venvs/vllm-0221b`, `VENV=` override everywhere, `check-environment.py` validates whichever venv is active.
3. **[ADR-6]:** pin `omnidocbench-rocm==0.3.2`; verify CPU-installability before pinning.
4. **[ADR-7] (revised):** CPU mock tests (mandatory CI gate) + a one-time real-GPU sanity run this cycle (manual, a few pages on a free partition via `vllm-0221b`), with the honest limits in ADR-7.

Spec approved → handing to `writing-plans` to produce the step-by-step implementation plan.
