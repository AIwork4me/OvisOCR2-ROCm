# OvisOCR2-ROCm P0 Backbone — Completion Report (2026-07-26)

Cycle 1 of 3. All 9 plan tasks done. No commits (per user guardrail); changes are
in the working tree. The published 95.88 and all committed result files are untouched.

## 1. Audit results (confirmed vs risk)

Confirmed (with evidence) and resolved this cycle:
- **`make eval-linux`/`make demo` silently ran smoke** (CLI `--backend default="smoke"` overrode config `vllm`; engine forwarded backend only if truthy). → Fixed (T1/T3).
- **`.env.local` documented but never loaded; `temperature` hardcoded `0.0`; any non-smoke string → vllm; `windows-hip+vllm` not rejected.** → Fixed (T1).
- **No batching (all 1651 pages + all PIL in memory at once); non-atomic writes; skipped→`ok` w/ `seconds=0.0` polluting throughput; no stem-collision guard.** → Fixed (T3).
- **No real sharding** (racy `--skip-existing &` loop); no merge tool. → Fixed (T2/T4).
- **CI imported one type** (no pytest/ruff/build/conformance). → Fixed (T6).
- **Timing bug**: `latency_s_per_page == 0.0` because `t_page` was set after `generate()` returned. → Fixed (T3); **verified on real GPU** (8.98 s/page).
- **Install path broken**: 3 venv names; `bash <(curl)` can't fetch `patches/`; engine `pip install` fails (not on PyPI). → Fixed (T5/T8).
- **Formula-CDM attribution** was already resolved by a colleague's real 0.19.0-vs-0.22.1 A/B (model-inherent, version-independent). I fixed the README leftover that still said "verified vLLM-version artifact" and added the `analysis/formula-gap/` methodology skeleton.

Risk/not-confirmed (stated honestly):
- `ground_truth_sha256` in the **committed** bundle is still `not_recorded` and the committed `latency_s_per_page` is still `0.0` — left untouched per guardrail; only future runs record real data.
- Throughput "≈1 h / ≈30 min" is a manual observation, not CI-derived (documented as such).

## 2. Changes (by file)

New (21): `adapter/postprocess.py`, `adapter/sharding.py`, `scripts/{merge-shards,check-environment,verify-reproduction-inputs}.py`, `scripts/bootstrap-linux.sh`, `analysis/formula-gap/{README.md,compare.py,sample_manifest.example.json,.gitkeep}`, `conftest.py`, `tests/{test_config,test_postprocess,test_sharding,test_adapter_smoke,test_adapter_failures,test_merge_shards,test_cli,test_repro_inputs}.py`, `docs/superpowers/{specs,plans}/2026-07-26-…`, this report.

Modified (11): `adapter/adapter_config.py` (Config dataclass + resolver + validation + `.env.local`), `adapter/run_adapter.py` (config-driven; batching+bisection; atomic writes; honest timing; `_performance.json`/`_pages.jsonl` sidecars; shard selection; stem-collision guard; new CLI flags), `adapter/setup/00-install-deps.sh` (delegates to bootstrap, venv `vllm-0221b`), `Makefile`, `pyproject.toml`, `.github/workflows/ci.yml`, `README.md`, `README.zh-CN.md`, `reproduce.md`, `REPRO.yaml` (`git_commit`→`fdef866`), `model_card.json` (note: documents 0.19.0→95.87).

## 3. Behavior changes
- **Default backend for a real eval is now `vllm`** (was smoke): `make eval-linux`/`make demo` with no `BACKEND=` runs the real model. `smoke` is explicit (`make demo-smoke`, CI). Breaking, intentional, tested.
- **Config priority**: explicit CLI > env (incl. `.env.local`) > defaults; CLI override flags default `None`. Backend is a closed set `{smoke,vllm}`; `windows-hip+vllm` → explicit error (no silent fallback).
- **Batching + recovery**: `--batch-size` (default 8); per-batch load/`with Image.open`/generate/postprocess/atomic-write; whole-batch failure bisects to single pages; one bad image never aborts the run.
- **Atomic writes** (`.md.tmp` + fsync + `os.replace`); **stem-collision hard error** before inference.
- **Skipped pages**: `status="ok"` (conformance: `ok+fail+fallback==count`) but `seconds=None`+`attempts=0` → excluded from throughput; split recorded in `_pages.jsonl`/`_performance.json`.
- **Deterministic sharding**: `--num-shards`/`--shard-index` → per-shard subdirs; `merge-shards.py` validates coverage/duplicates/conflicts/failed-pages (never last-write-wins).
- **Honest telemetry** (vllm path): real `model_load_seconds`/`inference_wall_seconds`/`mean_latency_s_per_page` + versions + git in `_performance.json`; per-page `_pages.jsonl`. `null` where not measurable.
- **CI**: ruff + pytest + build + conformance + check-env + smoke demo on py3.11/3.12; `permissions: contents: read`; 25-min timeout; engine pinned via `git+…@v0.3.2`.
- **Install**: pinned installer clone (not `curl|bash`); unified `vllm-0221b` default + `VENV=` override; engine from git tag (not PyPI).

## 4. Test evidence (commands, exit codes, counts)
- `ruff check .` → All checks passed (exit 0).
- `pytest -q` → **40 passed** (1 smoke + 8 config + 5 postprocess + 7 sharding + 3 adapter_smoke + 6 adapter_failures + 3 merge + 3 cli + 4 repro), ~3.6 s, exit 0.
- `python -m build` → built `ovisocr2_rocm-1.0.tar.gz` + wheel, exit 0.
- `omnidocbench-rocm conformance .` → CONFORMANT, exit 0.
- `make smoke-test` → 40 passed. `make demo-smoke` → exit 0. `make eval-windows` → exit 1 with unsupported message. `make -n eval-linux` → `--backend vllm`.
- `bash examples/run_demo.sh` → exit 0 (smoke output).
- `--help` on `run_adapter.py`, `merge-shards.py`, `check-environment.py`, `verify-reproduction-inputs.py` → all exit 0.
- `verify-reproduction-inputs.py` → `sha256(model.safetensors) == REPRO` confirmed.
- `recompute_overall(committed metric) == 95.88` (unchanged).
- **Real-GPU sanity** (vLLM 0.22.1, GPU 1, 1 page): exit 0; `_performance.json` shows `inference_wall_seconds=8.979`, `mean_latency_s_per_page=8.979`, `model_load_seconds=37.5`, `gpu/vllm/torch/transformers` populated; non-smoke Markdown written.

## 5. Not verifiable in this environment
- Full 1651-page re-run / score reproduction: **0 dataset images** present and the box is shared — so the committed bundle is left as-is (forward-looking telemetry only).
- Throughput claim: shared box, small sample → stays "observed/manual".
- `max_memory_allocated_mb` reads `0.0` on this ROCm build (not reliably exposed) — recorded as-is, not fabricated.
- Docker / `verified` badge: no Docker in env.
- Hub (`OmniDocBench-ROCm`) registry sync: not checked out here (Cycle 3).

## 6. Remaining work
- **Cycle 2 (P1/P2):** support matrix (Tested/Expected/Unsupported/Community-wanted — currently "any gfx1100 ≥16 GB" over-claims), test expansion, template/CoC cleanup (`opensourcedocs@amd.com` contact review), demo-real README example, README first-pass de-densifying, `.ruff_cache/` untracking (pre-existing).
- **Cycle 3 (Hub):** exact registry-table generation steps for `OmniDocBench-ROCm`; verify "#1 in the zone" against the generated registry.
- **Future (needs a dedicated GPU + dataset):** full re-run to republish the bundle with real `gt_sha256`/timing; Docker `verified` reproduction.
- Decisions for you: when to `git add`+commit the 32 changed/new files (none committed yet); whether to commit the design spec/plan/report under `docs/`.

## 7. Risks
- **Backward compatibility:** default-backend flip (smoke→vllm) is a breaking change for anyone relying on the old default; mitigated by explicit CI smoke + tests + `make demo-smoke`/`demo-real`.
- **Result provenance:** committed bundle intentionally untouched; its `latency=0.0`/`gt_sha256=not_recorded` are documented limitations, not fixed in-place.
- **Schema:** contract `_run_stats.json` shape unchanged; rich telemetry lives in sidecars (`_performance.json`/`_pages.jsonl`); `PageStatus(**s)` stays valid.
- **Cross-repo:** engine pinned via git tag `v0.3.2` (not on PyPI); a future breaking engine change would need a new tag pin.
- **Documentation mismatch:** design spec/plan under `docs/superpowers/` describe the original problems (so they contain the phrases the greps flag) — these are working docs, not shipped claims.

```
Current quality score: 8/10
Expected after Cycle 2 + a dedicated-GPU full re-run: 9/10
Expected after Docker `verified` reproduction: 9.5/10
```
(95.88 stands; score reflects P0 scope completed and GPU-verified, not inflated by docs.)
