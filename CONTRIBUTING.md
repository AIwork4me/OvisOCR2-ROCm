# Contributing to OvisOCR2-ROCm

Thanks for helping improve this adapter! A few ground rules.

## Getting started

```bash
pip install -e ".[dev]"
pip install omnidocbench-rocm
make smoke-test        # runs pytest
make publish           # runs the conformance checker
```

The `smoke` backend (no GPU) should always pass — keep it working.

## Wiring up a real model

1. Implement `_infer(img, platform, config) -> str` in `adapter/run_adapter.py` (return markdown for one image).
2. Set `BACKEND` in `adapter/adapter_config.py` to your backend name (or pass `--backend <name>`).
3. Fill in `adapter/setup/00-install-deps.{sh,ps1}` so a fresh machine can provision weights + runtime.
4. Run `make eval-linux` (or `eval-windows`), then `make publish`.

## Before opening a PR

- `make smoke-test` is green.
- `make publish` reports `CONFORMANT`.
- `model_card.json` reflects reality (hardware, badge, artifacts) — don't claim `verified` without committed, reproducible results.
- Update `docs/known-gaps.md` — close resolved bullets, add new ones.

## Reporting results

Commit the engine-produced `run_summary.json` + `provenance.json` under `results/omnidocbench/v16/<platform>/` and update `model_card.json.artifacts` to point at them. A maintainer will review for the `verified` badge per the policy in `omnidocbench-rocm/contracts/badge-policy.md`.

## Code of conduct

Participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Be excellent to each other.
