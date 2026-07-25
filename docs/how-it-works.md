# How it works

`OvisOCR2-ROCm` is a per-model adapter repo for the **omnidocbench-rocm** engine.
The engine drives the OmniDocBench v1.6 pipeline; this repo supplies only the
model-specific inference step — OvisOCR2 via in-process vLLM on ROCm.

## The contract

One function:

```python
def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict) -> dict:
```

For each image in `img_dir` it writes `out_dir/<image_stem>.md` and records a
`PageStatus` (`ok` / `failed: <reason>`). Per-page failures are caught and
recorded — the run never raises (a missing page scores zero downstream).
Finally it writes a schema-valid `_run_stats.json` (`RunSummary.write`) and
returns `RunSummary.to_run_stats()`. The engine consumes only those artifacts —
it never imports the adapter.

## How OvisOCR2 is called

`backend == "vllm"` loads `vllm.LLM(model=<weights>, ...)` once per process,
builds the official OvisOCR2 chat prompt, and runs **batched `generate()`** over
the whole page list (vLLM schedules batches internally). Each output is
post-processed exactly as the upstream model card's `OvisOCR2Parser`:

1. Drop visual-region `<img src="images/bbox_*.jpg">` tags (card default).
2. `_clean_truncated_repeats` — trims the repetitive tails the model can emit on
   hard pages (verbatim from the card).

Sampling is greedy (`temperature=0.0`, `max_tokens=16384`); pixels are clamped to
`448²–2880²`. `--skip-existing` resumes an interrupted run.

## Backends

| `backend` | what it does | GPU? |
|---|---|---|
| `vllm` | in-process vLLM/ROCm, full official recipe | yes |
| `smoke` | writes a placeholder `.md` per image | no |

See [`backends.md`](backends.md).

## Stages (engine-side)

`make eval-linux` (`omnidocbench-rocm run --stage all`) runs:

1. **download** — fetch the pinned OmniDocBench v1.6 dataset (`2b161d0`).
2. **infer** — invoke `adapter/run_adapter.py` over the dataset images.
3. **score** — Edit_dist / TEDS / CDM against gold answers (eval-venv 3.11).
4. **publish** — assemble + schema-validate `run_summary.json` + `provenance.json`
   into `results/omnidocbench/v16/linux-rocm/` (full-set enforced).

`make publish` runs `omnidocbench-rocm conformance .` to verify the repo still
satisfies the contract.
