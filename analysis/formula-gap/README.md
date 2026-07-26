# Formula-CDM gap analysis

The published result reproduces OmniDocBench v1.6 Overall **95.88** (vLLM 0.22.1)
vs the paper's 96.58. The 0.70-pt gap is entirely formula CDM (95.41 vs 97.53).
The **localization** is arithmetic-provable:

```
Overall = ((1 - text_EditDist)*100 + formula_CDM*100 + table_TEDS*100) / 3
        = ((1 - 0.0260)*100 + 95.41 + 94.82) / 3 = 95.88
```

and `(97.53 - 95.41) / 3 = 0.71` ≈ the Overall gap.

## What is NOT a serving artifact

The **attribution** — why CDM is lower — was investigated and is **model-inherent
and version-independent**: a full 1651-page A/B reproduced CDM 95.41 on both vLLM
0.19.0 and 0.22.1 (the earlier 0.19.0 run scored Overall 95.87). The gap is the
model's formula segmentation vs the GT annotation, not a vLLM-version artifact.

## Re-verifying the attribution (the 9 locked variables)

To re-run this A/B and make a stronger claim, hold these fixed so the ONLY
variable is the vLLM version / GDN prefill backend:

1. Same weights revision (`REPRO.yaml`: `65c619d…`).
2. Same images (OmniDocBench v1.6 @ `2b161d0`).
3. Same prompt (`adapter/run_adapter.py` `_PROMPT`, verbatim).
4. Same pixel preprocessing (448²–2880²).
5. Same batch size.
6. Same scoring commit (`opendatalab/OmniDocBench` @ `2b161d0`).
7. Only the vLLM version (0.19.0 vs 0.22.1) and/or the GDN prefill backend varies.
8. Save per-formula CDM and diff with `compare.py`.
9. Only claim version-attribution if 0.22.1 recovers CDM toward 97.53. (Current
   evidence says it does not → model-inherent.)

`compare.py` diffs two per-formula CDM JSON files. `sample_manifest.example.json`
records the locked variables. **Do not commit fabricated result files** — only
real per-formula outputs from an actual re-run.
