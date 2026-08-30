# CLM-0.4-mini — Stage 05 experiment adapter

> Status: M0 execution smoke implemented; M1 scientific experiment not yet implemented or run.

The frozen protocol lives in
[`research/validations/clm-0.4-mini-language-validation/`](../../../validations/clm-0.4-mini-language-validation/).

Reusable token-level CLM primitives live in `src/minicells/clm04mini/`. This directory is the
stage-aligned research adapter. M0 validates only execution plumbing:

`route → candidate → dependency validation → rollback → growth → commit → reuse → checkpoint → replay`

M0 uses seed `90400`, emits only `SMOKE_ONLY`, and explicitly rejects development seed `90401`
and formal seeds `90411`, `90412`, and `90413`.

Run:

```bash
python scripts/research/run.py clm-0.4-mini-m0 --device cpu
python scripts/research/report.py clm-0.4-mini-m0
```

M0 results under `results/clm-0.4-mini-m0/` are smoke artifacts, not canonical scientific evidence.
