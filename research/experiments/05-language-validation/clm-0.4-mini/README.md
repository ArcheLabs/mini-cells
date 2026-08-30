# CLM-0.4-mini — Stage 05 experiment adapter

> Status: M0 execution smoke complete; M1 infrastructure implemented; development calibration not yet run.

The frozen protocol lives in
[`research/validations/clm-0.4-mini-language-validation/`](../../../validations/clm-0.4-mini-language-validation/).

Reusable token-level CLM primitives live in `src/minicells/clm04mini/`. This directory is the
stage-aligned research adapter.

## M0 — execution smoke

M0 validates only the minimal transaction plumbing:

`route → candidate → dependency validation → rollback → growth → commit → reuse → checkpoint → replay`

M0 uses seed `90400`, emits only `SMOKE_ONLY`, and explicitly rejects development seed `90401`
and formal seeds `90411`, `90412`, and `90413`.

```bash
python scripts/research/run.py clm-0.4-mini-m0 --device cpu
python scripts/research/report.py clm-0.4-mini-m0
```

## M1 infrastructure

The M1 implementation preserves the frozen formal architecture and provides:

- executable protocol/seed guards;
- the ~5M formal model configuration;
- deterministic byte-level BPE tokenizer manifests;
- manifest-driven 30M-token base-corpus sharding;
- deterministic 192-transaction Math + Story curriculum;
- answer-only teacher-forced certification examples;
- exact-key story supersede semantics;
- `local_always`, `local_tx`, and `local_tx_growth` from one base checkpoint;
- exact dependency indexing and hidden full-history oracle measurement;
- real candidate pass/rollback/growth decisions without M0 path overrides;
- registered M1 gate evaluation;
- Cell registry, transaction journal, checkpoints, and hash replay;
- protocol-lock construction after development calibration.

The infrastructure smoke still uses seed `90400` and a reduced model/data projection. It exists only to verify the formal interfaces before seed `90401` is opened:

```bash
python scripts/research/run.py clm-0.4-mini-m1 --smoke --device cpu
python scripts/research/report.py clm-0.4-mini-m1
```

Formal data assets are generated outside Git and identified by cryptographic manifests. A pinned TinyStories revision is mandatory:

```bash
python scripts/research/prepare_clm_0_4_mini_data.py \
  --dataset-revision <PINNED_REVISION> \
  --routing-salt clm-0.4-mini-v1 \
  --out <DATA_DIR>
```

## Scientific boundary

Neither M0 nor the M1 infrastructure smoke is a scientific run. Their output status is always `SMOKE_ONLY`.

The next allowed scientific step after infrastructure validation is **development seed `90401` calibration** over the already registered finite optimizer grids. A committed `protocol-lock.json` must then exist before any of `90411`, `90412`, or `90413` may run.
