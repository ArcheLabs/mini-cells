# CLM-0.4-mini — Stage 05 experiment adapter

> Status: M0 execution smoke complete; M1 infrastructure implemented; calibration harness frozen; development seed `90401` not yet run.

The frozen protocol lives in
[`research/validations/clm-0.4-mini-language-validation/`](../../../validations/clm-0.4-mini-language-validation/).

Reusable token-level CLM primitives live in `src/minicells/clm04mini/`. This directory is the stage-aligned research adapter.

## M0 — execution smoke

M0 validates only the minimal transaction plumbing:

`route → candidate → dependency validation → rollback → growth → commit → reuse → checkpoint → replay`

M0 uses seed `90400`, emits only `SMOKE_ONLY`, and explicitly rejects development seed `90401` and formal seeds `90411`, `90412`, and `90413`.

```bash
python scripts/research/run.py clm-0.4-mini-m0 --device cpu
python scripts/research/report.py clm-0.4-mini-m0
```

## M1 infrastructure

The M1 implementation preserves the frozen formal architecture and provides the ~5M model, deterministic tokenizer/data manifests, the 30M-token base corpus, the fixed 192-transaction Math + Story curriculum, exact dependency validation, hidden global oracle measurement, transactional variants, growth/reuse, checkpoints, state hashes, and M1 gate evaluation.

Infrastructure smoke remains seed `90400` only:

```bash
python scripts/research/run.py clm-0.4-mini-m1 --smoke --device cpu
python scripts/research/report.py clm-0.4-mini-m1
```

The seed-independent formal data identity was generated with TinyStories revision:

```text
f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
```

and is frozen in `calibration-assets.json` before `90401` is opened.

## M1 calibration harness

Calibration is a development-only selection step, not a scientific decision. The repository freezes:

- development seed `90401` only;
- 81 direct/growth optimizer combinations from the registered grids;
- deterministic ordering by `(direct_steps + growth_steps, direct_steps, growth_steps, direct_lr, growth_lr)`;
- first-pass-stop selection;
- one base-model training run/checkpoint reused unchanged by every candidate;
- 64 math and 64 story base-prerequisite evaluation examples;
- minimum base Cell activation as half of the uniform Top-k expectation;
- exact pre-90401 tokenizer/base-corpus/curriculum hashes.

Validate the plan without opening `90401`:

```bash
python scripts/research/run.py clm-0.4-mini-calibration --plan-only
python scripts/research/report.py clm-0.4-mini-calibration
```

After this calibration-harness PR is merged, run the dedicated Kaggle notebook:

```text
research/notebooks/05-language-validation/clm-0.4-mini-m1-calibration.ipynb
```

The real calibration command is intentionally explicit:

```bash
python scripts/research/run.py clm-0.4-mini-calibration \
  --data-dir /kaggle/working/clm-0.4-mini-data \
  --out /kaggle/working/clm-0.4-mini-calibration \
  --device cuda \
  --seed 90401 \
  --confirm-development-seed 90401
```

If a configuration passes, calibration emits `protocol-lock.candidate.json`. That file must be reviewed and committed separately as the canonical `protocol-lock.json` before any formal seed can run.

## Scientific boundary

M0, infrastructure smoke, plan-only validation, and `90401` calibration do **not** produce the M1 scientific status.

Formal seeds `90411`, `90412`, and `90413` remain forbidden until a canonical committed protocol lock exists. If no registered calibration candidate passes, the grid must not be expanded post hoc; revise the protocol before any formal execution.
