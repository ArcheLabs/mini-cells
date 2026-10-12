# CLM-0.4-mini M1-v2 — Stage 05 adapter

This adapter implements the v2 development protocol after v1 seed `90401`
exposed base-task/evaluation misalignment.

## Boundary

- v1 `90401`: historical development diagnosis; never reused as the clean v2 seed.
- v2 `90402`: development calibration only.
- `90411/90412/90413`: formal seeds; forbidden until the v2 protocol lock is committed.

## Phase A — seed-independent data

```bash
python scripts/research/prepare_clm_0_4_mini_v2_data.py \
  --dataset-revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --out /kaggle/working/clm-0.4-mini-m1-v2-data
```

Commit the resulting hashes into the validation `asset-lock.json`. Calibration
will refuse `90402` while the lock is `DATA_LOCK_PENDING`.

## Phase B — plan-only

```bash
python scripts/research/run.py clm-0.4-mini-m1-v2-calibration \
  --plan-only \
  --out /tmp/clm-v2-plan
```

## Phase C — development calibration, after data lock

```bash
python scripts/research/run.py clm-0.4-mini-m1-v2-calibration \
  --data-dir /kaggle/working/clm-0.4-mini-m1-v2-data \
  --out /kaggle/working/mini-cells/results/clm-0.4-mini-m1-v2-calibration \
  --device cuda \
  --devices cuda:0,cuda:1 \
  --seed 90402 \
  --confirm-development-seed 90402
```

Run the report and publish the curated result branch:

```bash
python scripts/research/report.py clm-0.4-mini-m1-v2-calibration \
  --results /kaggle/working/mini-cells/results/clm-0.4-mini-m1-v2-calibration

python scripts/research/publish.py clm-0.4-mini-m1-v2-calibration \
  --results /kaggle/working/mini-cells/results/clm-0.4-mini-m1-v2-calibration \
  --push
```

Dense baselines are diagnostics. They do not alter the CLM candidate order or gates.
