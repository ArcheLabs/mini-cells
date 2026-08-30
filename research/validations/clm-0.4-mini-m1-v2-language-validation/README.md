# CLM-0.4-mini M1-v2 Language Validation

Status: **PROTOCOL_FROZEN_DATA_LOCK_PENDING**

M1-v2 is an explicit revision after v1 development seed `90401` stopped at
`CALIBRATION_BASE_PREREQUISITES_FAILED`. V1 evidence is preserved in
`v1-development-failure.json`; it is not overwritten or reinterpreted as a
formal scientific result.

## What v2 changes

V2 changes only the base-admission/data-alignment and comparison protocol:

- controlled Story training is context-conditioned retrieval QA, matching base evaluation;
- controlled Math uses question/answer forms across the same pre-continual families;
- the primary base admission metric is teacher-forced per-answer exactness at `>= 0.85`;
- greedy exact match, answer-token accuracy, and answer NLL remain diagnostics;
- 64 held-out addresses per base domain are route-balanced so every one of 32
  base Cells in both sparse layers is covered;
- development calibration moves to fresh seed `90402`;
- formal seeds remain `90411/90412/90413`;
- Dense-5M equal-parameter and dense equal-active-FFN-compute baselines are added;
- dense baselines are diagnostic and cannot control CLM commit, growth, candidate
  selection, or the formal decision.

The continual curriculum, dependency index, transaction semantics, 81-candidate
grid, first-pass-stop rule, M1 gates, growth mechanism, and formal seed set remain
unchanged.

## Dense controls

Executable parameter contracts:

| Model | FFN layout | Parameters | Role |
|---|---|---:|---|
| CLM | L1/2 dense-768; L3/4 32×Cell-32 Top-2 | 5,009,920 | primary |
| Dense equal-parameter | 4×dense-904 | 5,010,464 | static + continual diagnostic |
| Dense equal-active-compute | 4×dense-416 | 4,009,088 | static diagnostic |

The equal-active-compute baseline has about 854,656 active FFN parameters per
token versus about 855,168 for CLM.

## Two-stage development lock

`90402` must not be opened immediately after this implementation lands.

1. Run seed-independent v2 data preparation with the pinned TinyStories revision.
2. Record `asset-summary.json`.
3. Replace the null fields in `asset-lock.json`, set `lock_status=LOCKED`, and
   commit that lock.
4. Only then run v2 calibration with explicit `--confirm-development-seed 90402`.

This separates implementation decisions from development-seed observation.

## Result publication

Calibration results are curated into Git without model checkpoints or raw
30M-token shards:

```bash
python scripts/research/publish.py clm-0.4-mini-m1-v2-calibration \
  --results /path/to/clm-0.4-mini-m1-v2-calibration \
  --push
```

The default result branch is:

```text
kaggle/clm-0.4-mini-m1-v2-90402-results
```

The same publisher also supports the preserved v1 failure:

```bash
python scripts/research/publish.py clm-0.4-mini-m1-v1-calibration \
  --results /path/to/clm-0.4-mini-calibration \
  --push
```

Publishing a development result does not authorize formal execution.
