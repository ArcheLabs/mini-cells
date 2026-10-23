# R3d-A — Checkpoint Read-Only Mechanism Audit

Status: PARTIAL
Completed Checkpoints: 17 / 18

Training updates: 0. Optimizer instances: 0. Confirmation: SEALED.

## Closed-loop Development NLL

| Seed | F0 | L | M | F | P | J |
|---|---|---|---|---|---|---|
| 151001301 | — | 4.609699 | 4.589830 | 4.512964 | 4.512949 | 4.524557 |
| 151001302 | 4.557351 | 4.590361 | 4.599856 | 4.526828 | 4.529036 | 4.544734 |
| 151001303 | 4.545962 | 4.613321 | 4.600713 | 4.500305 | 4.502492 | 4.517769 |

## Q1–Q4

- Q1: INCONCLUSIVE
- Q2: INCONCLUSIVE
- Q3: INCONCLUSIVE
- Q4: NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS

## Next research direction

INCONCLUSIVE

Observed values, ranking associations and within-checkpoint interventions are reported separately. The affine correction can only express a shared state projection plus Stage biases; these data do not prove a richer predictor would or would not learn expected future value.

Hindsight Candidate Opportunity Gap uses true labels; it is neither a deployable Oracle Router nor a guaranteed achievable gain. Last-Stage short/long equality is a mathematical consistency check, not evidence for long-horizon learning.

Historical Train ranking is pre-update sampled-candidate telemetry. Calibration and Development rankings here enumerate all 15 actions at frozen checkpoints. Quantile levels were fixed in advance; numerical cutoffs use Calibration only.

Fixed-Checkpoint lambda interventions isolate routing effects at that saved parameter state. Differences between F/P/J final Cell weights do not identify historical training causes. Such claims remain NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS.

Only three figures are produced. No training, hyperparameter selection, new model, or R3e run is performed. Private raw scores/indices and Checkpoint caches stay outside public GitHub.

## Missing or failed work

[{'arm': 'F0', 'code': 'FileNotFoundError', 'http_status': None, 'seed': 151001301, 'status': 'ANALYSIS_FAILED_CHECKPOINT_NOT_SUBSTITUTED', 'step': 0, 'type': 'FileNotFoundError'}]
