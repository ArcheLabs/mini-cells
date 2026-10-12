# CLM-0.4-mini M1-v2 Calibration

> Status: **V2_CALIBRATION_BASE_PREREQUISITES_FAILED** — development calibration only.

- v1 development seed `90401` is historical diagnosis only.
- v2 development seed observed: `True`
- Formal seeds `90411/90412/90413` observed: `false`

## Aligned base admission

- Math teacher-forced answer exact: `0.34375`
- Story teacher-forced answer exact: `0.96875`
- Math greedy exact (diagnostic): `0.34375`
- Story greedy exact (diagnostic): `0.96875`
- Base prerequisite pass: `False`

## Static dense controls

- `equal_active_compute` params `4009088`: math TF answer exact `0.421875`, story `1.0`
- `equal_parameter` params `5010464`: math TF answer exact `0.390625`, story `1.0`

Candidate search did not start because the aligned CLM base admission gate failed.

## Scientific boundary

Dense baselines are diagnostic-only and never control CLM commit, growth, candidate selection, or the formal scientific decision.
