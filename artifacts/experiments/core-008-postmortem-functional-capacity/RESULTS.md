# Core 008 Postmortem — Functional Capacity Decomposition

- Status: `POSTMORTEM_INCOMPLETE`
- Scientific decision: `false` (diagnostic bridge on already-observed Core 008 seeds)
- Completed seeds: `[80821]`
- Missing seeds: `[80822, 80823]`

## Seed summary

| seed | classification | rank-16 per-write | PCA-32 | best factorized | Core008 oracle |
|---:|---|---:|---:|---:|---:|
| 80821 | PER_WRITE_LOW_RANK_BUT_NOT_SHARED | 0.0009 | 0.6657 | 0.7815 (r=1) | 0.8574 |

## Reading the result

Per-write SVD measures intrinsic rank. PCA-32 measures a shared linear subspace without parameter-budget matching. The factorized dictionary keeps the same 32-rank-unit budget as Core 008 but removes online allocation, certificate, and deployable routing constraints.

The 0.35 value is carried over only as an interpretive reference from Core 008; it is not a new confirmatory gate.
