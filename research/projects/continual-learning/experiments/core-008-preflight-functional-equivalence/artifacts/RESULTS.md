# Core 008 Preflight — Functional Equivalence Bridge Results

- Status: `WEAK_EFFECT_CONFOUND_DOMINATES`
- Scientific decision: `False`
- Source Core 007 status changed: `False`

This bridge uses the already-observed completed Core 007 seeds only to decide what Core 008 should test. It is not a new confirmation result.

## Seed diagnostics

| seed | Core007 reproduced | eval mode agreement | mismatch same-owner fraction | owner-mismatch normalized NLL regret (median) | owner-mismatch normalized logit difference (median) | owner-mismatch symmetric KL (median) |
|---:|:---:|---:|---:|---:|---:|---:|
| 80721 | True | 0.285714 | 0.075000 | 0.602299 | 0.99435 | 4.25444e-07 |
| 80722 | True | 0.339286 | 0.027027 | 0.512107 | 0.846058 | 3.28072e-07 |

## Interpretation

The decisive quantity is not raw mode-label agreement. A mode mismatch that retains the same final Cell owner is exactly functionally equivalent under the current architecture because both routes use the same matrix A. For true owner mismatches, route regret is normalized by the magnitude of the Cell's own intervention relative to the frozen foundation path.

`WEAK_EFFECT_CONFOUND_DOMINATES` means the tiny whole-model NLL gap cannot be used as evidence of functional equivalence: Cell interventions are themselves too small, while normalized owner-mismatch regret is not near-equivalent. `FUNCTIONAL_REDUNDANCY_EVIDENCE` means either mode labels substantially over-split identical owners or true owner mismatches remain behaviorally near-equivalent after normalization.

The resulting status is a bridge diagnostic only. Core 008 requires a fresh protocol and fresh formal seeds.
