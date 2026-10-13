# Constructive CLM-005 — Scaffold Removal / Endogenous Control

- Status: `LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED`
- Scientific decision: `True`
- Protocol SHA-256: `3188e56af52763d2de75e4d13f25c5cbff2e56b25d5b32520495148e5e65b27d`
- Completed seeds: `[90811, 90812, 90813]`
- Missing seeds: `[]`

| seed | pass | router meta | growth meta | write meta | cells | children | final sim MSE | final seq MSE | history MSE | unsafe reuse MSE | max exec frac |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 90811 | True | 1.0000 | 1.0000 | 1.0000 | 16 | 4 | 1.131e-06 | 1.342e-06 | 5.123e-32 | 2.405e-02 | 0.1855 |
| 90812 | True | 1.0000 | 1.0000 | 1.0000 | 16 | 4 | 1.249e-06 | 1.341e-06 | 2.163e-32 | 2.631e-02 | 0.1875 |
| 90813 | True | 1.0000 | 1.0000 | 1.0000 | 16 | 4 | 1.263e-06 | 1.706e-06 | 2.316e-32 | 2.957e-02 | 0.2002 |

A positive result supports a learned control-plane transition over the already-supported Cell substrate: learned pairwise routing plus learned write/grow decisions preserve the registered protected, bounded-growth and compositional invariants. The Core-005 certificate/projector remains a fixed safety primitive, and this result does not by itself establish an LLM-scale endogenous CLM.
