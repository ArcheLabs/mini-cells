# MOM-STATE-001 baseline

**BASELINE_MISMATCH — DEVELOPMENT_ONLY.** [Immutable run evidence](https://github.com/ArcheLabs/mini-cells/tree/kaggle/mom-state-001-9272ec8bdab2/artifacts/experiments/mom-state-001-9272ec8bdab2).

The restored MoM cache matched before execution. After D, 211 tensor
fields differed (97 recurrent and 114 convolution); max absolute difference
was 0.0078125. All 25 answer-score pairs differed, while all 25 margin
directions agreed (post-run audit); maximum margin difference was about
0.01192. This does not satisfy zero tolerance. The original receipt is
preserved. Diagnosis: divergence during computation after restore; cause
unresolved. External Cell state is outside this experiment.

Service Runtime Tag was already negative in the continuous path at ABC
and ABCD. Treat recall interference separately from restore consistency.
Full conflict, interference, and version-update acceptance remains pending.
See [MOM-STATE-DIAG-001](../mom-state-diag-001/README.md).
