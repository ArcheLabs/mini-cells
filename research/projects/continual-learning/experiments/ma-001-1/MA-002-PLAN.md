# MA-002 — independent confirmation plan (not authorized by MA-001.1)

Author 180 independent cases: 30 single-cell semantic, 30 paraphrase,
30 multi-cell, 30 hard negatives, 36 no-match and 24 ambiguous. Record weak
matches as a separate diagnostic stratum if collected; do not pool them into
the 36 no-match cases. Use distinct templates and, where feasible, distinct
target events from development cases. Freeze matching, fusion weights,
thresholds, software and fixture hashes before revealing confirmation labels.
Treat MA-001's 56 cases as observed historical evidence only.

Zero errors in 36 independent no-match cases gives a two-sided exact 95%
binomial upper bound of 9.74% (1 − 0.025^(1/36)). This calculation does not
make correlated synthetic queries statistically independent. Publish counts,
intervals, and failures by stratum. MA-003 is separate and remains pending.
