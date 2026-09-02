# Native CLM v0 — M3L-1 Historical Address-State Capacity

- Classification: `LOW_RANK_CAPACITY_SUFFICIENT`
- Scientific decision: `False` (checkpoint-only mechanism diagnostic)
- Valid edges: `24/24`
- Offline oracle median AUC: `0.9281`
- Minimum passing low rank: `32`
- Full covariance passes M3L gates: `True`
- Rank-16 parent identity: `True`

## Capacity curve

| Candidate | Median AUC | >=0.85 | Recovery | Old FPR | Current TPR | Historical bytes | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| rank-0 | 0.6103 | 0.000 | 0.2625 | 0.3830 | 0.5801 | 3080 | FAIL |
| rank-8 | 0.8915 | 0.625 | 0.9182 | 0.2146 | 0.8294 | 15400 | FAIL |
| rank-16 | 0.8968 | 0.750 | 0.9356 | 0.1855 | 0.8204 | 27720 | FAIL |
| rank-32 | 0.9130 | 0.792 | 0.9561 | 0.1539 | 0.8116 | 52360 | PASS |
| rank-64 | 0.9207 | 0.917 | 0.9681 | 0.1019 | 0.7815 | 101640 | PASS |
| rank-128 | 0.9199 | 1.000 | 0.9771 | 0.0729 | 0.7305 | 200200 | PASS |
| full-covariance | 0.9254 | 1.000 | 0.9855 | 0.1122 | 0.8086 | 591368 | PASS |

## Transition medians

- `A+B->C`: rank-0=0.6414, rank-8=0.9461, rank-16=0.9442, rank-32=0.9551, rank-64=0.9621, rank-128=0.9624, full-covariance=0.9650
- `A->B`: rank-0=0.5836, rank-8=0.8383, rank-16=0.8496, rank-32=0.8610, rank-64=0.8746, rank-128=0.8777, full-covariance=0.8850
- `B->C`: rank-0=0.8407, rank-8=0.9636, rank-16=0.9787, rank-32=0.9781, rank-64=0.9784, rank-128=0.9773, full-covariance=0.9811

Interpretation:

A finite low-rank Gaussian historical address state satisfies the original M3L feasibility gates; the M3L shortfall is primarily rank/capacity limited rather than a Gaussian-family failure.

Boundary: consumed M3R checkpoints only; no Native CLM parameter update, growth, or new formal seed. Raw historical queries are reduced into each registered candidate address state before gate construction and are otherwise used only by the offline evaluator.
