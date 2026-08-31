# CLM-0.4 Release Results

- Profile: **release-30m**
- Status: **RELEASE_30M_COMPLETE**
- Base tokens: **30,000,017**
- Transactions: **192**
- Pipeline SHA-256: `cb56f7aff59770ed07df664fdf5fddc2f1f63c7d724bd7ec60fca0b4e5e3000c`

> The 30M profile is the CLM-0.4 release benchmark/output.

## Equal-parameter base comparison

- CLM parameters: **5,273,088**
- Dense parameters: **5,273,120**
- Difference: **32 parameters**
- CLM base Math: **92.19%**
- Dense base Math: **100.00%**
- CLM base Story: **92.19%**
- Dense base Story: **100.00%**

## After continual learning

- CLM final Math: **92.19%**
- Dense final Math: **0.00%**
- CLM final Story: **92.19%**
- Dense final Story: **0.00%**
- CLM protected retention: **98.03%**
- Dense protected retention: **48.57%**
- CLM effective commits: **75**
- Dense commits (always-finetune baseline): **192**
- CLM parameter growth: **22.76%**
- Dense parameter growth: **0.00%**

## Comparison visualizations

- `visualizations/base-capability-clm-vs-dense.png`
- `visualizations/math-retention-clm-vs-dense.png`
- `visualizations/story-retention-clm-vs-dense.png`
- `visualizations/protected-retention-clm-vs-dense.png`
