# CLM-0.4 Release Results

- Profile: **smoke-1m**
- Status: **RELEASE_SMOKE_COMPLETE**
- Base tokens: **1,000,219**
- Transactions: **192**
- Pipeline SHA-256: `cb56f7aff59770ed07df664fdf5fddc2f1f63c7d724bd7ec60fca0b4e5e3000c`

> The 1M profile is an end-to-end engineering smoke and is not a release-quality capability benchmark.

## Equal-parameter base comparison

- CLM parameters: **5,273,088**
- Dense parameters: **5,273,120**
- Difference: **32 parameters**
- CLM base Math: **18.75%**
- Dense base Math: **14.06%**
- CLM base Story: **23.44%**
- Dense base Story: **32.81%**

## After continual learning

- CLM final Math: **15.62%**
- Dense final Math: **0.00%**
- CLM final Story: **18.75%**
- Dense final Story: **4.69%**
- CLM protected retention: **95.95%**
- Dense protected retention: **104.62%**
- CLM effective commits: **79**
- Dense commits (always-finetune baseline): **192**
- CLM parameter growth: **22.76%**
- Dense parameter growth: **0.00%**

## Comparison visualizations

- `visualizations/base-capability-clm-vs-dense.png`
- `visualizations/math-retention-clm-vs-dense.png`
- `visualizations/story-retention-clm-vs-dense.png`
- `visualizations/protected-retention-clm-vs-dense.png`

## 30M readiness

- Status: **READY_FOR_30M**
- analysis_complete: **PASS**
- asset_budget_valid: **PASS**
- clm_complete: **PASS**
- clm_parameter_identity: **PASS**
- dense_complete: **PASS**
- dense_parameter_identity: **PASS**
- profile_is_smoke_1m: **PASS**
