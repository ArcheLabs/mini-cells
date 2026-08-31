# CLM-0.4 Preview — Public Metrics

> Status: **PREVIEW_COMPLETE**

## Model

- Cells: **136** (64 base + 72 grown)
- Parameters: **6,473,472**

## Learning

- Transactions: **192**
- Commits: **79** (41.1%)
- Growth rescue: **83.7%** (36/43)
- Private reuse acceptance: **23.7%**
- Protected probes: **2080**

## Safety / locality

- False-safe rate: **1.27%**
- Maximum structural escape: **7.41%**
- Mean direct dependency coverage: **23.19%**
- Final protected token accuracy: **54.89%**

## Growth / cost

- Private bundles: **36**
- Parameter overhead from growth: **22.76%**
- Base training: **51.4s**
- Continual candidate training: **214.5s**
- Validation: **502.8s**

## Visualizations

- `visualizations/cells-growth.png`
- `visualizations/learning-decisions.png`
- `visualizations/learning-vs-regression.png`
- `visualizations/dependency-coverage.png`
- `visualizations/compute-cost.png`
- `visualizations/parameter-growth.png`
- `visualizations/capability-over-time.png`
- `visualizations/cell-activity-top20.png`

These are Preview telemetry metrics, not a formal scientific claim.
