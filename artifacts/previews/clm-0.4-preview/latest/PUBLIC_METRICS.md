# CLM-0.4 Preview — Public Metrics

> Status: **PREVIEW_COMPLETE**

## Model

- Cells: **136** (64 base + 72 grown)
- Parameters: **6,473,472**

## Learning

- Transactions: **192**
- Commits: **74** (38.5%)
- Growth rescue: **100.0%** (36/36)
- Private reuse acceptance: **23.9%**
- Protected probes: **1728**

## Safety / locality

- False-safe rate: **1.35%**
- Maximum structural escape: **1.92%**
- Mean direct dependency coverage: **23.45%**
- Final protected token accuracy: **78.19%**

## Growth / cost

- Private bundles: **36**
- Parameter overhead from growth: **22.76%**
- Base training: **1582.0s**
- Continual candidate training: **233.5s**
- Validation: **500.4s**

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
