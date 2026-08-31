# CLM-0.4 Preview — Public Metrics

> Status: **PREVIEW_COMPLETE**

## Model

- Cells: **136** (64 base + 72 grown)
- Parameters: **6,473,472**

## Learning

- Transactions: **192**
- Commits: **75** (39.1%)
- Growth rescue: **94.7%** (36/38)
- Private reuse acceptance: **25.3%**
- Protected probes: **1792**

## Safety / locality

- False-safe rate: **0.00%**
- Maximum structural escape: **0.00%**
- Mean direct dependency coverage: **23.45%**
- Final protected token accuracy: **72.07%**

## Growth / cost

- Private bundles: **36**
- Parameter overhead from growth: **22.76%**
- Base training: **1463.1s**
- Continual candidate training: **217.8s**
- Validation: **490.7s**

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
