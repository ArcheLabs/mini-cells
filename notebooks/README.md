# Hosted accelerator notebooks

This directory contains **launchers**, not canonical experiment implementations.

The invariant is:

```text
notebook = environment/bootstrap/orchestration
scripts/research = canonical experiment implementation
research/validations = frozen scientific protocol and durable decision record
artifacts/experiments = published per-seed evidence
```

Do not move training logic, scientific gates, thresholds, aggregation rules, or result interpretation into a notebook. A hosted notebook may install dependencies, expose accelerator information, load secrets, check out a branch, invoke a canonical runner, and publish completed artifacts.

## Platforms

- `kaggle/` — primary hosted GPU execution path for current MiniCells research.
- `colab/` — reserved for equivalent launchers when Colab is materially useful.

Every long-running hosted experiment should publish each completed seed before beginning the next seed. Ephemeral notebook storage must never be the only copy of a completed scientific result.
