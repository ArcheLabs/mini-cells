# Repository Scripts

Scripts are thin operational entrypoints; reusable model, evaluation, release, and runtime logic belongs in `src/minicells`.

- `research/`: run, report, and publish scientific experiments. The unified `run.py`, `report.py`, and `publish.py` interfaces cover current Core Validations; retained named adapters preserve historical reproducibility.
- `release/`: model/release construction and release-benchmark tasks.
- `runtime/`: performance and sparse-runtime tasks.
- `maintenance/`: repository catalog, artifact, link, branch, and architecture integrity utilities.

Examples:

```bash
python scripts/research/run.py core-validation-004 --smoke
python scripts/research/report.py core-validation-004
python scripts/research/publish.py core-validation-004
```
