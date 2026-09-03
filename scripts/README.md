# Repository Scripts

Scripts are thin operational entrypoints; reusable model, evaluation, release, and runtime logic belongs in `src/minicells`.

- `research/`: scientific run/report/publish/orchestration entrypoints. See [`research/README.md`](research/README.md) for the compatibility taxonomy and path-stability policy. The flat named adapters are retained for historical reproducibility; new work should prefer unified dispatch or a family package.
- `release/`: model/release construction and release-benchmark tasks.
- `runtime/`: performance and sparse-runtime tasks.
- `maintenance/`: repository catalog, artifact, link, branch, and architecture integrity utilities.

Preferred research interfaces when the family is registered with the dispatcher:

```bash
python scripts/research/run.py core-validation-004 --smoke
python scripts/research/report.py core-validation-004
python scripts/research/publish.py core-validation-004
```

Do not infer scientific status from script placement or filename. Canonical protocols/results live under `research/validations/`, and cross-experiment claim boundaries live under `research/audits/`.
