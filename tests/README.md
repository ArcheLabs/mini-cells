# Test Organization

- `unit/`: isolated implementation behavior.
- `integration/`: multi-component release and runtime behavior.
- `research/`: scientific implementation regression checks, grouped by research stage.
- `smoke/`: explicitly low-cost execution checks that never emit formal decisions.

Run a hierarchy directly:

```bash
pytest tests/unit
pytest tests/integration
pytest tests/research
pytest tests/smoke
```

Configured markers are `unit`, `integration`, `research`, `smoke`, and `gpu`. Formal CUDA experiments are not part of ordinary test execution.
