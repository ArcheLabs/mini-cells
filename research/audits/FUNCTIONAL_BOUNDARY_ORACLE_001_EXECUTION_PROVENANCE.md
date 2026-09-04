# Functional Boundary Oracle 001 — Execution Path Clarification

## Status

The scientific decision remains:

```text
FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED
```

with formal seeds `26090511`, `26090512`, and `26090513` all published as PASS under one protocol hash and one consistent Tesla T4 / FP32 environment signature.

While preparing History Compression 001, the repository was re-audited because the historical `run_seed.py` engine still contains an obsolete packed-expert detector that assumes:

```text
model.config.intermediate_size == per-expert intermediate width
```

For the frozen Granite substrate, the observed layer-23 tensors are:

```text
input_linear.weight  [32, 1024, 1024]
output_linear.weight [32, 1024, 512]
```

so the per-expert intermediate width is 512.

## Canonical formal path is repaired

The important clarification is that the canonical formal runner, `scripts/research/functional_boundary_oracle_001/run_formal_seed.py`, explicitly imports `identify_packed_expert_tensors` from `src/minicells/granite_moe_layout.py` and assigns:

```python
engine._runtime_packed_parameters = identify_packed_expert_tensors
```

**before** calling `engine.run(args)`.

Therefore the successful formal artifacts do not require an uncommitted Kaggle-only scientific patch. The committed formal execution path already overrides the obsolete bare-engine detector with the correct packed-geometry inference.

The historical bare `run_seed.py` should not be treated as a standalone canonical formal entry point. `run_formal_seed.py` is the canonical formal entry point for Oracle 001.

## Evidence retained

The durable formal artifacts retain:

- exact model revision and Conversion 001 identity;
- one protocol SHA-256 across all three seeds;
- consistent environment signatures;
- full per-seed metrics and gates;
- serialized sparse mutation artifacts;
- exact selected layer/expert/group geometry;
- fresh-base mutation reapply/router verification records.

The mutation manifests record a 512-wide expert and a 32-channel group, consistent with the formal wrapper's geometry path.

## Prospective cleanup

1. Current layout inference is centralized in `src/minicells/granite_moe_layout.py` and derives roles from actual packed tensor geometry rather than `model.config.intermediate_size`.
2. History Compression 001 directly calls that helper and does not rely on monkey-patching the obsolete Oracle engine detector.
3. Oracle 001 remains frozen scientifically; only non-scientific console logging may be compacted prospectively.
4. Future formal experiments should keep the canonical engine path explicit so the standalone helper implementation and formal wrapper cannot appear to disagree.

This is an execution-path clarification, not a scientific relabeling or provenance failure.
