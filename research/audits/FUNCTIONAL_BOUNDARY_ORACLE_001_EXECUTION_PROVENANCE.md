# Functional Boundary Oracle 001 — Execution Provenance Note

## Status

The scientific decision remains:

```text
FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED
```

with formal seeds `26090511`, `26090512`, and `26090513` all published as PASS under one protocol hash and one consistent Tesla T4 / FP32 environment signature.

This note records an **execution-source provenance defect** discovered while preparing History Compression 001. It does not relabel, delete, or rerun the published Oracle result.

## Defect

The final artifact-publishing commit `7d23027e6caecb5d326dc18e1bed27bde6c77b46` contains an older `scripts/research/functional_boundary_oracle_001/run_seed.py` implementation whose packed-expert detector incorrectly assumes:

```text
model.config.intermediate_size == per-expert intermediate width
```

For the frozen Granite substrate, the observed layer-23 tensors are:

```text
input_linear.weight  [32, 1024, 1024]
output_linear.weight [32, 1024, 512]
```

so the per-expert intermediate width is 512. The older detector deterministically rejects this layout when it reads the model-level value 1024.

The hosted run that produced the successful formal artifacts necessarily used the repaired packed-geometry inference available in the Kaggle runtime, but that repaired runner source was not included in the later artifact-only publishing commit. This is a source-tree/provenance mismatch, not a protocol-hash or result-data mismatch.

## Evidence retained

The durable formal artifacts retain:

- exact model revision and Conversion 001 identity;
- one protocol SHA-256 across all three seeds;
- consistent environment signatures;
- full per-seed metrics and gates;
- serialized sparse mutation artifacts;
- exact selected layer/expert/group geometry;
- fresh-base mutation reapply/router verification records.

The mutation manifests themselves record a 512-wide expert and a 32-channel group, consistent with the repaired geometry path.

## Remediation

1. Current layout inference is centralized in `src/minicells/granite_moe_layout.py` and derives roles from the actual packed tensor geometry rather than `model.config.intermediate_size`.
2. The Oracle formal wrapper is repaired prospectively so a future diagnostic rerun uses the geometry helper even though the historical `run_seed.py` source remains part of the original lineage.
3. History Compression 001 directly imports the geometry helper and does not call the defective Oracle packed-tensor detector.
4. History Compression 001 freezes a new protocol and new formal seeds; it does not treat the Oracle source tree as an exact executable dependency.

## Scientific interpretation boundary

This provenance defect lowers confidence in **bit-for-bit source-tree replay of the historical hosted run** unless the repaired layout path is applied. It does not by itself invalidate the numerical 3/3 Oracle evidence, because the failure mechanism is known, deterministic, and isolated to model-layout identification before the scientific optimization/evaluation steps.

Future formal experiments must publish code and results from the same checked-out commit lineage. Hosted notebook state must not contain uncommitted scientific runner changes when artifacts are published.
