# Stage 07 — Safe Model Evolution

## Scope

This stage treats CLM as a substrate for controlled model evolution rather than assuming that a Cell is a natural semantic knowledge atom.

The active questions are model-level and operational:

- can a pretrained model expose a sufficiently local writable coordinate;
- can new behavior be acquired without unacceptable damage to withheld old behavior;
- how much historical information is required to discover and protect that coordinate;
- can a real post-training knowledge domain be acquired as a bounded, verifiable and rollbackable mutation;
- can independently produced mutations later be composed, verified, rolled back, and merged.

## Current evidence chain

### MoE Substrate Conversion 001

A real Granite MoE checkpoint can be represented as an immutable CLM substrate without changing Hugging Face execution semantics. MoE experts remain execution primitives, not CLM Cells.

### MoE Mutation 001

Whole-expert mutation showed strong writeability but insufficient locality: the target task was acquired while control behavior degraded. This rejected whole-expert granularity as a safe default write boundary under that protocol.

### Functional Boundary Oracle 001

With explicit frozen-base historical supervision, a 32-channel aligned sub-expert coordinate at layer 23 could be selected and trained while preserving withheld historical calibration behavior. Formal decision: `FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED` (3/3 seeds).

This supports existence of a safe sub-expert writable coordinate under historical supervision. It does not establish zero replay, autonomous Cell discovery, composition, or mergeability.

### History Compression 001

Frozen formal decision: `HISTORY_COMPRESSION_TO_8_SUPPORTED`.

- `full_32`: 3/3 PASS
- `tiny_8`: 2/3 PASS
- `tiny_2`: 0/3 PASS
- `zero_0`: 0/3 PASS

The result supports substantial compression of learner-visible historical calibration under the registered toy-task protocol, but does not support a zero-history claim. Release-oriented work therefore retains the proven `full_32` positive-control history boundary rather than optimizing for the minimum observed budget.

### JAM Knowledge Mutation 001

Current frozen release-oriented experiment.

It replaces the synthetic one-token target with `jam-knowledge-v0.1`: 180 source-locked JAM concepts derived from Gray Paper 0.8.0, trained through answer-only multi-token causal LM loss.

The writable footprint is allowed to expand conservatively through a fixed capacity ladder:

```text
1 aligned group -> 2 aligned groups -> 4 aligned groups
```

Each selected group is 32 of 512 intermediate channels and selected groups must belong to distinct experts. Capacity candidates are trained independently from the same frozen Granite base. The smallest candidate satisfying all JAM heldout, historical safety, router, rollback, artifact-reapply and standard-HF materialization gates is selected.

Formal seeds: `26090711`, `26090712`, `26090713`.

Current status: **PROTOCOL FROZEN — FORMAL GPU RUNS PENDING**.

A positive result is only a release-candidate prerequisite. Base-vs-RC general benchmarks and LoRA/PEFT baselines remain required before public model release.

## Research discipline

- Protocols and formal seeds are frozen before hosted-GPU execution.
- Withheld evaluation examples are never learner-visible or used for checkpoint selection.
- Scientific failures are preserved and published.
- Full logs are durable files; notebook stdout is compact.
- Numerical evidence lives under `artifacts/experiments/`.
- Research notebooks live under `research/notebooks/07-safe-model-evolution/`.
- Visualizations are derived from durable result files and do not replace `result.json` / `decision.json` as scientific authority.
- Experiment PRs that receive hosted formal result commits must remain open until the intended result commits are durably published.
