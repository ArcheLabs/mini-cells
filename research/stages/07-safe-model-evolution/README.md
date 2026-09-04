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

Frozen formal decision: `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED` (0/3 formal seeds PASS).

The experiment replaced the synthetic one-token target with `jam-knowledge-v0.1`: 180 source-locked JAM concepts derived from Gray Paper 0.8.0, trained through answer-only multi-token causal LM loss. The writable footprint used the frozen capacity ladder:

```text
1 aligned group -> 2 aligned groups -> 4 aligned groups
```

All three seeds selected the same ranked writable prefix. Across seeds, capacity growth consistently improved validation and overall JAM heldout NLL while maintaining low registered-history KL, router identity, rollback, and artifact-reapply behavior. The registered failure was reproducible: the misconception reference-answer NLL gain remained below the frozen `0.25` gate for every capacity and seed, while the other JAM family gains and registered safety gates were otherwise stable enough to isolate this bottleneck.

The formal No-Go is preserved. It does not by itself distinguish insufficient sparse knowledge capacity from an evaluation-formulation effect because misconception training and heldout references use different answer prefixes (`No.` versus `The claim is incorrect.`).

### JAM Knowledge Mutation 001 Failure Diagnostic

Current post-hoc diagnostic. It does not retrain the model, alter the canonical dataset, change any formal gate, or revise the frozen JAM001 decision.

The diagnostic re-applies all nine already-published mutation artifacts (`3 seeds × capacities 1/2/4`) and exactly decomposes the original misconception reference-answer token NLL into:

```text
formal answer prefix
+ canonical JAM content
+ EOS
```

It first requires reproduction of the original full misconception NLL, then reports continuous prefix/content/EOS gains and a paired training-style-prefix counterfactual. The original `0.25` threshold is used only as a reference line; no new post-hoc PASS/FAIL gate is introduced.

Current status: **ENGINEERING READY — DIAGNOSTIC GPU RUN PENDING**.

Dedicated diagnostic guard has passed source/provenance validation, compilation, lint, and CPU tests. The remaining hosted work is one forward-only diagnostic run; it performs no training.

## Research discipline

- Protocols and formal seeds are frozen before hosted-GPU execution.
- Withheld evaluation examples are never learner-visible or used for checkpoint selection.
- Scientific failures are preserved and published.
- Post-hoc diagnostics may explain a failure mode but may not rewrite the frozen formal decision.
- Full logs are durable files; notebook stdout is compact.
- Numerical evidence lives under `artifacts/experiments/`.
- Research notebooks live under `research/notebooks/07-safe-model-evolution/`.
- Visualizations are derived from durable result files and do not replace `result.json` / `decision.json` as scientific authority.
- Experiment PRs that receive hosted result commits must remain open until the intended result commits are durably published.
