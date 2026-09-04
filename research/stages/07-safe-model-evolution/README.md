# Stage 07 — Safe Model Evolution

## Scope

This stage treats CLM as a substrate for controlled model evolution rather than assuming that a Cell is a natural semantic knowledge atom.

The active questions are model-level and operational:

- can a pretrained model expose a sufficiently local writable coordinate;
- can new behavior be acquired without unacceptable damage to withheld old behavior;
- how much historical information is required to discover and protect that coordinate;
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

Current frozen experiment. It reduces only the learner-visible historical prompt budget while holding the substrate, writable granularity, optimizer family, and withheld safety gates fixed:

```text
full_32 -> tiny_8 -> tiny_2 -> zero_0
```

The goal is to identify the smallest observed historical prompt budget that remains supported. Historical-certificate mechanisms such as Fisher or low-rank gradient sketches are intentionally deferred until this replay-budget boundary is measured.

## Research discipline

- Protocols and formal seeds are frozen before hosted-GPU execution.
- Withheld evaluation prompts are never learner-visible.
- Scientific failures are preserved and published.
- Full logs are durable files; notebook stdout is compact.
- Numerical evidence lives under `artifacts/experiments/`.
- Research notebooks live under `research/notebooks/07-safe-model-evolution/`.
- Visualizations are derived from durable result files and do not replace `result.json` / `decision.json` as scientific authority.
