# Stage 07 — Safe Model Evolution

## Scope

This stage treats CLM as a substrate for controlled model evolution rather than assuming that a Cell is a natural semantic knowledge atom.

The active questions are model-level and operational:

- can a pretrained model expose or host a sufficiently local writable coordinate;
- can new behavior be acquired without unacceptable damage to withheld old behavior;
- how much historical information is required to discover and protect that coordinate;
- can a real post-training knowledge domain be acquired as a bounded, verifiable and rollbackable mutation;
- can a mature model grow new semantically addressable functional Cells rather than treating pretrained Expert/channel boundaries as Cells;
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

All three seeds selected the same ranked writable prefix. Across seeds, capacity growth consistently improved validation and overall JAM heldout NLL while maintaining low registered-history KL, router identity, rollback, and artifact-reapply behavior. The registered failure was reproducible: the misconception reference-answer NLL gain remained below the frozen `0.25` gate for every capacity and seed.

### JAM Knowledge Mutation 001 Failure Diagnostic

The forward-only post-hoc diagnostic is complete. It preserved the frozen JAM001 No-Go and decomposed all nine published mutations (`3 seeds × capacities 1/2/4`) into formal prefix, canonical JAM content, and EOS contributions.

Classification:

`CANONICAL_CONTENT_GAIN_ALSO_BELOW_ORIGINAL_THRESHOLD_AT_CAPACITY4`

The original prefix-dilution hypothesis was rejected. At capacity 4, the answer prefix and EOS contributed positive gain while canonical JAM content had negative mean NLL gain. A training-style `No.` reference could exceed the old full-answer threshold even though canonical content still worsened. The correct interpretation is therefore not that the write channel is inert, but that the fixed sparse pretrained coordinate preferentially learns cheap output/template behavior rather than a persistent semantic write address.

This diagnostic, combined with Core 002/009C and the positive Constructive CLM formation/growth/composition results, motivates a missing intermediate stage: functional Cell formation on top of a mature frozen model.

### CLM Conversion Kill Test 001 — Mature MoE Functional Cellization

Current protocol status: **PROTOCOL FROZEN — GPU FORMAL EXECUTION PENDING**.

Foundation:

- `ibm-granite/granite-3.1-1b-a400m-base`
- revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- foundation parameters remain frozen.

Conversion 001 does not search for a pretrained Expert/channel Cell. It installs a zero-output plastic overlay whose Cell is:

```text
semantic read key
+ cross-layer low-rank residual transform
+ independently mutable state
```

The registered write sites are layers `7 / 15 / 23`; routing is read once and reused across sites in the same forward pass. Zero initialization makes installation a compatibility shell before Cell learning.

The controlled world uses fictional entities and nonce protocol/region codes. Training, checkpoint validation, and final heldout direct/negation/relation/routing families are separate. The experiment then tests:

1. zero-init foundation compatibility;
2. functional Cell formation and cross-paraphrase routing persistence;
3. one-Cell semantic rewrite with unrelated-knowledge locality;
4. parent-to-child contextual growth under conflict;
5. two disjoint branch mutations, merge retention, and exact rollback.

Formal support requires at least 2 of 3 untouched seeds to pass every registered gate. A PASS would establish only the registered mature-MoE mechanism bridge; matched LoRA/static-routed-adapter baselines remain a required follow-up before any independent CLM advantage is claimed.

## Research discipline

- Protocols and formal seeds are frozen before hosted-GPU execution.
- Training, checkpoint-validation, and final heldout examples are separated when checkpoint selection is part of the protocol.
- Scientific failures are preserved and published.
- Post-hoc diagnostics may explain a failure mode but may not rewrite the frozen formal decision.
- Full logs are durable files; notebook stdout is compact.
- Numerical evidence lives under `artifacts/experiments/`.
- Research notebooks live under `research/notebooks/07-safe-model-evolution/`.
- Visualizations are derived from durable result files and do not replace `result.json` / `decision.json` as scientific authority.
- Experiment PRs that receive hosted result commits must remain open until the intended result commits are durably published.
