# Integrated Replay-Free CLM Kill Test 001 (KT001)

Status: **FROZEN_FOR_IMPLEMENTATION**

This protocol defines the causal structure of KT001. It is intentionally frozen before the runner is implemented. Formal execution MUST NOT begin until the implementation, tests, publisher, and formal-seed guard are complete.

## Scientific question

KT001 asks whether the replay-free combination of:

1. realized AdamW write-transaction safety,
2. persistent historical address state, and
3. lineage-preserving read/growth isolation

produces a continual-learning benefit that cannot be explained by any one component in isolation.

The experiment is a kill test. A positive result requires the integrated replay-free arm to retain useful plasticity while preserving historical behavior substantially better than the unsafe and single-mechanism controls. The matched replay arm is an oracle comparator, not part of the replay-free claim.

## Canonical mechanisms

KT001 MUST reuse the repository's existing mechanisms rather than approximate them:

- **R0b write mechanics:** canonical `adamw_final_update_projection` semantics. In write-safe arms, safety is applied to the realized AdamW parameter delta after every optimizer step; gradient projection alone is not an acceptable substitute.
- **M3L-2 historical address state:** the existing rank-32 persistent online address-state/checkpoint representation is the historical read substrate.
- **M3R lineage mechanics:** the existing lineage-preserving routing/read-growth isolation is the structural isolation substrate.
- **Shadow expansion:** phase boundaries force creation/activation of the Shadow path according to the integrated runner's frozen phase schedule. The expansion event is not triggered by evaluation outcomes.

Any incompatibility discovered while integrating these mechanisms MUST be fixed explicitly and documented; silently replacing a mechanism with a similar implementation invalidates the protocol.

## Five causal arms

The formal comparison contains exactly these arms:

| Arm | Realized-update write safety | Historical address read | Lineage isolation / Shadow growth | Raw replay |
| --- | --- | --- | --- | --- |
| `unsafe` | no | no | no integrated protection | no |
| `write_transaction_only` | yes | no | no historical-read benefit | no |
| `read_history_only` | no | yes | yes where required for the canonical read path | no |
| `full_no_replay` | yes | yes | yes | no |
| `matched_replay_oracle` | matched to the integrated comparator | matched as required by the comparator | matched as required by the comparator | yes |

Arm-specific switches MUST be explicit in machine-readable run metadata. Hidden differences in optimizer, data order, training budget, evaluation frequency, model initialization, phase boundaries, or model capacity are prohibited unless the difference is intrinsic to the named mechanism and is recorded as such.

## Matched experimental budget

Across arms and within a seed, the implementation MUST match as closely as mechanically possible:

- foundation checkpoint and initialization,
- tokenizer and data materialization,
- continual phase order,
- optimizer hyperparameters,
- number of optimizer steps,
- per-step batch/token budget,
- evaluation checkpoints,
- random-stream policy,
- baseline parameterization before forced Shadow expansion.

Where Shadow expansion changes capacity, the runner MUST record the exact phase, parameter count, trainable-parameter count, lineage state, and address-state checkpoint before and after expansion.

## Replay boundary

`full_no_replay` and all non-oracle controls MUST NOT access raw examples from completed continual phases after those phases close. Historical information may survive only through the frozen model/optimizer state and the explicitly permitted persistent structural/history state inherited from the canonical mechanisms.

`matched_replay_oracle` may access its frozen replay buffer only through the oracle path defined by the runner. Oracle replay volume and sampling policy MUST be recorded so that the comparison is auditable.

## Realized AdamW safety invariant

For every optimizer step in a write-safe arm:

1. compute the ordinary AdamW candidate update using the canonical optimizer state;
2. obtain the realized candidate parameter delta;
3. project/correct that realized delta with the canonical R0b final-update safety operator;
4. install the corrected parameters;
5. preserve/update optimizer state according to the canonical R0b mechanics;
6. record sufficient diagnostics to verify that the installed delta, not merely the gradient, satisfied the safety transaction.

A runner that protects gradients but permits the realized AdamW update to escape the protected subspace is non-conforming.

## Phase-boundary invariant

Shadow expansion is forced by the protocol at the frozen phase boundary. Evaluation metrics MUST NOT decide whether expansion occurs in KT001. This removes a lifecycle-policy confound from the causal test.

The runner MUST checkpoint immediately before and immediately after each forced expansion and record lineage and historical-address state hashes.

## Seed isolation

The canonical development and formal seeds are defined only in `SEEDS.json` in this protocol directory.

Formal seeds MUST NOT be used by:

- unit tests,
- smoke tests,
- debugging,
- implementation development,
- threshold selection,
- protocol tuning.

Formal seeds are single-use scientific evaluation seeds once the formal protocol is sealed. Any accidental formal-seed execution before sealing MUST be reported and the formal registry replaced before scientific execution.

## Required per-run evidence

Every run MUST emit machine-readable provenance sufficient to reconstruct and audit the comparison, including at minimum:

- experiment id and arm,
- seed and phase,
- git commit SHA,
- protocol SHA-256,
- seed-registry SHA-256,
- model/checkpoint provenance,
- data-manifest identity,
- optimizer configuration,
- replay policy and replay volume,
- address-state rank and checkpoint/hash,
- lineage/Shadow state and checkpoint/hash,
- realized-update safety diagnostics for protected arms,
- evaluation metrics at all frozen checkpoints,
- parameter/trainable-parameter counts,
- failure record if a run exits incompletely.

## Aggregation and decision discipline

Formal aggregation MUST consume completed formal-seed artifacts only. It MUST reject:

- mixed protocol hashes,
- mixed seed-registry hashes,
- missing arms,
- incomplete formal seeds,
- artifacts produced by a different git implementation without explicit protocol-compatible provenance.

No formal acceptance threshold is invented in this document. If a numerical decision rule already exists in the canonical prior protocol, the implementation MUST reuse it verbatim; otherwise the rule must be frozen in a later protocol-only commit before any formal seed is executed.

## Implementation sequence

Implementation proceeds in small independently pushed commits:

1. frozen protocol;
2. seed registry and formal-seed guard data;
3. integrated five-arm runner;
4. aggregator and publisher;
5. tests and execution guards;
6. Kaggle/CI entry points and final documentation.

Formal seeds remain untouched throughout implementation and smoke validation.
