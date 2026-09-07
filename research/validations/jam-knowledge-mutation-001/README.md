# JAM Knowledge Mutation 001

Status: **PROTOCOL FROZEN — FORMAL GPU RUNS PENDING**

Protocol version: **1.2**.

This is the first release-oriented real-domain mutation in the Safe Model Evolution line.

## Question

Can the frozen Granite 3.1 1B-A400M MoE acquire the source-locked `jam-knowledge-v0.1` domain through a bounded sparse mutation while preserving the registered non-JAM safety set, exact rollback, router behavior and standard Hugging Face materialization parity?

## Frozen inputs

- Base: `ibm-granite/granite-3.1-1b-a400m-base`
- Revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- MoE conversion identity: `dd2b9c750567ff73b1d48e39eb7d1e1213eea9116a68c5164d023420f5a4d670`
- JAM dataset content commit: `5016cb36f8eb5ca715b6fd7796384ae5b607bd12`
- Repaired JAM dataset manifest SHA-256: `d2925ef66c3a7775e5485acea0be40bdd7887e22b89e7b809cb0c07f8102be15`
- JAM concepts: 180
- Gray Paper pin: `0.8.0` / `e5375148597a45a99d31c9aa6bce6c7bf3a48998`
- Formal seeds: `26090711`, `26090712`, `26090713`

The dataset semantic review boundary is closed by `DATASET_SEMANTIC_SPOT_AUDIT.md`.

### Pre-formal manifest integrity repair

The engineering guard discovered that `jam-knowledge-v0.1/manifest.json` recorded an incorrect SHA-256 for `taxonomy.json`. The taxonomy bytes at the original dataset commit and on this branch are identical; their actual SHA-256 is:

`4f3e2da96a2cfa0d0b0496da191e21a50c38779dc804c54c1fc21b5f236cf886`

The stale manifest value was:

`16a993d3eaa66da35c5c3e4bc6b6e8daa1f2dc59a3015d2089edf779199af51b`

Only manifest metadata was corrected. No canonical concept, taxonomy, reasoning-holdout or generated-dataset bytes were changed. Version 1.2 pins the repaired manifest hash directly in `protocol.json`, and the canonical formal runner refuses to start if it differs. The repair and identity pin were completed before any formal GPU seed was opened.

## Mechanism

The writable unit remains the Oracle-validated aligned Granite group: 32 gate/up intermediate rows plus the matching 32 down-projection columns at layer 23.

This experiment does **not** assert that a group is a natural Cell or knowledge atom.

The release-oriented mutation may use a bounded prefix of independently ranked coordinates:

```text
capacity 1 -> capacity 2 -> capacity 4
```

Coordinates must use distinct experts. Each expert therefore exposes at most `32 / 512 = 6.25%` of its intermediate width to the mutation.

Each capacity is trained independently from the same frozen base. The smallest capacity satisfying every gate is selected.

## Real language objective

Unlike Oracle 001 and History Compression 001, the new task is not a one-token synthetic target.

Training uses answer-only causal language-model loss over complete JAM answers:

```text
Question: <question>
Answer:<reference answer><eos>
```

Prompt and padding tokens are masked from the target loss.

The optimization objective is:

```text
L = L_JAM_answer_tokens + 12 * KL(base || mutated)_registered_history
```

Only the selected aligned groups receive parameter updates.

## Evaluation separation

Checkpoint selection may use:

- all 409 JAM training rows;
- the separate 180-row validation set;
- the learner-visible 32-prompt historical calibration set.

Checkpoint selection may **not** use the heldout JAM evaluation families:

- factual: 180
- relational: 66
- misconceptions: 49
- reasoning: 50

The final heldout families are scored only after the safe checkpoint for a capacity has been selected.

## Safety boundary

The history budget deliberately uses the proven History Compression 001 `full_32` positive control rather than the observed minimum of eight prompts. The independent 32-prompt history evaluation set remains learner-invisible.

Formal verification additionally fresh-reloads Granite and checks:

- pinned dataset-manifest identity;
- router top-k identity;
- artifact reapply parity;
- subtraction rollback;
- temporary standard Hugging Face checkpoint materialization parity.

## Formal decision

A capacity passes only if every registered gate passes. A seed passes if any capacity passes, with the smallest passing capacity selected. The experiment is supported only if at least 2 of 3 formal seeds pass.

Possible aggregate statuses:

- `JAM_KNOWLEDGE_MUTATION_SUPPORTED`
- `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED`
- `JAM_KNOWLEDGE_MUTATION_INCOMPLETE`

A supported result is necessary but not sufficient for public model release. After support, one release-candidate mutation must be frozen and the unified Base-vs-RC benchmark plus LoRA/PEFT baselines must still be run.

## Run

Hosted formal execution uses:

`research/notebooks/07-safe-model-evolution/jam-knowledge-mutation-001-kaggle.ipynb`

Each completed seed is published immediately to the same research branch so a hosted-session interruption cannot erase completed formal evidence.

**Do not merge the experiment PR before all intended formal result commits have been published.**
