# JAM Knowledge Mutation 001 Failure Diagnostic

Status: **ENGINEERING IN PROGRESS — POST-HOC GPU DIAGNOSTIC PENDING**

This diagnostic does not rerun or modify JAM Knowledge Mutation 001. The frozen upstream decision remains:

`JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED`

## Question

All three formal seeds failed the same registered gate: misconception reference-answer NLL gain remained below the frozen `0.25` threshold, while factual, relational, reasoning, overall heldout, history-safety, router, rollback, and artifact-reapply gates were otherwise stable.

The source dataset uses different answer formulations for misconception training and heldout evaluation:

- training answer: `No. <canonical fact>`
- heldout answer: `The claim is incorrect. <canonical fact>`

Because the formal metric is token-level reference-answer NLL, the post-hoc question is:

> Did the canonical JAM content itself improve enough, with the full formal answer missing the gate mainly because of answer-prefix/formulation dilution, or was canonical-content gain also below the original threshold?

## Frozen upstream evidence

- Base model: `ibm-granite/granite-3.1-1b-a400m-base`
- Base revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- JAM001 protocol SHA-256: `e934be45009d9025adf3b48ee2551f55a7099281196265b478e503d746559a54`
- JAM dataset manifest SHA-256: `d2925ef66c3a7775e5485acea0be40bdd7887e22b89e7b809cb0c07f8102be15`
- Formal seeds: `26090711`, `26090712`, `26090713`
- Capacities: `1`, `2`, `4`
- Frozen upstream decision: `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED`
- Main commit containing all formal artifacts: `762873b525d230fb36acc472b8994bbf7b53525a`

The diagnostic branch is required to leave the upstream protocol, canonical dataset, and formal artifacts byte-identical to that merge commit.

## Method

No training occurs.

For each of the nine frozen mutation artifacts (`3 seeds × 3 capacities`):

1. Load the frozen Granite base.
2. Reconstruct the exact 49 heldout misconception rows.
3. Tokenize each original formal answer exactly once.
4. Use fast-tokenizer character offsets to partition the original supervised answer tokens into:
   - answer prefix;
   - canonical JAM content;
   - EOS.
5. Apply the already-published mutation artifact.
6. Re-evaluate the same rows.
7. Restore the exact original writable slices before the next artifact.
8. Reproduce the original full misconception NLL within a small engineering tolerance.
9. Decompose the full NLL gain into exact token-weighted segment contributions.

A paired counterfactual keeps the heldout question fixed but changes only the answer reference to the training-style prefix:

`No. <canonical fact>`

This is diagnostic only and is not a replacement benchmark.

## Interpretation

No new post-hoc PASS/FAIL gate is introduced.

The original `0.25` misconception threshold is used only as a reference line. At capacity 4:

- if all three `content + EOS` gains are at or above `0.25` while all three original full-answer gains remain below it, the formal answer prefix alone is sufficient to explain the registered failure;
- otherwise, if all three canonical-content gains are at or above `0.25`, non-content tokens (prefix and/or EOS) dilute otherwise above-threshold canonical-content gain;
- if all three canonical-content gains remain below `0.25`, answer formulation cannot by itself explain the failure and canonical-content acquisition is also below the original gate;
- otherwise the result is mixed.

The classification is a diagnostic description only. It cannot change the formal JAM001 decision.

## Outputs

Hosted GPU execution writes:

```text
results/jam-knowledge-mutation-001-failure-diagnostic/
├── diagnostic.json
└── per_row.jsonl
```

After publication:

```text
artifacts/experiments/jam-knowledge-mutation-001-failure-diagnostic/
├── README.md
├── diagnostic_plan.json
├── diagnostic.json
└── per_row.jsonl
```

`per_row.jsonl` contains 441 records (`3 × 3 × 49`) so later analysis can determine whether the ceiling is broad or concentrated in particular misconception concepts.

## Run

Use:

`research/notebooks/07-safe-model-evolution/jam-knowledge-mutation-001-failure-diagnostic-kaggle.ipynb`

The notebook performs source/provenance validation before loading the model, runs only forward evaluation, and publishes the completed diagnostic back to its dedicated branch.
