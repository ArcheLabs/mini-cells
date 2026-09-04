# JAM Knowledge Mutation 001 — Frozen Protocol

The machine-readable authority is `protocol.json`.

Protocol version: **1.2**.

All protocol amendments below were completed before any formal GPU seed was opened:

- **v1.1** makes answer boundaries explicit and evaluates artifact/materialization parity above the measured frozen-base forward-repeatability floor, matching the numerical treatment already used for rollback.
- **v1.2** repairs the stale `taxonomy.json` hash metadata without changing canonical dataset bytes, pins the repaired JAM dataset manifest SHA-256, and requires the formal runner to verify that exact identity before execution.

## Registered claim

The experiment asks whether a bounded, sparse, rollbackable mutation can add the frozen JAM Knowledge v0.1 domain to the exact Granite MoE substrate while preserving a registered non-JAM safety set.

It does not test autonomous Cell discovery, zero-history learning, arbitrary editing, multi-mutation composition, or superiority to parameter-efficient fine-tuning.

## Dataset

The canonical JAM content originates from repository commit:

`5016cb36f8eb5ca715b6fd7796384ae5b607bd12`

The corrected, source-locked dataset manifest is frozen by SHA-256:

`d2925ef66c3a7775e5485acea0be40bdd7887e22b89e7b809cb0c07f8102be15`

The manifest correction changed only the recorded hash for `taxonomy.json`:

- stale recorded value: `16a993d3eaa66da35c5c3e4bc6b6e8daa1f2dc59a3015d2089edf779199af51b`
- actual taxonomy SHA-256: `4f3e2da96a2cfa0d0b0496da191e21a50c38779dc804c54c1fc21b5f236cf886`
- canonical taxonomy bytes at the original dataset commit are identical to the current bytes.

Formal execution must satisfy both layers of dataset identity:

1. exact repaired manifest SHA-256;
2. every canonical file must match the SHA-256 recorded inside that manifest.

Expected generated counts are:

| split | rows |
|---|---:|
| train | 409 |
| validation | 180 |
| factual | 180 |
| relational | 66 |
| misconceptions | 49 |
| reasoning | 50 |

The 64 rows used for coordinate ranking are the first 64 training rows after ascending sort by `sha256(id)`. They remain ordinary training rows; the four final heldout families are never learner-visible.

## Writable geometry

- layer: 23
- expert intermediate width: 512
- group width: 32
- groups per expert: 16
- candidate capacity ladder: 1, 2, 4 groups
- at most one selected group per expert
- router frozen
- all non-selected parameters frozen

Candidate ranking uses answer-token JAM gradient energy, historical top-1 gradient importance and expert routing specificity. Four coordinates are ranked once on the frozen base. Capacity 1, 2 and 4 train independent prefixes from an identical base state.

## Sequence loss

For each row, the prompt is:

```text
Question: {question}
Answer:
```

Prompt and answer are tokenized without automatic special tokens. If the tokenizer defines BOS, one BOS token is prepended. The canonical reference answer follows immediately and exactly one EOS token is appended. Only answer/EOS tokens carry labels. Sequences are capped at 192 tokens.

## Historical safety

The experiment imports the exact 32 learner-visible and 32 withheld history prompts from History Compression 001. The withheld set is never used for selection, optimization or checkpoint choice.

The learner objective is:

`JAM answer-token CE + 12 * frozen-base history KL`.

The minimum eight-prompt HC001 result is deliberately not used.

## Training

- FP32
- batch size 4
- history batch size 4
- manual masked SGD
- learning rate 0.01
- selected-gradient norm cap 1.0
- 96 steps per capacity
- candidate evaluation every 12 steps
- candidate must satisfy learner-visible history KL <= 0.05
- among safe candidates, maximize validation reference-NLL gain

## Heldout gates

The final safe checkpoint must achieve at least:

- validation reference-NLL gain: 0.25
- factual gain: 0.25
- relational gain: 0.15
- misconception gain: 0.25
- reasoning gain: 0.10
- token-weighted overall heldout gain: 0.20

Safety and artifact gates require:

- withheld history mean KL <= 0.05
- withheld history top-1 identity >= 31/32
- router top-k identity = 1.0
- <= 4 coordinates
- <= 6.25% writable intermediate width in any selected expert
- distinct selected experts
- nonzero delta
- exact parameter rollback
- forward rollback excess over measured base repeatability <= 1e-5
- fresh-base artifact reapply excess over measured base repeatability <= 1e-5
- temporary standard-HF materialization excess over measured base repeatability <= 1e-5
- exact registered conversion and dataset identities

Raw parity errors and the measured base repeatability are both retained in the result artifacts; the gate is applied only to the excess above that numerical floor.

## Decision

Formal seeds are:

- 26090711
- 26090712
- 26090713

For a seed, the smallest capacity satisfying all gates is selected. At least two formal seeds must pass for `JAM_KNOWLEDGE_MUTATION_SUPPORTED`.

Formal failures are durable results and must be published rather than rerun with altered gates. Any post-observation protocol amendment requires a new protocol version and fresh formal seeds.
