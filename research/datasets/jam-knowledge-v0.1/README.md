# JAM Knowledge v0.1

`jam-knowledge-v0.1` is the source-locked JAM knowledge registry intended for the first real CLM domain-mutation release experiment.

Its purpose is not to mirror or republish the Gray Paper. It converts one frozen Gray Paper revision into a bounded set of paraphrased protocol concepts, explicit relations, misconception corrections, and learner-invisible reasoning questions that can be used to build JAM training data and evaluate a JAM mutation.

## Status

`CURATION_COMPLETE_SOURCE_LOCKED`

The source pin and dataset structure are locked and mechanically validated. Before opening formal release-training seeds, perform and record a semantic spot audit of the highest-impact concepts (especially Refine/Accumulate, authorization, guaranteeing/assurance, availability and PVM host-call semantics).

## Source lock

Canonical semantic authority:

- repository: `gavofyork/graypaper`
- Gray Paper version: `0.8.0`
- commit: `e5375148597a45a99d31c9aa6bce6c7bf3a48998`
- source pin: `graypaper-0.8.0-e5375148`

Auxiliary conformance source:

- repository: `w3f/jamtestvectors`
- commit: `1dc503af37ae8ecc7b6e24a393f6102801c0c80c`

See `sources.lock.json`. Canonical facts are grounded in the Gray Paper pin; test vectors are auxiliary and do not override Gray Paper semantics.

## Canonical assets

```text
jam-knowledge-v0.1/
├── README.md
├── DATASET_CARD.md
├── sources.lock.json
├── schema.json
├── manifest.json
├── taxonomy.json
├── concepts/
│   ├── foundations.jsonl
│   ├── block_state.jsonl
│   ├── consensus.jsonl
│   ├── services.jsonl
│   ├── authorization.jsonl
│   ├── work.jsonl
│   ├── guarantees.jsonl
│   ├── accumulation.jsonl
│   ├── pvm.jsonl
│   └── serialization.jsonl
└── evaluation/
    └── reasoning/
        ├── part-01.jsonl
        ├── part-02.jsonl
        ├── part-03.jsonl
        ├── part-04.jsonl
        └── part-05.jsonl
```

The canonical registry is sharded by knowledge domain for reviewability and future Gray Paper diffs. The build script emits an aggregate `generated/concepts.jsonl` for training pipelines.

## Coverage

| Category | Concepts |
|---|---:|
| foundations | 15 |
| block_state | 15 |
| consensus | 15 |
| services | 22 |
| authorization | 10 |
| work | 25 |
| guarantees | 20 |
| accumulation | 18 |
| pvm | 25 |
| serialization | 15 |
| **Total** | **180** |

The registry also contains 66 explicit concept-relation edges and 49 registered misconception statements. The canonical reasoning holdout contains 50 cross-concept questions marked `derived=false`.

## Concept row

Each JSONL row has a stable concept ID, one compact canonical fact, optional relation edges and misconceptions, exact upstream source paths, source pin, tags, difficulty and canonical status.

Example:

```json
{
  "id": "jam.services.refine",
  "title": "Refine entry point",
  "category": "services",
  "canonical_fact": "...",
  "relations": [
    {
      "target": "jam.services.accumulate",
      "statement": "Refine is separated from the stateful Accumulate entry point."
    }
  ],
  "misconceptions": [
    "Refine directly mutates the global JAM state."
  ],
  "source_refs": ["text/accounts.tex"],
  "source_pin": "graypaper-0.8.0-e5375148",
  "tags": [],
  "difficulty": 1,
  "status": "canonical"
}
```

`source_refs` are paths inside the frozen upstream Gray Paper revision, not copied source text.

## Build derived QA

Run from repository root:

```bash
python scripts/research/jam_knowledge_v0_1/build_dataset.py
```

This deterministically materializes:

```text
research/datasets/jam-knowledge-v0.1/generated/
├── concepts.jsonl
├── train.jsonl
├── validation.jsonl
└── evaluation/
    ├── factual.jsonl
    ├── relational.jsonl
    ├── misconceptions.jsonl
    └── reasoning.jsonl
```

`generated/` is intentionally ignored by Git. With the current canonical registry the minimum deterministic view contains:

- 409 training examples;
- 180 validation examples;
- 180 factual evaluation examples;
- 66 relational evaluation examples;
- 49 misconception evaluation examples;
- 50 non-template reasoning examples.

The 409-row training view is a reproducible minimum, not a target upper bound for `JAM Knowledge Mutation 001`. Additional paraphrases or instruction styles may be generated later, but they must preserve the canonical fact layer and must never use the reasoning holdout.

Template-derived factual performance must not be represented as evidence of generalization. The 50-row reasoning holdout is the primary v0.1 cross-concept test.

## Validate

```bash
python scripts/research/jam_knowledge_v0_1/validate_dataset.py
```

The validator checks source pinning, exact category counts, unique IDs, relation targets, source-reference shape, canonical file hashes, reasoning isolation, split IDs, concept references and exact train/evaluation question duplication when generated views exist.

## Training policy

For `JAM Knowledge Mutation 001`:

1. record the exact audited dataset commit before formal training;
2. materialize generated training data only from that commit;
3. never expose canonical `evaluation/reasoning/**` to coordinate selection, training, checkpoint selection or early stopping;
4. keep base-model preservation/calibration prompts outside this JAM dataset so JAM acquisition and historical-safety supervision remain independently measurable;
5. compare the final release candidate with the frozen base model on the exact same JAM heldout and general-capability evaluation before public release.

## What v0.1 does not claim

This dataset does not cover every equation, constant, host-call edge case or implementation detail in Gray Paper 0.8.0. It is deliberately bounded conceptual coverage for the first ~1B MoE release experiment. It does not establish that a model learned JAM merely because template-derived factual accuracy is high, and it does not replace JAM client conformance testing.

A material Gray Paper change should create `jam-knowledge-v0.2` rather than silently mutating v0.1. Concept additions, edits and removals should be reviewable as a versioned knowledge diff.
