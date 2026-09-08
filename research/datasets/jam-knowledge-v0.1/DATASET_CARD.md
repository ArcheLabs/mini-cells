# Dataset Card — JAM Knowledge v0.1

## Intended use

Training and evaluating a bounded JAM domain-knowledge mutation for the MiniCells/CLM Safe Model Evolution release candidate.

## Source

Canonical semantics are paraphrased from Gray Paper 0.8.0 at commit `e5375148597a45a99d31c9aa6bce6c7bf3a48998`. No Gray Paper source blobs are vendored into this dataset. `w3f/jamtestvectors` is pinned separately as an auxiliary future conformance source and does not override Gray Paper semantics.

## Unit of curation

The primary unit is a protocol concept, not a question-answer pair. Concepts are sharded by domain and contain canonical facts, relation edges, misconception statements and exact source references. Deterministic training/basic-evaluation questions are generated from this layer.

## Scope

- 180 canonical concepts across 10 JAM domains;
- 66 explicit relation edges;
- 49 misconception records;
- 50 curated, non-template, cross-concept reasoning questions kept learner-invisible.

## Splits

- Training and basic factual/relational/misconception evaluation are deterministically generated under `generated/` and are not canonical source files.
- The 50-row reasoning holdout is canonical, sharded under `evaluation/reasoning/`, and marked `derived=false`.
- The reasoning holdout must not be used for coordinate discovery, training, checkpoint selection or early stopping.

## Known limitations

- English only in v0.1.
- Conceptual rather than exhaustive formal-mathematical coverage.
- Does not replace Gray Paper conformance testing for a JAM client.
- Template-derived evaluation is intentionally easy and must be reported separately from reasoning performance.
- The target Granite model may already contain some historical JAM/Polkadot knowledge from pretraining; release evaluation therefore must report the frozen base model on the exact same JAM heldout.
- Source locking and structural validation are complete, but a semantic spot audit of high-impact concepts should be recorded before formal release-training seeds are opened.

## Versioning

Once the audited commit is used for formal release training, v0.1 should be treated as immutable. A material Gray Paper change should create a new dataset version; a material factual correction should be captured by an explicit erratum or new version rather than silently changing the trained-data identity.
