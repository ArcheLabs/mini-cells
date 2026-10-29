# Repository-wide data usage registry — proposed design

Status: DESIGN_ONLY. No historical registration, dataset identity, seed, model or result is rewritten. This proposal is based on the existing GLOBAL_DATASET_USAGE_LEDGER.json and both failed R3m amendments.

## Purpose

A range that lies beyond an old training prefix can still contain the same stories. A global integer interval also has no meaning without the dataset Revision, original split, shard order and raw-vs-filtered indexing convention. Registration must check indices, content and model-visible windows separately.

## Proposed location and consumers

`research/governance/data-usage/` contains small manifests/schema/review decisions. A future shared pure-CPU component under `shared/python/minicells_data_registry/` supplies the same audit functions to every experiment. Do not migrate experiment-specific algorithms into it.

Large immutable hash indices belong in the existing artifact/HF transport, referenced by fixed Revision, SHA-256, size and manifest identity. GitHub receives compact proofs and pointers, not raw text, tokenizer training data or huge checkpoint/index files. Build each verified hash index once and reuse it; do not repeat global scans or test suites on Kaggle GPUs.

## Entities

1. **Dataset snapshot**: repository, immutable Revision, original split, ordered files with LFS/file hashes, per-file raw row counts, global offsets, loader/order definition and footer verification level. Metadata-only verification must not be called full-file hashing.
2. **Immutable usage claim**: experiment ID; original source/config identities; role; snapshot ID; half-open raw-global ranges OR an explicit raw-index membership file; original selection/normalization/empty-row rules; source evidence path/SHA; state (proposed/used/reserved/unknown). `train_stories_consumed` is a filtered count, never silently a raw row endpoint.
3. **Content index**: raw UTF-8 SHA-256; versioned NFKC/casefold/whitespace hash; separate word-sequence signature. Exact/format-equivalent collisions block release. Word-only collisions are conservative review flags, not proof of semantic equivalence. Optional near-duplicate rules require an independently frozen algorithm and threshold.
4. **Window claim**: exact original tokenizer hash, preprocessor identity, EOS policy, token sequence length including labels, stride, story/cross-story boundaries, source raw IDs, token offsets, input/target hashes, context span and selection rule. Corpus-eligible windows and windows actually sampled by an optimizer are different entities. Never claim observed-window equivalence without saved starts or reproducible sampler state.
5. **Audit proof/review**: immutable request ID, candidates, all input-index roots/hashes, collision sources, coverage gaps, isolated auditor version/source, access log, pass/fail per gate, reviewer decision and resulting identity. Main/Confirmation text is never handed to a model by the auditor.

## Example usage claim

```json
{
  "claim_id": "sha256-of-canonical-claim",
  "experiment_id": "historical-original-id",
  "role": "train",
  "snapshot_id": "sha256-of-dataset-layout",
  "index_namespace": "RAW_GLOBAL_ZERO_BASED",
  "raw_ranges": [[0, 476901]],
  "membership_ref": "immutable-file-containing-exact-nonempty-raw-ids",
  "filtered_records": 476847,
  "original_identity": {"source_sha": "preserved", "config_sha256": "preserved-or-unrecorded"},
  "evidence": {"path": "original/corpus-manifest.json", "sha256": "preserved"},
  "import_status": "REPRODUCED_FROM_ORIGINAL_TOKEN_STREAM",
  "historical_identity_rewritten": false
}
```

## Release state machine

`PROPOSED → SNAPSHOT_VERIFIED → RANGE_AUDITED → CONTENT_AUDITED → WINDOW_AUDITED → REVIEWED → REGISTERED`.

Every transition appends evidence. A collision or unknown required input yields BLOCKED; rejected candidates remain in history. No silent exclusion, range relocation, tokenizer change or identity replacement. A clean range check is never sufficient for registration. Only after all required gates and approval may the new Source/Config identity be minted and CPU engineering gates run. GPU persistence/recovery remains a separate required release condition.

Reservation markers such as `reserved_boundary=60000` lacking an upper endpoint are retained as UNKNOWN_SCOPE until an owner resolves them. Do not invent an infinite sealed tail or silently declare the remainder free. Read-only integrity auditors may compare opaque hashes; models must not receive Confirmation or unreleased Main Report inputs.

## Migration plan

1. Import the current 111 evidence documents/181 interval claims as immutable historical claims. Retain source hashes and unresolved identities; do not label imported guesses verified.
2. Reconcile filtered counts using original loader behavior and preserved token-stream hashes (007 is the example). Record raw membership rather than changing the old manifest.
3. Build shared hash indices for each snapshot/split and cross-snapshot content comparisons. Keep historical role information even when the same content was used in multiple experiments.
4. Add reusable window manifests and audit-only access separation. If historical observed starts are absent, report that limitation and block claims requiring them.
5. Adopt the registry for new experiments first; historical experiments stay unchanged. CI checks claims/proof integrity and missing evidence, never launches a GPU experiment or auto-approves a scientific release.

## Acceptance tests for the future shared component

- Cross-shard mapping, raw/filtered index separation, original split distinctions and out-of-bounds rejection.
- Exact duplicates at different indices; NFKC/format variants; cross-role and cross-Revision collisions.
- Input-plus-target and cross-story window overlaps, tokenizer mismatch, missing observed starts.
- Immutable failed requests; atomic concurrent reservations; stale index/input hash rejection.
- Sealed-role auditor/model separation; metadata-only publication; no historical identity edits.

R3m amendment002 remains rejected regardless of this future mechanism: its Main candidate contains150 exact historical training stories. No replacement range is selected here.
