# Development pilot deterministic story deduplication

The initial pilot selected raw TinyStories row ranges and stopped on exact duplicate
content. No training started in the reported failed run.

The development pilot now selects the first occurrence of each exact UTF-8 story
hash, in source order, before assigning unique story ordinals to training and
development ranges. It retains the configured 50,000 training and 1,024 development
story counts. It never scans raw rows at or beyond the reserved confirmation start
(60,000); insufficient unique content before that boundary fails explicitly.

The asset manifest and pilot provenance record the policy version, scanned raw-row
count, removed duplicate count, unique count, scan boundary, and selected source row
indices. The tokenizer still trains on training stories only. The existing strict
encoder and cross-split content checks remain enabled. The formal raw-row asset
policy and confirmation preparation are unchanged.

The updated Notebook uses new code and asset directories ending in `dedup-v1`.
Prior partial assets are preserved; no automatic deletion or overwrite occurs.
The pilot stays unregistered development only, with no formal scientific verdict.

Validation: 78 local CPU tests passed, including deterministic split selection,
cross-split duplicates, reserved-row non-access, full mocked asset preparation,
strict duplicate rejection, and historical reused Experiment 005 tests. No GPU
training or real corpus download was performed for this repair. GitHub Actions
validates the pushed commit separately.
