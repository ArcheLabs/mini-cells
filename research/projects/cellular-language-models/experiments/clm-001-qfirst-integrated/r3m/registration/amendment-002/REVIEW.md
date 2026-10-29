# Amendment 002 — [500000,501024) content audit

**Decision: REGISTRATION_BLOCKED_TEXT_LEAKAGE. Not registered. No replacement selected.**

| Candidate | Raw records present | Exact historical duplicates | Normalized historical duplicates |
|---|---:|---:|---:|
| Trigger [500000,500256) | 256 | 0 | 0 |
| Pilot [500256,500512) | 256 | 0 | 0 |
| Main [500512,501024) | 512 | 150 | 150 |

All candidates have raw UTF-8, NFKC/casefold/whitespace and NFKC word-sequence hashes. No within-candidate duplicates were found by these rules. The150 affected Main records are29.3% of that role. These are actual identical texts at different raw indices, not just a range conflict. Example: candidate500872 equals historical train83105; their raw SHA256 is `9b7da4ce7a0573175b0a76b93ebcb2f74c3f7f37db866a6d2126d915f7ea2b12`.

## Source and index identity

Unchanged source: `roneneldan/TinyStories`, original `train`, Revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Frozen shard order/row counts are529930,529930,529930,529929 (total2119719). Global ranges are[0,529930),[529930,1059860),[1059860,1589790),[1589790,2119719). All1024 current candidates are in shard0, with local row equal to global row. The mapper also checks cross-shard boundaries and invalid indices.

The full first train shard and validation shard match the frozen repository LFS SHA256 values. Other train shards were inspected via pinned HTTP-range Parquet footers for row counts; they were NOT downloaded and full-file hashed. Verification levels are explicit in SOURCE_LAYOUT.json. No candidate crosses into those shards.

## Historical comparisons

Every source document referenced by the existing global usage ledger was rehashed (111 documents,181 interval claims). All declared interval extents are covered by the scanned train/validation prefixes.58 claims lack a recorded Revision; they are retained as a coverage limitation and not silently certified equivalent to this snapshot.

Experiment007 is independently tied to this Revision by the prior exact120000000-token train stream reproduction. This audit compares actual text hashes for ALL476847 consumed nonempty stories through raw index476900, not merely the old476847 filtered-count interval. It also reproduces the original1000000-token validation stream SHA256 `4a8c15e6a53657488c78209e37fdd98e67e4e159875946cce1fc8663221f8d8f` and compares all4396 historical validation stories. No validation text collision was detected. Historical hash-index roots and individual collision witnesses are in the proof; raw texts are not published.

## Window gate and access boundary

The exact R3m tokenizer artifact (required SHA256 `612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`) was not available locally. It was NOT rebuilt or replaced. No claim is made that exact R3m token-window IDs were verified. Historical actual optimizer start logs are also unavailable; eligible training-corpus windows are not equated with windows actually sampled by the optimizer.

Complete-story historical leakage already fails the content gate. Therefore no Main Report evaluation assets were built and the window gate is explicitly NOT PASSED. This is not an all-clear certificate for Trigger or Pilot, despite zero text collisions there.

The auditor imports no model, performs zero optimizer/quality-evaluation steps and persists hashes/metadata only. Main is used solely for identity/leakage checks; no model receives Confirmation. Config and lock hashes before/after are identical. Original proposals and previous failed proofs remain unchanged.

## Reproduce

Use `verify_shard_layout.py` for pinned shard metadata, then `audit_text_boundaries.py` with the verified train/validation files and this directory's SOURCE_LAYOUT.json/SOURCE_FILES.json. Existing matching proofs are hash-verified and reused; changed inputs/code require a new request directory instead of overwriting this evidence. The small `audit_checks.py` suite is CPU-only and does not scan data or train models.

The repository-level registry design is in `../GLOBAL_DATA_USAGE_REGISTRY_PLAN.md`. It is a prospective design, not a retrospective rewrite of any experiment identity.
