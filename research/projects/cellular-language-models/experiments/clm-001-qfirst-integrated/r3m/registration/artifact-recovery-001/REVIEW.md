# Artifact Recovery & Historical Coverage Closure 001

No Amendment004 data selection. Amendment003's 512 candidates, all earlier failed proofs, the frozen R3m config, historical models/checkpoints and Confirmation remain unchanged. Registration and GPU training remain BLOCKED.

## P0 — original artifact recovery

The local environment has no HF_TOKEN. The earlier anonymous HTTP401 is an authorization result, not evidence of artifact loss. `recover_original_artifacts.py` reuses the project's existing `HubBackend.get` exclusively, against the original fixed repository/revision/bundle in frozen R3h configuration. It checks the bundle, identity, manifest, exact tokenizer SHA256 and window object hashes. It downloads only three metadata artifacts: manifest.json, tokenizer.json and windows.json. No token arrays, model checkpoints, training/evaluation or remote writes occur.

The CPU-only recovery Notebook uses the existing Kaggle `HF_TOKEN` label. Run All once in a new CPU session of the original Notebook and retain Output. No credentials should be pasted. A successful recovery yields `RECOVERY_PROOF.json` and exact original bytes for subsequent isolated auditing; it deliberately does not pass the Window Gate. The historical window file is classified as eligible/evaluation metadata, not proof of actual optimizer starts.

## P1 — 58 original unpinned claims

Authoritative replay report: `HISTORICAL_CLAIM_CLASSIFICATION_INT64.json`.

- 50 claims: original tokenizer hash, story counts and complete model-visible token-stream hashes reproduced against the frozen snapshot. Five unique prefixes were replayed once per split. These are recovered **encoded source identities**, not rewritten original Revision fields or raw-text identity assertions.
- 2 claims: 007's exact historical 120M/1M stream reproduction reused from immutable prior proof. No unnecessary repeat of the full replay.
- 6 claims: R3k/R3l raw ranges have fixed Revision constants and source-hash checks in their original loader. Their intended sources fall within the conservative 007 prefix, but actual asset manifests/window membership remain unrecovered. `SOURCE_CHAIN_CLASSIFICATION.json` records code hashes and remaining requirements. This conditional superset coverage is not an identity PASS.
- No claim has been demonstrated to use a different Revision by this audit; this is not evidence that no such claim exists. Unknown reservation upper endpoints and remaining source identities still block global release.

The engineering diagnostic under `engineering/UINT16_ENCODING_DIAGNOSTIC.json` records why the uint16 storage representation used by 007 is inappropriate for matching earlier manifests' torch.long hashes. The authoritative small-stream replay uses the historical **little-endian int64** representation, as defined by `language_data._tensor_sha256` and `language_30m._legacy_int64_sha256`. It is not a tokenizer or data change. The intermediate diagnostic is not a scientific isolation proof.

No raw-story count was substituted for a filtered story count. Reports record actual raw endpoints, including skipped empty stories. Original declarations stay unchanged. Encoded-stream recovery narrows the unknowns; it does not silently certify source coverage, actual optimizer windows or semantic independence.

## P2 — Window Gate review

`WINDOW_PROOF_PATHS_REVIEW.md` specifies the pending review of A (observed windows) and B (conservative source/window superset). Whole-story hashes alone cannot detect copied fragments or cross-story windows. Cross-tokenizer numeric IDs cannot be compared directly. No B exception has been approved or enabled in code.

## Engineering checks

Two synthetic fake-store tests cover pinned object hashes, metadata-only read requests, failed-bundle early stop and immutable output. No repeated model suite or GPU diagnostics are added. Existing CPU CI includes these tests. Real authenticated HF recovery remains pending; do not describe the fake-store test as successful private-HF access.
