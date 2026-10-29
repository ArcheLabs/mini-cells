# Historical sampling trace audit001 — implemented, original boundaries pending

R3m retains all512 Main candidates and its original frozen configuration. Tokenizer/asset reconstruction remain PASS; scientific registration and global content isolation remain BLOCKED. No GPU, model forward/backward, optimizer update or quality-based selection occurs here.

## Completed locally

1. CI for source4a0f11a5 completed successfully: https://github.com/ArcheLabs/mini-cells/actions/runs/40081479137. This is engineering evidence only.
2. Original R3 batch sampler, R3e Trainer and R3h Trainer files were fetched at historical source01f226e259356bb06a8ef275e8314649062446e4 and each matched the current file byte-for-byte (`ORIGINAL_SAMPLER_SOURCE_CHECK.json`).
3. CPU seed replay reproduced500 sampler steps ×8 sampled windows for each of151001301/302/303. The full local candidate traces contain4,000 windows per seed. Compact boundary states and trace hashes are in `candidate-replay/SAMPLING_REPLAY_REVIEW.json`. Full speculative JSONL traces are local-only and deliberately not another public historical-data archive. They can be regenerated from the immutable algorithm/inputs; no HF durability is claimed.
4. Three synthetic CPU checks compare sampling with the original batch function, full RNG cache state, missing boundaries and mismatch/version blocking. No model is instantiated. Synthetic checkpoint evidence is never called historical evidence.

No original R3h Checkpoint is available locally and the local process has no HF_TOKEN. Thus all recorded candidate intervals remain NOT_CERTIFIED. No actual sampling trace has been certified yet.

## Next authenticated CPU step

`recover_sampling_boundaries.py` reuses the original HubBackend, verified_manifest and Store.verified. It reads the fixed original R3h HF manifest referenced by frozen R3i baseline, selects only B-H for three seeds at0/100/200/300/400/500, verifies receipt identity and complete checkpoint byte/state fingerprints, and extracts data_rng/cursor into small JSON. It never constructs a Trainer/Model or calls update. Trusted project checkpoints are deserialized on CPU using the existing Store validation path. Model definitions may be imported by that existing Store dependency, but no model is instantiated or evaluated.

The original manifest is checked before selecting objects. Each unique needed object uses the existing HF cache; no blanket snapshot download, write preflight, remote commit, training arrays or model-quality analysis is requested. Only RNG evidence/replay reports need be retained in Kaggle Output. Historical checkpoint originals remain unchanged.

The accompanying CPU-only Notebook runs boundary extraction then replay automatically using the existing HF_TOKEN. It records missing checkpoints as failures without inventing a match. Save Version and return `CHECKPOINT_BOUNDARY_EVIDENCE.json` and `SAMPLING_REPLAY_REVIEW.json`. This requires no GPU and does not rerun the previous tokenizer/asset reconstruction or test suite.

## Certification limits

Both endpoint full RNG states (including has_uint32/uinteger), cursors and original checkpoint identities must match. Missing or divergent endpoints block their interval. Source and window ordering cannot be changed to fit. Producer NumPy version evidence is currently absent; the current Kaggle NumPy version is not relabeled historical. State-consistent intervals may be reported as such, but strict certification stays NOT_CERTIFIED until the required version provenance is resolved/reviewed. The small extracted evidence can be reused for that review without downloading checkpoint objects again.

Initial certified scope, if achieved, would be B-H normal R3h main-stream consumption only. Other R3h arms and R3i–R3l Proposal/Shadow/case continuations/research roles remain separately NOT_AUDITED, as detailed in `BRANCH_COVERAGE.md`. Eligible R3 windows and R3h replay do not identify those other exposures.

Historical source claims retain the existing50 encoded-stream recoveries,2 reused007 proofs and6 later-role asset gaps. Fragment/cross-story/cross-tokenizer isolation has not been established. No new fragment threshold or automatic substitute for Window Gate is introduced; proof routeB remains pending independent scientific review.
