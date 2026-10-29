# Historical sampling coverage — separate streams, separate evidence

## R3h normal main training (initial implementation)

`r3e.Trainer` seeds an owned NumPy generator with `model.seed + data_seed_offset`. R3h delegates normal updates to it. The bridge delegates batch sampling to original R3 `data.batches`. Each update makes eight vector-length1 `integers(0,23940,1)` calls in the original configuration. Vector grouping matters because PCG64 retains cached32-bit draws. Replacing these calls with a single convenient vector call is not assumed equivalent.

R3h monitoring receives the model, not the Trainer's owned data generator. Step0 is seeded afresh, rather than restoring the parent's training RNG. Resume restores the saved owned state. All three relevant files match historical source01f226e259356bb06a8ef275e8314649062446e4 byte-for-byte; the proof is retained here.

The local CPU replay emits candidate window indices, raw IDs, input/label spans and full states at six boundaries for each seed. No unverified interval is labeled historical. Original Checkpoint comparisons require authenticated artifact access. Initial remote scope is B-H only (three seeds), to address the direct R3m lineage without downloading every historical arm. Other R3h arms remain NOT_AUDITED.

## R3i Proposal and Shadow

`r3e.policy_proposal.collect` peeks batches using a deepcopy of the data RNG and an independent route RNG copy; it must not be counted as consuming the main stream. It still exposes training content to the proposal mechanism and belongs in a separate exposure ledger. R3i paired Shadow continuations compare both branches' full data RNG and cursor, but accepted/abandoned/resumed transaction paths must be identified from real receipts. Do not derive their branch histories merely from R3h's seed replay. Relevant code: r3i/adaptation.py, r3i/trainer.py and the proposal/transaction storage helpers.

## R3j/R3k continuations

`r3j.continuation.advance` restores the full historical Trainer and advances both paired branches, checking data and route stream equality. Start from each case's actual restored data_rng, not a fresh seed or a global-step approximation. The case manifest, selected parent checkpoint, continuation receipts and branch identity are required. R3k's predictor probe/value holdout are separate research-role windows from fixed raw ranges; their asset manifests/membership must be recovered independently. They are not among the four recovered R3 roles.

## R3l branches and research roles

R3l restores each finalist's official anchor, or that branch's own saved state after interruption. Only active physical branches at the same step are advanced, with pair checks; multiple logical roles can share a physical branch. Retained receipts use continuation-relative step values0/10/50/100 while state.step remains absolute300/310/350/400. Do not compare a relative receipt index with `step*512` or replay the same logical branch twice. Search/select/report/path_diagnostic use their own manifests and windows, not R3h's fixed evaluation list.

## NumPy producer identity

The historical Trainer schema saves generator state but no NumPy producer-version field. The available evidence has not identified that version. Extraction records current NumPy only as extraction runtime, never as historical producer. Matching endpoints can be reported as BOUNDARY_CONSISTENT while strict historical certification remains NOT_CERTIFIED until version evidence is recovered/reviewed. This limitation is explicit; no speculative package downgrade/upgrade is performed.

## Global isolation

The earlier50 reproduced encoded-source claims,2 reused007 claims and6 unresolved later-role asset identities retain their separate meanings. Whole-story hashes alone do not certify fragment/cross-story overlap exclusion. The current sampler work does not approve alternative proof routeB. Fragment comparison scope, preprocessing and exact context/label-span rules require the independent review already proposed in WINDOW_PROOF_PATHS_REVIEW.md. No new fragment cutoff is chosen to retain candidates, and no quality scores enter any selection.
