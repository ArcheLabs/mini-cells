# R3h B-H sampling boundary consistency — independent review001

## Result

The latest user-supplied evidence is genuinely new and contains18 successfully extracted original checkpoint boundaries, zero extraction failures. The pinned extractor's source identity, fixed HF manifest/reference, original R3h configuration and original sampler identity are verified by the local replay entry point. The Kaggle replay source hash also matches the retained implementation.

Independent CPU replay under NumPy1.24.4 reproduces the submitted Kaggle NumPy2.1.3 output: full PCG64 state (including cached32-bit state), data cursor, selected-window trace hashes and all predicted boundary states match for all three seeds.

| Seed | Matched checkpoint boundaries | Adjacent boundary-consistent intervals | Replayed window selections |
|---|---:|---:|---:|
| 151001301 | 6/6 | 5/5 | 4,000 |
| 151001302 | 6/6 | 5/5 | 4,000 |
| 151001303 | 6/6 | 5/5 | 4,000 |

Boundaries are0,100,200,300,400,500. Cursors are0,51,200,102,400,153,600,204,800,256,000. Every submitted comparison records verified original checkpoint identity and complete-state fingerprint. Original checkpoint bytes remain in the private historical store; the local independent audit checks extracted RNG evidence and replay consistency rather than claiming another local verification of unavailable checkpoint bytes.

## Meaning of NOT_CERTIFIED

All15 intervals have `boundary_consistent: true` and `comparison_performed: true`. Their strict status remains NOT_CERTIFIED solely because historical NumPy producer-version identity has not been established under the frozen audit rule. `historical_numpy_identity_match: false` represents UNKNOWN, not an observed version conflict. No seed, sampler, window ordering or threshold was modified to obtain a match.

The cross-version agreement is useful finite compatibility evidence; it is not a declaration of the historical producer package version. No certificate or protocol exception is granted by this report. Any scientific acceptance of a boundary-consistency alternative must be independently reviewed rather than silently implemented as a gate change.

This evidence is sufficient to preserve the scoped engineering conclusion **all18 original boundaries are consistent with deterministic replay of the frozen R3h B-H normal main stream**. It is not a blanket statement about every historical exposure or every intermediate operation in other experiment branches.

## Scope and remaining work

- Exact original tokenizer and four-role asset reconstruction remain PASS.
- Other R3h arms and R3i Proposal/Shadow, R3j–R3l case continuations and research roles are not certified by this result.
- The earlier50 encoded-source identities,2 reused007 proofs and6 research-role asset gaps retain their original classifications.
- Complete historical source coverage, fragment/cross-story/cross-tokenizer isolation and unknown reserved extents remain unresolved.
- Conservative proof routeB still awaits independent scientific review.
- R3m registration remains BLOCKED. All512 Main candidates, Trigger/Pilot pending status and frozen scientific config remain unchanged.

No further run of this same checkpoint extraction/replay is needed to demonstrate the observed boundary matches. Preserve this evidence and resolve the remaining provenance/isolation/review questions. This independent verification used CPU only: no model loaded, no quality evaluation, no optimizer update and no GPU work. Code CI97243d9a passed (run40085822308); that remains engineering evidence only.
