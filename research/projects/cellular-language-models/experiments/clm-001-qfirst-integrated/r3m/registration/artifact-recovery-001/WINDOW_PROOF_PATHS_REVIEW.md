# Window Gate paths A/B — proposal for independent review only

Status: NOT_APPROVED. This document does not amend the scientific protocol or satisfy the current Window Gate. Amendment003 and its 512 candidate indices remain unchanged. No new stories are selected.

## Path A: observed historical windows

Recover exact original tokenizer bytes, immutable asset manifest, preprocessing/EOS rules, story membership/offsets and window metadata. Distinguish eligible starts, fixed evaluation starts, and starts actually consumed by training. Recover observed starts or reproduce them using the original sampler implementation, complete saved RNG/cursor state, worker ordering and any retry/skip rules. An evaluation-start list or final data cursor alone is insufficient. Include input, labels, context and cross-story boundaries in comparison. Bind every result to the original source/config/artifact hashes. Missing inputs keep this route BLOCKED.

## Path B: conservative source/window superset

Proposed sufficiency argument: if every model-visible historical window lies within a verified exclusion universe U, and no new evaluation window overlaps any relevant member of U under the approved leakage definition, absence of actual sampled starts need not prevent isolation certification. This is a containment argument, not proof from unequal whole-story hashes.

Necessary evidence, before this path can be accepted:

1. Enumerate all relevant historical training, tokenizer-training, calibration, development, validation and reserved Confirmation claims. Define scope explicitly. No unresolved Revision, loader mapping, reservation endpoint or missing source can be treated as absent. Provide separate decisions for unknown claims, using evidence and conservative supersets where justified.
2. Verify the entire source corpus/eligible token stream that contains every observed window, including terminal partial stories and all preprocessing variants. Encoded-stream equality is useful but does not prove raw text equality, nor does it preserve the original unrecorded Revision retroactively.
3. Whole-story exact/NFKC/word hashes are only the first exclusion layer. Check partial-story copying and actual input-plus-target spans, not just complete story equality. Freeze any minimum span, normalization and near-duplicate rule through scientific review before applying it; do not tune rules to retain the current candidates.
4. Include old concatenated-stream windows crossing story boundaries and their context. Candidate story isolation does not make historical cross-story windows disappear. All-eligible-window enumeration can replace observed starts only after corpus containment and boundary semantics are verified.
5. Token IDs from different tokenizers cannot be compared directly. Recover each relevant original tokenizer/preprocessor or a proven equivalent representation; compare with correctly bound text/token-span maps and include special tokens/labels. The 2048-vocabulary tokenizer used to reproduce old streams is not a replacement for the required R3m 4096-vocabulary tokenizer.
6. Use an isolated integrity auditor, never a model evaluator. Reserved Confirmation and Main remain inaccessible to models. The auditor emits hashes and overlap/provenance metadata, not quality scores.
7. Independently review the containment proof, leakage definition, coverage and any residual limitations. Only an approved, versioned protocol amendment may enable Path B. Engineering code must not silently turn a proposed B pass into the existing Gate A pass.

## Current disposition

Neither path passes today. Historical eligible/evaluation metadata may be recoverable from the pinned HF bundle; it does not by itself restore actual optimizer starts. Source stream replay resolves a substantial subset of model-visible byte identities but does not establish complete source/fragment/window isolation. Keep both routes marked BLOCKED_PENDING_REVIEW_AND_EVIDENCE.
