# M1-002A · Single-fact protocol bridge

The prior capacity run completed on T4, but the live full-context control reached only 8/12. This bridge tests protocol changes before spending GPU time on multi-fact writes.

Use the pinned 1.3B Developer Preview on one T4. Freeze 12 episodes and two wordings per episode. The reference reuses the first 12 M1-001 fixtures. The anchor replaces their project identifiers with the M1-002 identifiers. All subsequent comparisons reuse those project identifiers and pair the same M1-002 target word with a randomly selected alternate word (seed 66110). The combined arm reproduces the original-value input and candidate set of M1-002 exactly.

| Arm | Project | Written code | Write/query wording | Answer options |
| --- | --- | --- | --- | ---: |
| Reference | M1-001 | Random multi-token code | M1-001 | 2 |
| Anchor | M1-002 | Same random multi-token code | M1-001 | 2 |
| Encoding | M1-002 | Single-token word | M1-001 | 2 |
| Candidates | M1-002 | Same word | M1-001 | 20 |
| Wording | M1-002 | Same word | M1-002 | 2 |
| Combined | M1-002 | Same word | M1-002 | 20 |

Query each condition with no memory, with a live full-context cache for each written value, and with a recurrent-only stored Cell for each written value. Each scoring stream is independent. Cell bundle SHA-256 must stay unchanged during read. Compare candidate log probabilities within each wording. A flip passes only if both written values each strictly outrank all distractors in both wordings. Report paired gains and losses for the six arm comparisons.

Run capacity inside the same job only if both full-context and Cell flips reach 80% in each of Reference, Anchor, and Combined, and Combined no-memory recall for the original value stays at or below 10%. With twelve episodes these bounds are at least 10 and at most 1. Intermediate arms diagnose the protocol and do not select a replacement dataset. If the bridge passes, run the frozen 1/3/8/16-fact capacity evaluation. A failed bridge publishes its outcomes and records capacity as skipped. Do not open later experiments.

Bridge dataset canonical SHA-256 is `3c3f307fef01bafc4222d28b6f15d2a0876e0cb9dd369654c954027bcba8a0d7`. The generator also verifies both original dataset hashes. Run `kaggle_bridge.py` as a Kaggle Python Script, with T4, Internet and the `GITHUB_TOKEN` secret.
