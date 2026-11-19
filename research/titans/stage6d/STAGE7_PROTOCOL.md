# Stage 7 protocol proposal

The Stage 6C baseline manifest matches the published result byte for byte.
Durable checkpoint retrieval and independent restore remain pending.

## Invariants

Use one target full-attention layer (index 3) and one logical Cell. Its
two KV shards are independent. Query heads 0–3 address shard 0 and heads
4–7 address shard 1. Within each shard, writes and reads use the same
projection weights, normalization and pre-RoPE position policy. Native
Qwen attention retains its own position encoding. Pooling and packet
templates must be identical for the exact-address control and pinned for
the paraphrase test. Each packet starts with empty KV, DeltaNet recurrent
and convolution state.

Contradictory writes target the same canonical address. The desired
behavior is latest-value retrieval after an explicit write. Report
new-value recall and stale-value errors. Treat replacement success as
empirical because a Titans update provides no exact overwrite guarantee.

## Sequential gates

1. **Addressing.** Write unique associations and query identical addresses.
   Compare correct, empty and permuted donor Cells using the same candidate
   order and matched write counts for correct and donor.
2. **Generalization.** Use the same comparisons on held-out paraphrases.
   Measure old and new values around contradictory updates and unrelated
   writes. Stop if addressing fails.
3. **Persistence.** After distractor writes, export, terminate the source,
   import in a clean process and repeat queries with empty session state.
   Compare ranks, scores and Cell hashes before and after. Reject mismatched
   identities and damaged artifacts. Stop if generalization fails.
4. **Integration.** Compare language generation after all three gates pass.

## Proposed freeze values for review

Use 64 independent episodes with eight facts and eight candidates per
episode. Pair all controls by episode. Use disjoint development and test
entity/value pools and a seeded derangement for donor values. Store
generator revision, seed, pool and template hashes before test access.
Report Recall@1, false recall on unknown queries, and episode-level paired
bootstrap intervals with 10,000 resamples. Propose Recall@1 at least 90%
for exact addressing and 80% for paraphrases, false recall at most 5%,
and a gain of at least 10 percentage points over each control with the
paired lower confidence bound above zero. Choose any rejection cutoff
using development data only. Persistence requires identical rankings,
unchanged read-only hashes and atol 1e-5 / rtol 1e-4 on scores.

Use a separate development pilot capped at 15 GPU minutes to measure
throughput and variance. Propose a 60-minute formal T4 budget, including
startup and publication. Freeze thresholds independently of pilot
outcomes. If the formal workload does not fit the measured budget,
revisit the execution plan before opening Test.

Formal execution requires a frozen data/config manifest and a verified
durable archive of the Update 64 checkpoint. Recover the saved bytes by
published SHA256, archive them outside expiring Output, download and hash
them again, then independently load and reproduce the frozen validation
fixture. Record the measured runtime and artifact receipt.
