# M1-002B · Reference-Protocol Capacity

M1-002A reproduced the M1-001 single-fact result on a Tesla T4 using random multi-token codes, the original write sentence and both original queries. M1-002B measures capacity using that fixed two-candidate protocol on the same MoM 1.3B FP16 Developer Preview.

| Stage | Facts per Cell | Independent groups | Targets tested |
| --- | ---: | ---: | ---: |
| 0 | 1 | 12 | 12 |
| 1 | 3 | 12 | 36 |
| 2 | 8 | 12 | 96 |
| 3 | 16 | 12 | 192 |

Stage 0 uses the first 12 frozen M1-001 cases. Later stages use independent datasets from the M1-001 generator with seeds 66203, 66208 and 66216. Write order is shuffled with seed 66200 before selecting targets. Every position in every group is tested. The canonical dataset SHA-256 is `6b2560de959dbf71fffaf955c96e582b23d3beda87fd4b6979bfe43fe58d05e7`.

Each target gets an independently created original Cell and an independently created swapped Cell. Each is populated with one SDK write using 64-Token forward chunks. The swap replaces exactly one fact at the same position with a token-count-matched alternate value. The script validates that all other write tokens and the total length are identical. There is no prior Cell state or append across trials. For full-context comparison, clone a complete live cache for each candidate round. Stage 0 also checks exact score agreement against queries that continue directly after writing, for both written values and both query forms. Candidate scoring uses M1-001 teacher-forced summed log probability, reverse evaluation order and independent caches. Verify unchanged persisted Cell hashes after scoring.

A target passes when the written candidate is strictly higher-scoring with both wordings and for both values. A group passes only when every fact passes. Full-context and persisted Cell each require at least 10 of 12 groups in a stage. If stage 0 fails, exit before any larger runs. If it passes, execute the three prespecified levels and report the first failing level. A full-context control failure at that level leaves the Cell-specific explanation unresolved. Later passing levels cannot erase an earlier failure. A pass at every tested level permits a separate 20–30-fact confirmation.

Publish per-target log probabilities, token log probabilities, margins, gold mean-token NLL, position, total tokens, write order and persisted state hashes. Include per-capacity group and fact counts, results by position, frozen fixtures, source and model provenance, and the CPU receipt. Stage 0 also records the no-memory prior for both candidates.

In Kaggle use the Python Script `kaggle_reference_capacity.py`, one visible Tesla T4, Internet and the `GITHUB_TOKEN` secret. CPU regression checks precede the real GPU gate in the same run. Results are published to an independent GitHub branch. Conclusions apply to this two-candidate protocol.
