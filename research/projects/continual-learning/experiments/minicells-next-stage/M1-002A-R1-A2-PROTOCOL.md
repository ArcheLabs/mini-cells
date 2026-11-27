# M1-002A-R1-A2 · Cross-Process Continuation Isolation

This diagnostic uses the identical Reference prefix, query token IDs, candidate order, model identity and T4 runtime from R1. The prior R1 verdict and artifacts are immutable.

A writes through the v2 SDK, captures the exact live Cache that produced its export, and executes the primary teacher-forced query on that same object. Candidate work after the primary trace uses independent copies of the pristine in-memory Cache. B and C open new Python processes, import A's exact Bundle bytes and execute the same query on their restored Cache. Each worker checks the persisted Cell and the shared export SHA-256 after its queries. PIDs, input hashes and raw starting tensor hashes verify independent execution and matched starts.

The pre-query gate is exact for all 263 tensors, their structure and Cache cursor. Query and post-query comparisons use the original per-field Live-vs-Live upper bounds from R1 artifact commit `4e65d9b5ca3012151baf6c4e241337703d3e228d`. The bounds file is pinned to SHA-256 `549b9b145b19f7a0231949e8fb650e11fac39057d46c104b9cadb22c08c693d3`. A field without a historical numeric allowance uses zero. The gate compares all three pairs for hidden states, logits, candidate scores and recurrent/Conv updates.

Publish pairwise deltas, first divergence, per-layer errors, lossless per-token hidden states and logits, state structures and pre/post tensor SHA-256. Large raw Cache traces and the Bundle remain in the Kaggle run's work directory. If any start differs, stop that worker before Query and publish the precise start mismatches.

A pass establishes continuation equivalence for this matched starting snapshot. R1-A's separate cross-process Append check is still required before R1-B. A2 never launches R1-B or capacity work.

Use `kaggle_continuation_isolation.py` as a Kaggle Python Script with Tesla T4, internet access and the `GITHUB_TOKEN` secret.
