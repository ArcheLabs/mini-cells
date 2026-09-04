# Engineering status

Current state: **ENGINEERING READY — DIAGNOSTIC GPU RUN PENDING**

The failure diagnostic is post-hoc and forward-only. It does not modify JAM Knowledge Mutation 001 training, protocol, dataset, formal gates, or the frozen `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED` decision.

Dedicated engineering guard run `33857011366` completed successfully. It verified:

- frozen upstream JAM001 protocol, dataset, decision, seed summaries, and all nine mutation artifact identities;
- byte-level immutability of the upstream protocol/dataset/formal-artifact paths relative to main commit `762873b525d230fb36acc472b8994bbf7b53525a`;
- diagnostic plan JSON validity;
- runner compilation;
- Ruff checks;
- CPU tests for source admission, exact prefix/content/EOS token partitioning, gain reconstruction, and post-hoc classification logic.

No diagnostic GPU forward pass has been opened yet. Hosted execution must use `jam-knowledge-mutation-001-failure-diagnostic-kaggle.ipynb` and publish the resulting `diagnostic.json` plus 441-row `per_row.jsonl` back to this branch before merge.
