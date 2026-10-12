# Analysis plan

1. Audit canonical Core 007 confirmation artifacts for seeds 80721 and 80722.
2. Determine whether per-example projected states, oracle/deploy mode assignments, Cell parameters, logits/NLL deltas, or equivalent sufficient statistics are already persisted.
3. If sufficient, compute disagreement-conditioned functional regret and normalize it by Cell contribution magnitude.
4. If insufficient for exact output-distance/KL reconstruction, compute every valid artifact-only diagnostic and document the minimal missing state required for a definitive test.
5. Do not rerun or alter Core 007. Any future recomputation must be a new preflight protocol with fresh data identity.
