# Shadow isolation and admission audit

One Cell birth and one connection/path proposal group are permitted in each arm. All resident slots are counted in both arms from initialization. A born Cell copies Cell 0 exactly into the private reserved slot; that slot is absent from accepted routing until a passing transaction.

Lifecycle: PROPOSED -> PROBATION -> EVALUATED -> PROMOTED / REJECTED / EXPIRED. Every transition carries a hash-chained record. Early proposals are blueprints; immediately before probation the private branch is explicitly rebased and logged against the **current** accepted model and optimizer. Thus a candidate evaluated against an old parent cannot be promoted into a newer one.

Probation updates only candidate Cell (for a Cell proposal) and routing dependencies on a private copy, with private optimizer/RNG. Shared embeddings, accepted Cells, Context and head are frozen in the branch. Neither branch gradients, AdamW moments nor data draws update the accepted model. QF and J0 use their respective learning mechanisms on both official and private branches.

Connection proposal: predecessor-conditioned learned edge-prior perturbation on the already legal graph; not a fictitious newly legal boolean edge. Path proposal: configured, target-independent causal prefix with each internal operator charged. It adds no resident parameters and is not a target-selected search answer. These simple proposal policies may be ineffective; zero promotion is valid.

Admission uses distinct story-disjoint Admission windows, fixed paired seed/policy/budget, NLL and lambda-normalized cost, grouped bootstrap, minimum story count and frozen positive-improvement/regression margins. Missing margins reject promotion, never default to zero; automatic development can continue with measured rejection. A passing candidate is chosen deterministically; other stale candidates expire after one transaction. Regression/causal contracts are checked independently of its apparent objective gain.

Commit loads the entire candidate parameter/routing dependency/registry/optimizer bundle into existing accepted Parameter objects, increments versions and clears label caches. Failure restores weights, optimizer, registry, version, cache, RNG and candidate lifecycle. Failure injection and negative/stale promotion tests cover this contract.

Checkpoint stores all active/terminal private branches and their optimizer/RNG/journal. Portable export copies the committed immutable checkpoint, configuration and assets with checksums. CPU copy/restore tests do not certify real Kaggle cross-session persistence; that gate remains false until a separately saved Output is attached in a new session.

Residual risk: synthetic transactions and toy gradients cannot establish useful Shadow maturation or safety on the language distribution. All proposals may fail Admission without invalidating a correctly executed negative experiment.
