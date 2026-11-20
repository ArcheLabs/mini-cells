# HAM-003 — historical audit

HAM-002 uses rectangular Delta state, a gated variant that decays state before applying the error update, and the pinned HAM-001 Titans adapter (one native store per token). Both branches write independently from the same input. Fixed fusion adds scaled Titans output to Delta output; coefficient zero preserves Delta output exactly but still maintains both states. Transfer copies state into newly initialized compatible instances.

The historical evaluator shares generated queries across arms and chooses the coefficient on validation seeds. The archived rows reproduce lambda=0 and approximately 51.88% mean holdout random KV Top-1 for Delta, versus 60.87% for beta=0.5. State sizes are descriptive. The XOR task has only one held-out combination and cannot establish general nonlinear rule induction.

Plan: freeze paired tasks and splits, then verify standalone memory on unseen combinations and distractor streams. Retain the historical adapter and update math. Introduce fusion or alignment only if the standalone evidence supports it.
