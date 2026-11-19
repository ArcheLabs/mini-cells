# HAM-003 — nonlinear memory complementarity

See [historical audit](HAM-003-AUDIT.md). This directory contains the archived
baseline recomputation and initial deterministic vector task generators. The
composition generator tests seen relation/entity bindings only. Unseen pair
generalization and alternate templates are not yet implemented. Multi-hop,
open-set and continual generators expose cases for the future harness; no
scores have been produced. M2–M7 are pending.

Run `python evaluate.py OUTPUT` for local standalones, or add
`--titans-upstream PINNED_CHECKOUT`. Kaggle CPU entry: `kaggle_ham_003.py`
(Accelerator None, Internet, GITHUB_TOKEN). No training. Multi-hop is symbolic
candidate decoding followed by another read; open-set uses a fixed diagnostic
norm threshold, not a calibrated rejection system. Results are screening on
three paired seeds; no hybrid or representation claims.
