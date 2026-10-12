# Experiment 002 — Cell Internal Router

Historical entry remains `experiments.cell_internal_routing_002.train`; the new
directory adds a six-section evidence notebook and thin compatibility scripts.
Authoritative configuration remains
`experiments/cell_internal_routing_002/configs/experiment.json` and frozen protocol
remains `research/validations/routing-002-cell-internal-routing/protocol.json`.
Published paths and checkpoint loading remain stable.

Only seed 42 is observed; the three-seed development matrix is incomplete.
Seed 42 did not support the registered efficiency/quality gate. Learned paths
are descriptive evidence; they do not prove general CLM superiority.
The thin evaluator uses the historical `evaluate_checkpoint` function through
a new repair adapter: `scripts/evaluate.py --checkpoint ... --original-output ...
--output NEW_DIR`. It verifies original data identity and writes diagnostic
evidence into a new directory without changing the original decision.
