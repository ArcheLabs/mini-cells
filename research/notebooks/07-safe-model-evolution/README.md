# 07 — Safe Model Evolution

This notebook group contains current model-level Safe Model Evolution experiments built on real pretrained MoE/CLM substrates.

Scientific authority lives in the matching frozen validation protocol and durable experiment artifacts. Notebooks are orchestration, recovery, and visualization entry points.

Current workflows:

- `hybrid-clm-prompt-address-001/hybrid-clm-prompt-address-001-kaggle.ipynb` — frozen diagnostic of prompt-anchor routing for Granite Hybrid CLM v0.1. It tests held-out address generalization, zero history-anchor false positives, unchanged write-safety gates, and fresh reload. Scientific PASS and FAIL are both published.
- `clm-conversion-kill-test-001-kaggle.ipynb` — current frozen formal test of whether a mature frozen Granite MoE can grow a zero-init, cross-layer, semantically addressable functional-Cell overlay and preserve local mutation, child growth, branch merge, and rollback. GPU formal seeds are pending.
- `jam-knowledge-mutation-001-failure-diagnostic-kaggle.ipynb` — completed post-hoc forward-only diagnostic for the repeated JAM001 misconception-gate failure; it reused all nine published mutation artifacts and preserved the upstream formal No-Go.
- `jam-knowledge-mutation-001-kaggle.ipynb` — completed release-oriented real-domain JAM knowledge acquisition experiment; frozen decision `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED` (0/3 formal seeds PASS).
- `history-compression-001-kaggle.ipynb` — completed replay-budget ladder after Functional Boundary Oracle 001; learner-visible historical prompt budgets `32 -> 8 -> 2 -> 0`.

Related evidence:

- Functional Boundary Oracle 001 — `FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED`, 3/3 formal seeds.
- History Compression 001 — `HISTORY_COMPRESSION_TO_8_SUPPORTED`; `full_32` 3/3, `tiny_8` 2/3, `tiny_2` and `zero_0` 0/3.
- JAM Knowledge Mutation 001 — `JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED`, 0/3 formal seeds.
- JAM Knowledge Mutation 001 Failure Diagnostic — `CANONICAL_CONTENT_GAIN_ALSO_BELOW_ORIGINAL_THRESHOLD_AT_CAPACITY4`; the prefix-dilution rescue hypothesis was rejected and fixed sparse mutation was shown to favor cheap output/template behavior over canonical semantic content.
- CLM Conversion Kill Test 001 — protocol frozen; no scientific decision until all three untouched formal seeds are durably published.

Rules for this group:

1. Freeze protocols/formal seeds before formal GPU execution and freeze diagnostic plans before post-hoc GPU analysis.
2. Scientific FAIL results are published, not discarded.
3. Post-hoc diagnostics may explain a frozen failure mode but may not rewrite the formal decision.
4. Notebook stdout stays compact; detailed training/coordinate/diagnostic data are durable files.
5. Training, checkpoint-validation, and final heldout data must remain separated according to the frozen protocol; final heldout data must never drive checkpoint selection.
6. Formal result publishers must reject seed artifacts whose protocol or dataset identity differs from the currently frozen experiment.
7. Visualizations are derived views; `result.json`, `decision.json`, and diagnostic result files remain canonical evidence for their respective scopes.
8. Research notebooks belong under `research/notebooks/`, not repository-level `notebooks/`.
9. If a hosted workflow publishes result commits back to its PR branch, do not merge that PR before the intended results have been durably published.
