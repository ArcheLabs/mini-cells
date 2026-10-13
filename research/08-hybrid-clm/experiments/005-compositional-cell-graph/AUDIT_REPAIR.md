# Audit repair record — 2026-10-13

This revision repairs the audit of implementation commit `3a073bf`. Protocol status remains **PROPOSED_NOT_REGISTERED**. CPU engineering checks cannot authorize GPU scientific training or replace the trained synthetic Gate 1.

| Finding | Repair and evidence |
| --- | --- |
| B1 field mismatch | Analyzer uses `b1_ppl_relative_improvement_min`; five-arm decision and archive regression exercise the actual protocol schema. |
| Graph metadata read from wrong level | Read top-level G0 `graph_learning` and `posthoc_diagnostics`; positive and negative mechanism fixtures reach the B2 gate. |
| Confirmation seeds rejected | Shared seed-role validator accepts disjoint registered development/confirmation lists. Metrics and checkpoint identities include role; development results cannot complete the confirmation matrix. |
| Notebook fetches main without 005 | Clone the research branch; default source is a full audit-fixed implementation SHA. Replace it with the registration commit SHA before training. |
| Inadequate graph exploration | Probe four cyclic source rows per event; all 33 sources are covered within nine events. Force a predecessor for unused Cell counterfactuals, log natural support separately, and require complete effective per-layer coverage. Insufficient exploration yields INCOMPLETE. |
| Different auxiliary objectives | Every arm uses the same probability-concentration objective and coefficient; sequential arms average over hops. Shared tensors initialize from name/seed-derived streams independently of module creation order. |
| Perturbation improvement accepted as benefit | Signed PPL degradation is required. A better perturbed graph cannot support beneficial learned connections. Absolute sensitivity is a different observation. |
| Too few p95 samples | Thirty cache-reset repetitions, 33 tokens each, yield 960 warmed samples. Bootstrap resamples whole repetitions. Timing excludes FLOP/trace hooks; profiler reports CUDA kernel aggregates separately from host/device elapsed. |
| Probe cost missing from totals | Include baseline and every candidate intervention in probe GEMMs; sum training optimizer-forward and probe GEMMs in `total_training_forward_gemm_flops`. Backward/optimizer FLOPs are not inferred. |
| Correlated NLL windows | Freeze a fixed seed for disjoint within-story holdout windows; persist story ids and resample paired story clusters. |
| No final research closure | Follow the review and registration process below. Automated support remains evidence pending human review. |
| Shared data loader regression | An offline fixture confirms default arrays/metadata are unchanged and verifies the optional prefix content hash. |

Counterfactual probes intentionally force a source even when its predecessor edge is absent from the current graph. Their losses measure downstream candidate utility under that intervention; they are not evidence of natural route use. Natural support, counterfactual effective tokens, changes and final route utilization are separate recorded fields. S0/R0 perform the same number of scheduled training-only interventions and discard topology updates. Actual executed compute can differ and is reported.

The runtime threshold, graph diagnostic thresholds, common auxiliary objective and measurement procedures are explicitly **audit-fixed proposals**. They are not registered merely because code can execute them. Remaining candidate counts, probe identities, independent holdout content and the disjoint seed matrix must be fixed and reviewed before any confirmation evidence is observed.

## Registration and final review

1. Prepare independent assets; verify the 004 regression-prefix identity, disjoint holdout and train-only probe file. Freeze content hashes, probe counts/candidate count/replacement rules, the development and untouched confirmation seeds, perturbation seeds, and every proposed measurement/decision rule in config and protocol. Record any required S0-shortlist control explicitly.
2. Run CPU preflight and the audit regressions. Run the proposal's trained synthetic Gate 1 separately and archive its evidence; the deterministic fixture is insufficient. Review this evidence and the frozen definitions. Only then set config `protocol_status=registered`, protocol `status=REGISTERED`, definition and registry `status=registered_unrun`, clear resolved open items, and commit. Pin this full commit in the Notebook.
3. Run the registered development and confirmation arms. Preserve failures and all negative results. Profile latency with other training stopped. Append immutable arm archives and the content-addressed analysis report. Do not alter registered rules after observing confirmation results.
4. A named research reviewer creates a project-owned immutable report containing source SHA, config/protocol/assets hashes, analysis input/report hashes, confirmation seeds, each B1/B2/System decision, limitations, approval or rejection and review date. Reviewers must reject support when evidence is incomplete. Negative completed results remain publishable scientific outcomes.
5. Commit the review and link it from the experiment README, conclusion and registry. Set the registry's final decision/status from that approved review. `scientific_decision: false` in automated reports denotes the absence of human review; it never silently promotes computational gates to a research claim.

## Verification

Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/research/routing_005` and `python scripts/research/run_experiment_005.py --mode preflight`. Tests use tiny CPU models and fixed offline fixtures, including exact next-step optimizer/RNG/graph restoration and source-wide actual probes. No dataset download or GPU is needed. The plugin flag isolates unrelated installed pytest plugins.
