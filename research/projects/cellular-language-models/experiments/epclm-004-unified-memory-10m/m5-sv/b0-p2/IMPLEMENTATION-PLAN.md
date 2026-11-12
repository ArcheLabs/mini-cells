# Implementation and execution sequence

1. Review current main ce9f1905b8d9a8badd2dbed111f68e180e71a619 and the verified B0 source8bb11b03da52ecf9ae74bca8846eb4ece065a091. Preserve existing implementations. See REPORT.md for findings.
2. Reconstruct the frozen previous Train/Validation banks, check original hashes, partition the4252-pair pool, create new Episodes and audit all per-symbol marginals. Freeze generated bank hashes in data/. Enforce the user's approved0.02/0.01-bit JSD limits.
3. Reuse b0_models, b0_data, b0_metrics, the private M2 trainer and atomic core checkpoint utilities. Add only P2 observations, stable probability diagnostics and a new entry. Validate tiny CPU updates, state isolation, exact continuation, archive integrity and deadline enforcement before GPU use.
4. One launch of notebook.ipynb in bingjiangli/minicells runs the actual-environment CPU tests, reconstructs/verifies data, independently retrieves prior complete artifacts, measures temporary GPU phases and freezes one common update budget. No formal P2 training occurs before this gate.
5. Execute each freshly initialized arm under the shared persisted7200-second GPU budget. Training and evaluation always use complete512-token Episodes; do not resume older experiments or tune against Validation.
6. Reuse publication utilities, independently download full archives and verify every file, then run downloaded-checkpoint re-evaluation within the remaining original GPU budget. Return per-arm and combined reports, paired Episode-bootstrap analysis and remote SHA proofs. Any failed gate stops progression.

## Files

- p2_dataset.py: deterministic split/packing and complete marginal audit.
- p2_metrics.py: read-only telemetry, full-probability diagnostics, NLL decomposition and Episode intervals.
- p2_training.py: private original M2 adapter, optimizer observations and new checkpoint export normalization.
- worker.py: isolated exact-model benchmark, bounded formal train or independent restore phase.
- p2_integrity.py: complete independent downloads, payload/probability verification and private archive writer using the original publisher.
- p2_run.py: one controller, cumulative budget, exact namespace/resume guards and final return.
- notebook.ipynb: one Save & Run All; original Notebook and secret labels.

No new third-party dependency, model, architecture layer, training framework, optimizer rule or inference kernel is introduced. Generic module names were avoided because the historical research code adds multiple directories containing run.py and training.py to sys.path.
