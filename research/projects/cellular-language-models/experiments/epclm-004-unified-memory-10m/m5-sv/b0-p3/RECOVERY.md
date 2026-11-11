# Post-run summary recovery

The original run completed arm publication and GPU checkpoint replay, then failed while constructing its cohort summary. In `launch`, an evaluation-loop string named `key` overwrote the experiment identity mapping. The final mapping expansion therefore raised TypeError. The fix uses distinct `experiment_identity` and `metric_key` variables and validates the identity type. Notebook errors now identify B0-P3; the structured error records the failing function and line.

`recover_summary.py` is a separate read-only audit path, with no training invocation. It retrieves the three fixed original arm refs, preserves their commit identities, validates common manifests, independently downloads each complete Release archive, verifies archive/member SHA256, regenerates only authorized Train/Validation data, and checks payloads, checkpoints, curves, data order and token ledgers. It recomputes the Stage1 gates. Original GPU numerical replay proofs remain attributed to the original run; recovery does not repeat GPU evaluation or training.

Run `python recover_summary.py --output <new-recovery-directory> --publish`. An optional `P3_ARCHIVE_DIRECTORY` supplies already downloaded E0.tar.gz, E6.tar.gz and E7-Innovation.tar.gz files; hashes remain mandatory. Publication creates a new main-derived summary branch, independently verifies its remote commit, and never pushes to an arm branch. Test remains sealed. The resulting summary distinguishes planned Stage2 budgets from actually completed stages.

Regression tests reject a string identity, mismatched cohorts and inconsistent dataset identities. Historical artifacts and checkpoints remain immutable. Recovery success does not make a blocked learnability gate pass.
