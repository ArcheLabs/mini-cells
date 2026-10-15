# Active execution workflow override

Current user instruction: automatic preflight, continue after success, no multi-stage manual operation. This supersedes the original manual approval/configuration/session sequence only; SPEC.md and earlier evidence remain unchanged.

One Kaggle Run All:
1. Synthetic CPU correctness / parameter / Oracle / data checks.
2. Require two GPUs and preflight the existing metadata-only GitHub return path.
3. Prepare or verify isolated four-role assets; record source/tokenizer/asset/config identities internally.
4. Run private frozen Oracle repeatability, bounded functional updates, save/restore and continuation checks. Project full training work from measured times.
5. If functional checks pass and projected full training work is within two hours, continue both 500-step development arms on separate T4s. No manual mode/approval/session switch, process timeout/kill, or reduced scientific budget.
6. Analyze and return results. Retained checkpoints and completed results are reused; publication failure does not require retraining.

Confirmation stays sealed. Results are exploratory. Lambda uses the declared candidate 0.01 as recorded development configuration; scientific thresholds are not manufactured. Unfrozen Admission margins cause measured rejection, never automatic promotion. Quality findings are reported as results rather than converted into manual approval blockers.

The two-hour check is an advance workload projection, not a guaranteed total runtime. Local restart evidence is not falsely labeled real Kaggle cross-session persistence. Save Version retains output. The coding agent does not start real GPU training while implementing this workflow; verification uses CPU fixtures/mocked orchestration.
