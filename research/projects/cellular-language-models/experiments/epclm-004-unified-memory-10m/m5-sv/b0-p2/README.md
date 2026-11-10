# B0-P2

Balanced Data and Extended Training Pilot for the unchanged E0/E6/E7 models. Development only; Test unaccessed, H1–H6 NOT_EVALUATED.

Read [SPEC.md](SPEC.md), [REPORT.md](REPORT.md) and [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md). Exact frozen bank identities and full per-symbol counts are in data/.

## Launch

Import notebook.ipynb into the existing bingjiangli/minicells Kaggle Notebook. Use Tesla T4, Internet and the existing GITHUB_TOKEN secret. Save & Run All once. The entry automatically verifies data/functional gates, reads prior real timings, benchmarks all three models, freezes the largest common1024/768/640-update budget fitting6480 seconds, trains fresh models, returns complete bundles and performs independent download/hash/checkpoint evaluation. The cumulative GPU-phase hard cap is7200 seconds, including benchmark, evaluation, checkpoint and restore. No manual source/seed/hash/budget fields or stage switching are required.

Preserve complete Output and attach it for same-source resume. GPU charges persist across sessions; resuming cannot reset or expand the original cap. Completed training and evaluation are reused. A published run without its completed Output stops instead of retraining. Do not attach an old checkpoint as P2 initialization.

CPU review: `OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python p2_run.py --output /tmp/b0-p2-review-new`.

CPU review does not run canonical training. Real execution requires the Notebook's --execute invocation and an actual T4. If source, frozen data, initialization, environment, update budget, results or available time differ, retain Output and report the failure. No algorithm/precision/model-size fallback is implemented.

## Outputs

Each model bundle contains run identity, protocol/config/data/init/environment/source hashes, the actual train/validation/diagnostic banks, per-update logs, initial and scheduled full probability arrays, NLL decomposition, prediction distribution, read-only state/Store observations, training/validation curves, resource costs, complete checkpoints and a per-file SHA256 inventory. During execution, training/checkpoint.pt is the atomic durable128-update checkpoint; latest/final export paths are created only after completion.

The original publisher's upload/read-back is followed by a separate full download, independent member verification, downloaded-checkpoint restore and repeated diagnostics. Main-derived per-arm result indexes contain final verification reports; an additional summary index preserves paired Episode-bootstrap comparisons and the complete cumulative GPU budget. No scientific-success label follows automatically from training completion.
