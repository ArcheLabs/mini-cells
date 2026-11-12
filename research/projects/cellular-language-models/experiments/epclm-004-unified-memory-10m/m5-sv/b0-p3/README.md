# B0-P3 — Minimum Associative Recall

Read SPEC.md, REPORT.md and IMPLEMENTATION-PLAN.md. Single-seed calibration with the unchanged E0/E6/E7 models. Only active-candidate loss is trained. Test remains sealed/unaccessed; H1–H6 are NOT_EVALUATED.

Import notebook.ipynb into the existing bingjiangli/minicells Kaggle Notebook, enable Tesla T4/Internet and the existing GITHUB_TOKEN secret, then Save & Run All once. The entry automatically performs CPU gates, actual64/128-slot Batch8 GPU probes/benchmarks, common-budget freeze, all-arm Stage1, the predefined cohort gate, conditional continuous Stage2, optional budgeted diagnostics and complete independent artifact verification. No source/hash/seed/budget form filling or manual stage switching.

Data contain64/128 slots with59/115 valid tokens. The adapter omits trailing PAD equally for all models and retains the complete real context. Primary metrics teacher-force earlier answers and score ANSWER→next VALUE causally. Native batch states are independent. Stage1 online75% uses all8192 pre-update targets; severe collapse is the owner-approved active-candidate criterion, not full-vocabulary collapse.

All arms must pass Stage1 before any Stage2 training. A blocked cohort still returns complete Stage1 diagnostic artifacts. It does not enlarge Stage1 exposure or let selected arms continue. Stage2 retains each own trained model/optimizer/RNG/global step and changes only the authorized data level.

Cumulative charged GPU phases share7200 seconds, including prototypes, raw logs, checkpoint, Validation, optional diagnostics and downloaded-checkpoint replay. Formal planning must fit6480 seconds at a common Stage2 budget; minimum192 failure stops all formal training. Charges and budget selection persist across sessions. Preserve/attach complete Output for exact-source resume; completed updates/evaluations are reused. Missing published completed Output stops instead of retraining.

CPU-only review: `OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python p3_run.py --output /tmp/b0-p3-review-new`.

The local CPU suite and dataset audit do not pass the actual T4 or Stage1 learnability gates. Real execution requires the generated Notebook and --execute. Source/model/precision/context/budget fallbacks are not implemented. Test data and Test hashes are not inspected; Train/Test mapping disjointness is explicitly unverified while sealed.
