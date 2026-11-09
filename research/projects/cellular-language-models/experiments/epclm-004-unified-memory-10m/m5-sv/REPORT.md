# M5-SV implementation and execution report

## Executive summary

Status: IMPLEMENTED_CPU_VALIDATED_T4_PENDING. H5: NOT_EVALUATED. No M5-SV T4 training or independent Test evaluation has run in this development environment. The frozen M5 run was a successful engineering Smoke, not evidence of H5. Actual measurements must come from the ordered Notebook and its archived outputs.

## Experimental setup

The versioned protocol declares unchanged M5 Mamba-1/SWA/EMA/embedding Store semantics, the frozen8192 tokenizer, independent pair partitions, three separately recorded model/data/store seeds, four trained arms, an evaluation-only Oracle and a sealed common Test bank. Main training is12M inputs; a separate Pilot is65,536 inputs total. Hardware target: Tesla T4, FP32, TF32 disabled. The run manifest records actual source/config/data/tokenizer hashes and execution identity.

## Measurements pending

SV1: per-record scores/recoverability, AUROC/AUPRC, admission precision/recall/coverage. SV2: per-layer ranked candidates, final-layer retrieval, Store On/Off, conditional EM, rescue/regression. SV3: all seeds/conditions, validation adequacy, paired Episode bootstrap, independent Test. Resource analysis: full static/runtime/Store breakdown, training peak, inference timings and checkpoint cost at both capacities. None of these scientific results are inferred from M5's Smoke.

## Scientific conclusion

Innovation validity: not evaluated. Admission advantage: not evaluated. Retrieval and utilization gains: not evaluated. Stable final accuracy gains: not evaluated. Resource tradeoff: not evaluated. Do not advance automatically to M6.

## Actual local verification

Eight focused regression tests passed. They cover structural split isolation, unchanged history inputs and metadata alignment, Query-fork isolation, Store On/Off, diagnostic Oracle retention, known-answer metrics and score ties, undefined one-class AUROC, unchanged native computation under observation, existing atomic checkpoint continuation, Pilot failure stopping main/Test, missing-Output fail-closed behavior, and conservative scientific verdicts.

A separate CPU tiny orchestration fixture committed4 real optimizer updates and128 input tokens, saved validation boundaries at updates2/4, reused the completed training record without further updates, and restored a separately persisted Runtime with an identical state digest. This is synthetic component evidence, not target-GPU execution. A second fixture checked actual Query aggregation, frozen-bank hash reuse and crossed paired-seed/Episode Bootstrap. Frozen baseline hashes, Notebook/Python/config syntax and report generation passed.

Evidence files: audit/development-pytest.log, audit/cpu-driver-fixture.json, audit/aggregation-fixture.json and audit/static-and-aggregation.json. No original M5 code/checkpoint/result was modified. The three requested historical documents received English workflow addenda. All newly written repository prose and comments are English.

## Delivery commits

- 5821fd3: frozen M5 baseline audit and source identities.
- 3bd370d: English specification, versioned configuration, data/instrumentation adapters and workflow documentation.
- a9d0edc: metrics tests and bounded automatic scientific pipeline.
- 6362766: resume-evidence checks, crossed Bootstrap, resources and reporting.
- Later delivery commits finalize English reports and local verification; exact hashes are available from branch history and the running source manifest.

## Outstanding execution

The full frozen Test bank has not been generated or evaluated in this workspace. No canonical10M T4 Pilot,12M main training or scientific Test run has occurred. Run notebook.ipynb on Tesla T4. Sessions are bounded and resumable; no completed unit is deliberately retrained. If Pilot learnability is insufficient, main training remains stopped with an explicit INCONCLUSIVE report. If full evaluation cannot finish, H5 cannot be SUPPORTED.

## Version190 publication repair

The saved run hit ModuleNotFoundError for scripts.research at publication time. Direct execution did not reliably expose the repository root as a Python package; another scripts namespace can also shadow it. The entry now loads the existing publisher by its exact file path, registers its module for dataclass annotations, and runs a cheap import preflight before experiment work. Experiment status/error is printed before publication and retained separately if publication fails.

Use [publication-repair.ipynb](publication-repair.ipynb) for this saved failure. CPU is sufficient; enable Internet plus the existing GITHUB_TOKEN and KAGGLE_API_TOKEN secrets. It uses the already tested exact-version downloader for bingjiangli/minicells Version190 only. It downloads curated metadata, validates the original46f17f3 source/config/code identities, and republishes the original decision and logs with a repair receipt. It never calls the experiment launcher, loads checkpoint tensors, downloads sealed Test inputs or executes optimizer updates. Script Version ID stays unknown unless exposed by the exact-version API.

The repair preserves the original experimental status, including any Pilot failure hidden by the publication traceback. A successful metadata publication does not turn an experiment failure into scientific success. Check repair-receipt.json for the original status/error and the verified remote commit. Original M5-SV binary checkpoints remain in saved Kaggle Output; this repair does not claim a weight archive or scientific continuation.

Three targeted regressions passed: direct isolated-file import despite a conflicting scripts module, publication-only evidence preservation without weights, and original experiment error visibility when publication fails. AST comparison confirms all scientific entry functions are unchanged; only publish/launch transport and logging changed. Actual recovery/publication of Version190 must run in Kaggle, where the existing secrets are available.

## Shallow publication checkout correction

The first repair reached publication but the isolated clone could not check out the original46f17f3 commit. A source commit fetched only into the parent checkout's FETCH_HEAD can be absent or hidden after cloning a shallow checkout. Publication now explicitly fetches that exact SHA from origin into the isolated checkout, verifies FETCH_HEAD equals it, then checks it out. It never substitutes current HEAD or main for the original source.

Four focused publication regressions pass, including real shallow Git repositories and the complete artifact-preparation path (only the final authenticated GitHub push is replaced). A separate read-only check against the real GitHub repository fetched and checked out the exact original commit and matched source bytes. No experimental execution, tensors or optimizer state were touched. Repeated publication-only runs also preserve original evidence while excluding generated receipts from original-file comparisons.

Run the same publication-repair.ipynb again; it fetches the updated repair branch automatically and still reads Version190. The original experimental status/error is now printed before publication so a publication error cannot hide it. Live Kaggle result publication remains to be executed; local source verification is not claimed as a completed Kaggle repair.
