# EPCLM-004 M5-SV

Scientific validation of the frozen M5 mechanism; no new memory architecture and no M6.

Read [SPEC.md](SPEC.md), [baseline audit](audit/m5-baseline-audit.md), [REPORT.md](REPORT.md) and [implementation decisions](implementation_deviations.md). All new repository prose is English; historical documents are preserved.

The single [Notebook](notebook.ipynb) targets the existing Kaggle Notebook with Tesla T4, Internet and existing secrets. It proceeds automatically through protocol checks, frozen data, development diagnostics, bounded Pilot, immutable freeze, three paired seeds and independent Test. Each session is limited to3 hours. Attach the prior saved Output for subsequent sessions; exact source/config identity is checked automatically. Completed updates and evaluation Episodes are reused.

A failed learnability Pilot is INCONCLUSIVE and stops further training. A timeout leaves PARTIAL work and publishes it. All required Test cases must finish before a scientific verdict. A completed engineering stage cannot set H5=SUPPORTED.

Implementation status: CPU component validation complete; real T4 execution pending. The regression suite and tiny orchestration evidence are under audit/. H5 remains NOT_EVALUATED until actual scientific runs. Main budget is12M input tokens; Pilot adds a separately bounded65,536 inputs. Capacity128 is an inference intervention on capacity64-trained weights.

## Version190 publication repair

The saved run hit ModuleNotFoundError for scripts.research at publication time. Direct execution did not reliably expose the repository root as a Python package; another scripts namespace can also shadow it. The entry now loads the existing publisher by its exact file path, registers its module for dataclass annotations, and runs a cheap import preflight before experiment work. Experiment status/error is printed before publication and retained separately if publication fails.

Use [publication-repair.ipynb](publication-repair.ipynb) for this saved failure. CPU is sufficient; enable Internet plus the existing GITHUB_TOKEN and KAGGLE_API_TOKEN secrets. It uses the already tested exact-version downloader for bingjiangli/minicells Version190 only. It downloads curated metadata, validates the original46f17f3 source/config/code identities, and republishes the original decision and logs with a repair receipt. It never calls the experiment launcher, loads checkpoint tensors, downloads sealed Test inputs or executes optimizer updates. Script Version ID stays unknown unless exposed by the exact-version API.

The repair preserves the original experimental status, including any Pilot failure hidden by the publication traceback. A successful metadata publication does not turn an experiment failure into scientific success. Check repair-receipt.json for the original status/error and the verified remote commit. Original M5-SV binary checkpoints remain in saved Kaggle Output; this repair does not claim a weight archive or scientific continuation.

Three targeted regressions passed: direct isolated-file import despite a conflicting scripts module, publication-only evidence preservation without weights, and original experiment error visibility when publication fails. AST comparison confirms all scientific entry functions are unchanged; only publish/launch transport and logging changed. Actual recovery/publication of Version190 must run in Kaggle, where the existing secrets are available.
