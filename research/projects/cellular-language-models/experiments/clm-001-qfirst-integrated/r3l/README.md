# R3l — Connection Reorganization Potential

Import `notebooks/kaggle-reorganization-potential.ipynb` into the original Kaggle Notebook. Start a **new cloud session**, use **T4 x2**, Internet and the existing **HF_TOKEN/GITHUB_TOKEN**, then **Run All once**. No separate pilot, manually copied source hash or Kaggle test suite.

## Fixed study

Twelve real R3i **official pre-Proposal Step300** anchors, three Seeds and A-CH/A-CT/B-CH/B-CT. Same six Cells, existing Pair registry, H/T routing and eight Cell calls per token. No model architecture changes, Connection optimizer, Cell lifecycle or new learning objective.

Each anchor has15 fixed logical candidates: Baseline, actual Historical, Reverse,8 Random and4 Structured. Dedicated CPU random streams and independently L2-normalized parameter blocks are frozen in config/code before SEARCH. Historical and Reverse do not bypass the original Trust Region. The existing R3f-v2 scan/decision functions are reused; only the storage callback is synchronized with transport. All rejections are retained.

SEARCH uses32 windows and SELECT128. SEARCH top4 across all valid non-Baseline families receive independent SELECT evaluation. RANDOM_CONTROL is the first valid random Candidate hash, never selected by quality. Finalists are immutable before REPORT access; matching parameter states share one physical continuation while retaining logical labels.

All branches start from the same official Step300 Cell/optimizer/RNG/cursor and continue100 Cell steps through400, without Proposal dispatch or warmup reset. Connections remain frozen. Store retains0/10/50/100 and the newest complete state. At most4800 updates /2457600 new training tokens; shared branches reduce actual cost.

REPORT evaluates all four horizons only after the complete training barrier, using the existing repeated four-grid evaluator. V=R+C; interaction stays separate. Story-paired bootstrap2000, Seed-level exploratory comparisons and Holm adjustment are reported separately from engineering success. Only three independent Seeds; no independent-confirmation claim.

The independent finite Beam diagnostic uses the actual `initial/select/transition/head` operations on four fixed anchors,16 fixed token positions, Beam8. It is explicitly **target-aware and non-deployable**, not a global optimum or mathematical quality bound. It never influences candidate selection or training.

## Storage and validation

Full states stay private on HF. Reuse the incremental publisher and Store/authentication primitives. Selective restoration fetches required horizons and latest states, preserves newer verified local progress and keeps restore proofs outside immutable checkpoint metadata. Completed phases are reused. `--report-existing` regenerates/publishes reports without training; GitHub receives only allowlisted aggregates descended from current main, with remote commit SHA verification.

CPU tests use synthetic inputs, cover A/B execution budgets, deterministic directions, identity/optimizer restore, candidate/role isolation, deduplicated continuation, frozen Connection, no Proposal dispatch, repeated evaluation, finite legal Oracle paths and reports. Real private HF/data/T4 checks run automatically in Kaggle. CPU success does not certify a real cross-session Kaggle run or scientific findings. Historical code and results are unchanged.
