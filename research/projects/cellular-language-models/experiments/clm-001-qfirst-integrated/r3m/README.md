# R3m — Stability–Reorganization Boundary

Independent study; historical models, implementations and results remain unchanged. No GPU experiment was run during implementation.

## Current release state

**Data registration blocked. Do not start scientific training yet.**

The protocol's candidate Story ranges `[10048,10304)`, `[10304,10560)`, `[10560,11072)` overlap Experiment 005's registered TinyStories train range `[0,50000)` at the same revision `f54c09...`. The registered source is `005-compositional-cell-graph/config/registered.json`; `data.boundaries()` reads that registration and stops before downloading new assets. The older governance dataset index does not provide a global row-level allocation ledger. R3m does not silently redefine global isolation as lineage-only isolation.

Resolve the intended isolation scope or register new verified unused ranges before freezing the runnable release. The included launcher deliberately fails this unresolved scientific input check.

The subsequent user-proposed amendment `[100000,101024)` also FAILED: historical Cellular Computing Experiment007 consumed476847 nonempty stories through raw index476900. Reproducing its120000000 uint16 training tokens at the requested Revision produced exactly the historical SHA256 `0d51d5779a168d4b65a1b5ff469669e42f46f69cba061ff7f7872a3f7290f8b8`. See `registration/DATA_BOUNDARY_PROOF.json` and `GLOBAL_DATASET_USAGE_LEDGER.json` (111 metadata files). No frozen range was changed. `[500000,501024)` is a new unregistered suggestion only, not automatically approved or content/window verified.

## Latest candidate audit: amendment002

The separately approved read-only `[500000,501024)` audit is also **BLOCKED**. Main Report contains150 exact historical Experiment007 training stories at different raw indices. Trigger/Pilot have no detected text matches, but no role is registered. See `registration/amendment-002/DATA_BOUNDARY_PROOF.json` and REVIEW.md. Frozen config/tokenizer/source/model are unchanged. The earlier500000-range suggestion above is retained as history and is superseded by this failed audit. No further range is chosen. Exact R3m token-window verification is not claimed without the original tokenizer artifact/observed historical starts.

## Implementation and reuse

- R3h B-H real Step300 receipts at a fixed HF revision, model/optimizer/RNG/cursor/router checks. No R3i adapted anchor substitution. Missing Step300 stops without retraining.
- Historical Cell trainer, eight real sequential calls, six Cells, self connections, existing loss and optimizer. Continuous fixed 600-step LR/epsilon schedule starts from the restored values; no warmup reset.
- 24 logical Pilot branches: three Seeds, N/C environments, P0/P1/P2/P3. Same 600 main updates and 512 tokens/update, fixed 90/10→70/30→50/50 curriculum. Two independent seed/environment shards per cloud launch, not a wall-time kill.
- Original Q-first collector/optimizer and 22-scale Trust Region rebound to **already-observed** Train batches. KL≤.01, action≤.10, path≤.20 and original margin/numerical checks unchanged. Candidate and scans persist before selection.
- Ten paired Shadow updates are part of the 600-step timeline. Old/new states, optimizer and streams are saved inside the transaction; restart resumes the pair. Ghost/sham work is separate from real retained training and does not imply equal measured total compute.
- P3 permutes row action/probability assignments, preserving each observed state's stage histogram and legal execution budget. Closed-loop aggregate usage and train/inference mismatch are reported; P3 alone cannot establish a causal stability benefit.
- Arithmetic canonical expressions, operands, answers and template families are disjoint. Full NLL, answer NLL, autoregressive exact match, carry strata and frequency baseline. Fixed report sampling uses 16 carry/16 noncarry canonical groups, two templates, and fixed Story groups. Task failure never becomes a Connection falsification.
- Separate report barrier, OO/ON/NO/NN at400/600, legal Cell replacements with fixed-ID, usage-matched and random-ID controls, normalized task specificity. Per-group bootstrap is descriptive and separate from independent-Seed inference.
- Reuse Store, selective restore and incremental HF transport. GitHub receives allowlisted metadata descended from main and verifies the remote commit, never checkpoints or workflow changes. `--report-existing` does not train.

## Kaggle launch after registration is resolved

Import `notebooks/kaggle-stability-reorganization.ipynb` into the original `bingjiangli/minicells`, use T4 x2, Internet and existing `HF_TOKEN`/`GITHUB_TOKEN`, then Run All.

The protocol explicitly requires actual cross-session verification. The first cloud run stores one real historical-model engineering update reference privately and performs **zero production updates**. Save Version and start a new cloud session; Run All the **same Notebook**. It downloads the pinned reference, compares the next update and continues automatically after success. No copied hashes, mode switches or Notebook test suite. A fresh process in the same container is rejected as insufficient proof.

Following launches automatically restore progress and select the next two Pilot shards. Three training launches cover six shards; partial reports remain inconclusive until the matrix completes. Publication failure can be retried from saved states without replacing training. This launch plan is a workload partition, not a measured duration guarantee.

## Main preparation and unresolved freezes

Main uses separate phase identity, five new Seeds, five strategies, shared PhaseI once/Seed, 600-step branches, at most two proposals, signed historical-gradient/loss detector, reserved Ghost opportunities, seeded random schedules, two training or evaluation shards/Session, independent Main Report and five Holm-adjusted Seed comparisons.

Main cannot run until a reviewed positive Pilot and a frozen release manifest are present. Still unfrozen: Main initialization mapping to verified pretrained Step0 sources, Train-only detector calibration/threshold, independent scientific effect/cost thresholds, and accepted cross-session proof. No Main Report is materialized in Pilot.

The gradient conflict observation is a **fixed signed gradient sketch** (32 components/tensor, zeros for unused Cells), not the full dense gradient cosine. This choice requires explicit review before Main calibration. Five Seeds give coarse exact sign-flip p-values; Holm correction does not create power or independent confirmation.

## Historical context reviewed

022 reports `EMERGENT_TRAIT_BIFURCATION_SIGNAL`; 023b reports `PROBATIONARY_TRAIT_GENESIS_SIGNAL` with Story-only/duplicate rejection and Story+Arithmetic acceptance. Their task-aware capacity controls and temporary trait branches motivate separate domains and sham trials; their models/results are not transplanted as R3m evidence. R3h/R3i/R3l provide the actual six-Cell engineering lineage.

## CPU validation

`python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m/checks.py`

Synthetic only: real Q-first/Trust integration, all state restores, interrupted Shadow resume, update/token budgets, role isolation, masked answers, read-only quality/four-grid evaluation, complete negative report matrix, Main gate/statistics and once-only shared PhaseI. GitHub Actions uses the same suite; Kaggle does not run it. CPU success is not real HF/T4 persistence proof or scientific support.
