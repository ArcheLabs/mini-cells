# R3m-E — Capability Differentiation & Reorganization Pilot

Independent exploratory experiment `CLM-001-R3m-E-Capability-Differentiation-001`.
It reuses the R3h B-H model and verified Step 300 checkpoints, plus the existing R3m
Cell trainer, Q-first proposal, Trust Region, paired Shadow transaction, checkpoint store,
and HF transport. It does not change R3m's blocked registration, data candidates, or history.

For the released exploratory matrix, use `notebooks/kaggle-development.ipynb` with T4 ×2, HF_TOKEN and GITHUB_TOKEN, then Save & Run All. It reads the existing Gate 2 proof and automatically trains, evaluates and publishes. A new cloud run restores the same source/config from private HF.

The original `exploratory.frozen.json` remains engineering-only. `python run.py --mode cpu-gate` runs the small
CPU checks. `--mode auto` runs the CPU gate, fetches the allowed Train and reused Development
assets, and hash-verifies the three pinned Step 300 checkpoints on CPU. `--mode smoke` is a
separate, disposable one-update-per-task real-checkpoint GPU check; it never advances the
study cursor or writes study checkpoints. The engineering Notebook still defaults to `auto`. The separate `exploratory.released.json` authorizes the original 600-step-per-arm matrix after completed cloud Gate 2 evidence; the development Notebook selects it explicitly.

The experiment reuses R3h historical development data and discloses known historical overlap.
It is not an independent confirmation study. Confirmation and the original R3m Main candidates
remain inaccessible.

CPU parent inspection retains the original checkpoint and records the original CUDA
RNG count/hash, while omitting CUDA RNG application to the temporary CPU copy.
This verifies model/optimizer identity only; it does not certify GPU restart.
Production GPU recovery continues to use the unmodified strict historical loader.


For the remaining Gate 2 checks, import `notebooks/kaggle-gate2.ipynb`.
Select T4 ×2 and run it once, save the version, then start a new Kaggle Session
and Run All in the same Notebook. Private HF transport automatically finds the
pinned mid-Shadow checkpoint and uninterrupted reference; no Input mount or
Kaggle version parameter is needed. The first pass performs 26 disposable Cell
optimizer updates plus the original 8-step Proposal. The second performs 14
Cell updates to finish and compare the paused trajectory. Actual Proposal
rejection is a valid engineering result. A separate fault injection tests rollback
without relaxing any scientific threshold. See `GATE2_PROTOCOL.md` for scope.
