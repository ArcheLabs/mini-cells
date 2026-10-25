# R3m-F-D1 Kaggle entry

Human authorization for B/C/D was received on 2026-10-25; see [RUN_AUTHORIZATION.md](RUN_AUTHORIZATION.md). Import [`notebooks/kaggle-r3mf-d1.ipynb`](notebooks/kaggle-r3mf-d1.ipynb) into the existing Kaggle Notebook. Select **T4 x2**, enable **Internet** and the existing **HF_TOKEN**, then use **Save Version → Save & Run All** once. No code switch or manually copied hash is required.

The Notebook defaults to `RUN_BCD=True`. It fetches execution source `16df822e95ac2466529f424a8ebfec63ee93c295`, runs the frozen CPU Gate automatically, and continues only if the checks pass. B/C/D each retain at most 200 logical updates, with evaluation at 0/50/100/200. A reuses the historical result. The full R3m-F Pilot, Main and Confirmation remain outside this authorization.

Full-state checkpoints are retained every 25 updates and at evaluation barriers under `/kaggle/working/r3mf-d1-output`. The final per-arm predictions and combined `RESULTS.json` are in the same Output tree. `KAGGLE_API_TOKEN` and `GITHUB_TOKEN` are not required; this Notebook retains results in Kaggle Output.

The 200-step budget limits logical trajectory length. Failed work since the last checkpoint may be repeated, so cumulative retry computation can exceed 200 optimizer calls. Manifests distinguish logical progress, unknown cumulative retry costs, requested GPU and observed GPU use.

Engineering validation includes 21 CPU tests, the pure CPU Gate and the exact sparse-checkout entry. Actual B/C/D CUDA execution is performed by the authorized Kaggle launch; it is not inferred from the CPU tests.
