# R3m-F-D1 Kaggle entry

Import [`notebooks/kaggle-r3mf-d1.ipynb`](notebooks/kaggle-r3mf-d1.ipynb) into the existing Kaggle Notebook and use **Run All** once. It fetches the pinned source commit, checks the original Tiny records and tokenizer, and runs the CPU Gate. The Notebook defaults to `RUN_BCD = False`; this does not load a model, use a GPU, or make optimizer updates.

B/C/D are deliberately unavailable until independent review and separate explicit authorization. After that authorization, set the single `RUN_BCD` switch to `True`, enable Internet and the existing `HF_TOKEN` secret, and select two visible T4 GPUs. Run All then performs the CPU Gate first and continues into the bounded, resumable B/C/D run only if it passes. Each arm is limited to 200 updates; checkpoints are retained in Kaggle Output every 25 updates and at each evaluation barrier. The original A result is reused, and no Main or Confirmation data are accessed.

The Notebook does not need `KAGGLE_API_TOKEN` or `GITHUB_TOKEN`; it keeps run artifacts in Kaggle Output. The supplied Kaggle log verifies the CPU Gate for the prior `b7685aeb` source only. The review correction is verified locally on CPU; no B/C/D GPU run has been performed.

The 200-step cap is a logical per-arm training cap. Failed work since the last checkpoint may be repeated after interruption, so cumulative retry computation can exceed 200 optimizer calls. New manifests explicitly distinguish logical progress, unknown cumulative retry cost, requested GPU and observed GPU use.
