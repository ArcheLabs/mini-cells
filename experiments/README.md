# MiniCells experiments

This directory is the research layer for experiments around routing, Cell
architectures, and continual learning. It is kept separate from the public
`src/minicells/` API and does not change HybridCLM behavior.

## First milestone: Kaggle pipeline smoke run

The initial workflow checks the GitHub Actions to Kaggle path and GPU output
collection. It runs `kaggle/run_experiment.py`, which allocates a CUDA device,
executes a small matrix operation, and writes `experiment.json` and
`metrics.json`. This is infrastructure evidence only; it does not train a
language model or support an architecture claim.

The workflow runs on pushes to `main` that change this directory or `kaggle/`,
and can also be started with **Actions → Kaggle Experiment → Run workflow**.
Each successful run uploads the Kaggle output and GitHub run context as an
Actions artifact named `kaggle-experiment-<run id>`.

## Configure GitHub Actions

In **Settings → Secrets and variables → Actions**:

1. Add the secret `KAGGLE_API_TOKEN` with an API token from your Kaggle account.
2. Add the repository variable `KAGGLE_USERNAME` with the account slug that
   owns the kernel.

The kernel is private by default. The first successful push creates the kernel
`<KAGGLE_USERNAME>/minicells-pipeline-smoke`; later workflow runs update it.
Your Kaggle account must have GPU access and available accelerator quota.

Kaggle credentials belong in Actions secrets or local user configuration only.
Do not commit `kaggle.json`, API tokens, or `.env` files.

## Local use

The Kaggle CLI reads `KAGGLE_API_TOKEN` directly for non-interactive
authentication. To push manually, set that variable in your shell, replace the
owner in `kaggle/kernel-metadata.json` with your Kaggle username, then run:

```sh
python -m pip install kaggle
kaggle kernels push --path kaggle --timeout 600
```

The first milestone intentionally contains no model trainer, dataset, router,
or checkpoint. Those belong to later experiment milestones with their own
configs and evaluation protocol.
