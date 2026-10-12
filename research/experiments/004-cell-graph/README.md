# Experiment 004 — Learned Cell Graph

Status: **REGISTERED / UNRUN**. Development seeds **42, 43, 44**; scientific confirmation is **false**. No result is claimed before all five arms finish for every seed.

## Architecture and reused components

Reuse the pinned TinyStories corpus, tokenizer, seeded windows, 61M parameter causal Transformer, eight FFN Cells per layer, four-hop limit, optimizer, staged temperature schedule, evaluator, observed matmul counter and Pipeline v2. Fixed entry Cell 0 and absorbing STOP are retained. The document's 100M example is not a new capacity target. A full attention block inside every Cell would change the 003 substrate and is outside this comparison.

003 already has contextual **internal** transition heads. 004 adds a persistent learned `8 x 9` prior per layer:

`p(next | state, Cell i) = softmax((head_i(updated_state) + T[i, :]) / temperature)`.

The prior contains eight destinations plus STOP. It is not an external global router. Softmax produces dense positive probabilities; empirical edge support and thresholded prior support are reported separately, not claimed as exact structural sparsity.

## Training

- First 25%: soft conditional-mean relaxation, temperature 2.
- Next 25%: geometric anneal 2 → 0.1, soft forward.
- Last 50%: selected hard forward with renormalized top-2 backup-gradient mixture. The chosen Cell and an alternative are evaluated at the same incoming state; STOP is identity. Top-2 probabilities are floored at 0.05 before renormalization so the backup compute receives a positive gradient even at low temperature. Backward uses the mixture; forward equals the chosen branch. Cells outside top-2 have no counterfactual gradient. This is a biased surrogate.

Hard inference disables backups. Training Cell visits and throughput are recorded separately; inference FLOP savings must never imply equal training savings. Comparison with 003 is a joint prior + gradient change; it does not isolate topology's causal effect.

## Manual Kaggle notebook

Use the single canonical entrypoint at [`research/notebooks/03-routing-and-growth/experiment-004-learned-cell-graph-kaggle.ipynb`](../../notebooks/03-routing-and-growth/experiment-004-learned-cell-graph-kaggle.ipynb) in the existing Kaggle notebook [bingjiangli/minicells](https://www.kaggle.com/code/bingjiangli/minicells). Copy/import its cells into that notebook; do not create a new Kaggle notebook for each experiment. The workflow does not call Kaggle notebook-push or Dataset-upload APIs.

Each run selects one `SEED` (42, 43, or 44) and one `SESSION` (`reference` or `graph`). Run all cells. For the first session, `SOURCE_COMMIT` may be `main`; copy the printed full SHA into the cell for the paired session and keep that SHA for all six sessions. The notebook reuses the registered CPU data-preparation, offline dual-T4 runtime, evidence collector, and existing Kaggle Secret GitHub publisher. It checks GitHub write access before GPU work and pushes verified results to `results/cell-graph-<source-sha>` with a normal append-only push.

The notebook reads `GITHUB_TOKEN` through Kaggle's existing `UserSecretsClient` path. `HF_TOKEN` is optional for this public, pinned dataset. Setup is limited to 300 seconds and fails before GPU work if setup plus the 3,300-second application cap would exceed the one-hour Kaggle session. A failed GitHub push can be retried from the notebook's final cell without rerunning training.

For **each seed 42, 43, 44**, run these two sessions from the **same source commit** and hash-verified prepared inputs:

| Session input | GPU queue A | GPU queue B | Maximum worker budgets |
|---|---|---|---|
| reference | soft-cell, then fixed-cell | global-router | 1850 + 850 / 850 seconds |
| graph | graph-cell | hard-cell (003 STE) | 3000 / 1850 seconds |

Physical GPU placement alternates by seed. There is no DDP. Each arm gets the full **10M token** target; baseline time caps rise from 600 to 850 seconds so the old 420-second training cutoff is not reused. Each arm reserves 180 seconds for checkpoint/evaluation. Application cap stays **3300 seconds**, server cap **3600 seconds**. Budget feasibility is a cap calculation, not a guarantee that graph will finish on T4. A slow arm remains incomplete; the token budget and cap are not relaxed after observing it.

Six sessions complete the development matrix. Worst-case application wall time is 5.5 hours total; conservative dual-GPU accounting is 11 GPU-hours if Kaggle bills each GPU separately. Actual quota accounting follows Kaggle. No GPU smoke is part of this implementation.

## Frozen development criteria

All five full-budget arms, identical validation population, capacity within 1%, source/config/protocol/input identity must exist before any gate is evaluated. Each seed must independently satisfy:

1. Mean per-layer full-path entropy `graph < hard`.
2. Graph validation PPL `<= 13`.
3. Observed graph forward GEMM FLOPs/token `< 50%` of **fixed-cell** (not the more expensive soft-cell baseline).
4. At least two distinct hard paths and two utilized non-entry Cells in every graph layer.

Per-layer full validation populations, learned STOP vs forced truncation, empirical edge counts, transition entropy, path diversity and plug-in token/Cell MI are preserved. Fixed entry is excluded from MI. MI is descriptive and biased at finite sample size; it is not proof of semantic specialization. Additional prior, path and efficiency plots are derived views.

## Analyze after both sessions are archived

```
PYTHONPATH=src:. python -m minicells.research.graph_report \
  --archive research/artifacts/004-cell-graph/<source-commit> \
  --output research/artifacts/004-cell-graph/<source-commit>/matrix-analysis
```

Collection updates this derived matrix automatically. Missing sessions or failed checks produce `DEVELOPMENT_INCOMPLETE`. All three supported paired development seeds yield `DEVELOPMENT_SUPPORTED`, never a formal confirmation. Historical 001–003 protocols and decisions retain their authority; the new hard baseline is a registered reproduction under 004, not a rewrite of 003.


## Recover an archived session after the old setup-timeout error

`DEVELOPMENT_SESSION_COMPLETED` means the selected reference or graph session completed. `scientific_decision: false` and `gates: null` are expected until both sessions for all three seeds exist. An archive path confirms local collection; a successful final push is needed for GitHub publication.

If the old notebook printed its archive path and then raised `Notebook setup exceeded 300 seconds`, keep the Kaggle kernel and working directory. In a new cell run:

```python
from functools import partial
from publish_experiment_results import run_git as existing_git
run_git = partial(existing_git, ROOT)
ALREADY_ARCHIVED = ARCHIVE_TARGET.exists()
```

Then rerun only the archive/commit cell and the final push cell. This reuses the historical Git helper, verifies the existing archive and publishes it without rerunning setup, data preparation or training. A push failure leaves the local commit available; retry only the final push cell. For the observed seed 42 reference run, preserve source `7c1d8b23adaf047195d38a6827f46f3e97ae1faa` and branch `results/cell-graph-7c1d8b23adaf` for its remaining paired matrix sessions.

The corrected notebook limits `run_setup_git` to setup. Publication imports `publish_experiment_results.run_git`; retrying the archive/commit cell checks the directory on disk instead of the pre-training `ALREADY_ARCHIVED` snapshot. The registered GPU queues remain part of the frozen protocol for this existing cohort. Changing their allocation requires a separate registered revision; archived observations must retain their original source and protocol identity.
