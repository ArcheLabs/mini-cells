# MiniCells Research Pipeline v2

## Execution contract

1. Define the experiment under `research/experiments/<id>/experiment.yaml`; numerical configurations live under `research/configs/`.
2. Register the hypothesis, matrix, quality gates, timing scopes and one-hour dual-T4 budget before a GPU run. 004 now has its own frozen development protocol and two-session matrix.
3. Actions prepares public pinned CPU assets with the existing TinyStories/train-only tokenizer implementation. Hash-named private Kaggle Dataset snapshots are reused without updating their contents.
4. Actions embeds the exact checked-out source into a seven-section notebook plus a compatible script for **bingjiangli/minicells**. It does not create a new notebook per experiment. The source bundle, configuration, protocol and inputs have separate hashes.
5. Kaggle runs offline with two independent T4 workers. It reads no HF_TOKEN/GITHUB_TOKEN, performs no clone, no download, and no GitHub mutation. The application budget is 3300s; server timeout is 3600s. Actions queue waiting is a separate CPU-side timeout.
6. Kaggle produces step-wise JSONL, aggregate metrics, routing/path evidence, frozen decision, nine plots, summary and a file-hash manifest. Required fields are explicitly missing/null when unavailable. Model workers stream periodic validation status while keeping full step logs on disk.
7. Actions polls/downloads the server-returned version, verifies identities and hashes, archives incomplete/negative evidence, uploads separate checkpoints, and proposes a result PR. It never overwrites observed runs.

## Actions setup

The existing Actions `KAGGLE_API_TOKEN` must belong to the account allowed to update `bingjiangli/minicells` and create its private input Datasets. Runtime Kaggle Secrets are no longer required. GitHub publication uses the workflow's built-in token with `contents: write` and `pull-requests: write`; repository settings must allow Actions to create PRs. If publication fails, retained Actions artifacts still contain the evidence.

Choose **Kaggle Experiment**: experiment 004 (default) or 003, seed group 1/2/3, `execute=false` to prepare; `execute=true` explicitly starts the GPU job. Preparation-only jobs produce the notebook and manifest, but do not upload a Dataset or consume GPU hours. Their CPU-prepared arrays are retained as an artifact for subsequent manual upload. The offline notebook needs the matching Dataset attached before manual execution.

001 and 002 remain historical notebooks/definitions at their existing paths. The ordinary Actions form offers the active 003/004 experiments. Historical entrypoints remain available for explicit reproduction. 004 executes the registered reference/graph sessions for development seeds 42, 43, 44. Select seed group 1/2/3 and reference/graph. Actions automatically pins and reuses source/data identities through a code-keyed input cache; no manual hashes are required. See its experiment README.

## Research authority and migration

`research/registry.yaml` is the active routing-study index. Existing `research/catalog.yaml` and all other research families remain intact. Old `experiments/registry.yaml` is retained as a historical index with a pointer to the canonical registry.

001 numerical configuration is migrated; its raw run evidence is not present in this repository checkout. Its old reported numbers are not synthesized into missing logs or routing data. 002 observed seed42 is not supported; its matrix remains incomplete. 003 seed42 retains DEVELOPMENT_INCOMPLETE. Original artifact/protocol paths are preserved; migrated views record every source file hash and leave absent evidence absent.

003's baseline time allocation is still the frozen 600s minus 180s reserve. This refactor does not retroactively fix the scientific protocol. Budget changes require a new registered identity before a future run.

## Log and plot semantics

Each training step records model (on aggregation), step, tokens, NLL/loss, training PPL, learning rate, compute-only tokens/s, allocated GPU bytes, route entropy, stage/temperature and Cell visits/dispatch scope. Validation PPL is plotted from held-out evaluation curves, never mislabeled as training PPL. Soft expected depth and actual executed visits are distinguished. Training global top-k dispatch is not a sequential path. Missing measurements are null or labeled unavailable.

Plots: loss, validation PPL, throughput, memory, routing entropy, Cell usage, path length, sampled paths and efficiency. Efficiency is a PPL-versus-observed-GEMM-FLOPs Pareto plot, not an invented capability score. Figures/summaries are derived and cannot upgrade frozen decisions.

## Acceptance and limits

The refactor supplies the registry, definitions, fixed input preparation, offline execution, structured logs, notebook preparation, plot/report generation and Actions collection. Migration creates inspectable historical views. GPU acceptance and external publication require an explicitly requested execution; no GPU smoke is part of this change. Do not describe the remote workflow as validated until a returned run passes identity checks.

004 is registered and unrun, using the 003 FFN Cell substrate and shared Attention. Its full development matrix requires six bounded sessions and five full-budget arms per seed. No empirical success is claimed. No HybridCLM regression is required for this routing pipeline change, per the user's scope.

## API provenance

The implementation was compared against the official Kaggle CLI source (`Kaggle/kaggle-cli`, version 2.2.4): kernels_push updates the existing id, kernels_status and kernels_output parse `owner/slug/version`, and dataset_status returns a lowercase status string. The HTTP failure type is requests.HTTPError with response.status_code. Do not substitute old 1.7.x clients that ignore the pushed version label. See https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py .
