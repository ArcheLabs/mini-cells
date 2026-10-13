# Experiment 005 — Learned Cell Connections and Compositional Efficiency

**Status: planned implementation / protocol proposal, not registered, not run.**

This experiment asks whether learned sparse predecessor-to-successor constraints help small FFN Cells compose more efficiently than fixed random connections and a state-aware global sequential router. It separately evaluates sequential composition (`S0` vs `P0`), learned connections (`G0` vs `R0` and `S0`), and the system comparison against native Top-2 MoE (`G0` vs `M0`). A favorable result on one question cannot stand in for another.

The proposal is preserved in [PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md). Machine-readable proposed values are in [config/proposed.json](config/proposed.json) and [protocol.json](protocol.json). They remain explicitly unregistered: audit-fixed initialization, auxiliary-loss and grouped measurement proposals are recorded; asset identities, seed matrix and remaining refresh settings still require preregistration. **No full training command will run until those items are frozen and the registry status changes.**

## Project-owned implementation

- [implementation/exp005](implementation/exp005/README.md) contains the new model, graph, accounting, profiling, training and decision code. It reuses the HybridCLM project's shared `FeedForward`, `GlobalRouter`, data-window and JSON-writing helpers.
- [notebook.ipynb](notebook.ipynb) is a thin entrypoint for the existing `bingjiangli/minicells` notebook. It runs CPU preflight/dry-run now; its training path stays blocked while this protocol is planned.
- [manual-execution.md](manual-execution.md) describes the permitted workflow and the gates required before training.
- [analysis.md](analysis.md) defines the evidence boundaries and reporting checklist. The analysis CLI returns `PROTOCOL_NOT_REGISTERED` until the protocol is frozen; `--output` writes a content-addressed report that can be appended with the same publisher.
- `scripts/research/run_experiment_005.py`, `scripts/research/analyze_experiment_005.py`, and `scripts/research/publish_experiment_005.py` are thin repository entrypoints. The publisher reuses the established Kaggle Git askpass and GitHub write-preflight helpers.

The old `kaggle/experiment-005-results` Consumer Language Bridge publisher is a separate historical study. This experiment uses the unique slug `005-compositional-cell-graph` and never writes to that old result path or branch.

See [AUDIT_REPAIR.md](AUDIT_REPAIR.md) for each repaired finding, verification and the final review/registration procedure.

## Current engineering status

The CPU preflight covers parameter parity, candidate masking, four-step execution, hard-forward/top-2-backup equivalence, active-weight and observed-GEMM accounting, and full-forward/KV-cache decode parity. Its synthetic fixture checks noncommuting Cell order and deterministic edge-refresh bookkeeping; it is **not** the proposal's trained Gate 1 comparison between learned and random adjacency. No GPU training or scientific gate has run.

## Run

From the repository root:

```bash
python scripts/research/run_experiment_005.py --help
python scripts/research/run_experiment_005.py --mode preflight
python scripts/research/run_experiment_005.py --mode dry-run
```

These commands do not start GPU training. They validate the proposed contract, report blockers, and show the planned matrix.

After an approved protocol is registered, use `--mode train` for one registered arm and `scripts/research/publish_experiment_005.py` to append its evidence. The existing Kaggle notebook runs GitHub write preflight before it prepares assets or starts GPU work. Each arm archive is immutable and source-specific; checkpoints stay outside Git.
