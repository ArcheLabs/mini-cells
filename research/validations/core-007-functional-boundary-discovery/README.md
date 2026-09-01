# Core Validation 007 — Functional Boundary Discovery

Status: **CONFIRMATION_V2_PROTOCOL_FROZEN_UNRUN**

## Current scientific state

Core 007 discovery is complete and immutable.

Frozen discovery seeds:

```text
80701
80702
```

Frozen winner:

```text
interference_cut
```

The committed `winner-lock.json` records:

- mean interference-cut fraction: `0.5508129572944579`;
- mean z-only/oracle routing agreement: `0.8973214285714286`;
- mean partition balance: `0.9375`;
- mean frozen selection score: `0.7127685550833804`;
- mean soft-Top2 oracle coverage: `0.9665178571428572`.

Discovery is mechanism selection only and is not a scientific supported/not-supported decision.

## Why confirmation v1 was retired

The original confirmation infrastructure used seeds `80711/80712/80713` in one long-lived Python/CUDA process. It accumulated all seed results in memory and wrote `raw.json` only after every seed completed.

During the first real run:

- `80711` completed and its console summary was observed;
- `80712` started, then the child process terminated before the runner produced a Python traceback or any per-seed checkpoint;
- `80713` was never opened.

Observed non-canonical `80711` summary:

```text
pass=False
winner=interference_cut
split=0.163
spawn=2.094
reg/unsafe=0.152
gain/replay=1.257
route=0.411
```

Because `80711` was observed and `80712` was partially opened, the complete original confirmation seed set is retired. It must not be rerun and described as untouched confirmation.

The repository cannot prove the exact cause of the `80712` process termination from the notebook output alone. The absence of a child Python traceback is consistent with a process-level termination such as CUDA/host OOM, but this is not recorded as a proven root cause. The infrastructure defect itself is proven: completed seeds were not checkpointed before the next seed began.

The audit is frozen in `confirmation-protocol-v1.1.json`.

## Confirmation v1.1 amendment

The amended untouched confirmation seeds are:

```text
80721
80722
80723
```

The amendment changes **infrastructure only**. It does not change:

- frozen model or dataset;
- data selection or manifest identity;
- the discovery winner (`interference_cut`);
- functional-mode or mitosis mechanism;
- Core 006 baselines;
- any confirmation gate threshold.

The original `protocol.json` and `winner-lock.json` remain the discovery lock. `confirmation-protocol-v1.1.json` references their exact hash and pins the known real-data manifest.

## Decision question

Core 006 established that real frozen Pythia representations contain substantial low-dimensional functional reuse and that bounded replay-free certificates reduce registered forgetting. It failed because semantic/address-based mitosis did not create a useful functional separation.

Core 007 asks:

> **Can bounded write-demand/interference geometry provide a better mitosis boundary than semantic address identity, and can that functional identity be recovered from inference-visible representations after the split?**

## Frozen mechanism

For frozen hidden state `h`:

\[
z=U^Th\in\mathbb R^{64}.
\]

Core 007 also uses the frozen-foundation projected loss signal

\[
u_t=U^T\frac{\partial L}{\partial h_t}
\]

and sequence write demand

\[
G=\frac1T\sum_t u_tz_t^T.
\]

Each semantic address may contain at most four bounded functional modes. A mode retains dependency counts, a `z` prototype, `z` second moment, running mean write matrix, and a rank-bounded write basis. Raw historical text is not learner state.

The discovery candidates were:

1. `semantic_singleton` — Core 006 semantic control;
2. `activation_community` — activation-subspace geometry;
3. `write_community` — write-demand geometry;
4. `interference_cut` — direct symmetric cross-write damage with deterministic balanced max-cut.

The frozen winner is `interference_cut`.

The global K-means address router remains frozen. The confirmation still distinguishes:

- oracle functional identity from frozen-foundation write demand;
- deployable local routing from inference-visible `z` prototypes;
- soft Top-2 routing as a diagnostic only.

## Frozen confirmation gates

Every amended confirmation seed must independently satisfy the original gates:

- registered regression <= `0.50x` unsafe;
- new-learning gain >= `0.80x` replay;
- functional-mitosis gain > no-growth certificate baseline;
- median split-conflict reduction >= `0.30` and greater than the same-seed Core 006 semantic split;
- spawned Cells <= `0.50 * 32 addresses`;
- at least four later transactions reuse children;
- oracle/deploy z-mode agreement >= `0.70`;
- deploy heldout NLL within `2%` of oracle functional routing;
- at least one non-zero heldout causal-ablation signal.

No threshold was changed after observing the retired attempt.

Only when all `80721/80722/80723` checkpoints are complete and identity-matched may the reporter emit either:

```text
FUNCTIONAL_BOUNDARY_MECHANISM_SUPPORTED
FUNCTIONAL_BOUNDARY_MECHANISM_NOT_SUPPORTED
```

Otherwise the only allowed confirmation status is:

```text
CONFIRMATION_INCOMPLETE
```

with `scientific_decision=false`.

## Resumable execution infrastructure

The amended confirmation is intentionally seed-isolated.

`scripts/research/run_core_validation_007_confirmation_seed.py`:

- accepts exactly one amended confirmation seed;
- starts from a fresh Python/CUDA process;
- verifies base protocol, amendment, winner and manifest identity;
- computes a hash over the scientific implementation files;
- atomically writes `confirmation/seeds/seed-<seed>.json` on success;
- writes a structured Python failure record on catchable exceptions;
- refuses to overwrite a checkpoint produced by another scientific identity;
- skips an already-complete matching seed.

`scripts/research/orchestrate_core_validation_007_confirmation.py`:

- runs preflight before GPU work;
- hydrates previously pushed partial seed checkpoints from canonical artifacts after a new Kaggle session starts;
- launches each seed in a separate child process;
- tees child stdout/stderr into a per-seed log;
- records process-level failure metadata if a child dies without a Python failure record;
- regenerates the aggregate report after every seed;
- publishes/commits/pushes partial canonical artifacts after every seed;
- stops after a failed seed without destroying completed checkpoints;
- resumes by skipping completed matching seeds.

`scripts/research/preflight_core_validation_007.py` validates, before GPU work:

- branch identity;
- clean tracked tree;
- discovery protocol and winner-lock hashes;
- amended confirmation identity;
- CUDA availability;
- GitHub push credentials using a `git push --dry-run`.

## Kaggle execution

Use only:

```text
research/notebooks/04-continual-learning-core/
core-validation-007-functional-boundary-discovery.ipynb
```

The notebook no longer exposes discovery or retired confirmation cells. It loads `GH_TOKEN` or `GITHUB_TOKEN` from Kaggle Secrets, optionally loads `HF_TOKEN`, runs preflight, and then runs:

```bash
python scripts/research/orchestrate_core_validation_007_confirmation.py \
  --branch codex/core-validation-007-functional-boundary-discovery \
  --push-results
```

The same command is safe to rerun after interruption. Completed seed checkpoints are recovered from the repository and skipped.

## Interpretation boundary

A positive Core 007 result would establish only that, under a frozen real Pythia representation and frozen coarse router, a bounded functional-interference geometry can generate a better mitosis boundary and that this identity is sufficiently predictable from inference-visible `z`.

It would not establish safe nonlinear foundation updates, learned global router drift, reconstruction of certificates for opaque historical checkpoints, or full-scale autonomous CLM growth.
