# Core Validation 007 — Functional Boundary Discovery

Status: **CONFIRMATION_V2_PROTOCOL_FROZEN_UNRUN**

## Frozen discovery result

Discovery is complete and immutable.

- discovery seeds: `80701 / 80702`
- frozen winner: `interference_cut`
- mean interference-cut fraction: `0.5508129572944579`
- mean z-only/oracle routing agreement: `0.8973214285714286`
- mean partition balance: `0.9375`
- mean frozen selection score: `0.7127685550833804`
- mean soft-Top2 oracle coverage: `0.9665178571428572`

Discovery is mechanism selection only; it is not a supported/not-supported scientific confirmation.

## Why the first confirmation set is retired

The original runner opened `80711/80712/80713` in one long-lived Python/CUDA process and wrote `raw.json` only after all seeds completed. In the first attempt:

- `80711` completed and was observed;
- `80712` started, then the process terminated before a per-seed checkpoint existed;
- `80713` was not opened.

Observed non-canonical `80711` console summary:

```text
pass=False
winner=interference_cut
split=0.163
spawn=2.094
reg/unsafe=0.152
gain/replay=1.257
route=0.411
```

The exact `80712` termination cause cannot be proven from the surviving notebook output. The infrastructure defect is unambiguous: a completed seed was not durably checkpointed before the next seed began.

Because part of the original set was observed, `80711/80712/80713` are retired and must not be rerun as "untouched confirmation".

The audit is frozen in `confirmation-protocol-v1.1.json`.

## Confirmation v1.1 amendment

Untouched confirmation seeds are now:

```text
80721
80722
80723
```

The amendment changes infrastructure only. It does **not** change:

- Pythia model or SlimPajama data;
- the pinned data manifest;
- discovery winner `interference_cut`;
- functional-mode / mitosis mechanism;
- Core 006 baselines;
- any confirmation gate threshold.

The original `protocol.json` and `winner-lock.json` remain frozen. `confirmation-protocol-v1.1.json` pins their identities and the expected data manifest.

## Decision question

Core 006 showed that frozen real Pythia representations contain substantial low-dimensional functional reuse and that bounded replay-free certificates reduce registered forgetting, but semantic/address-based mitosis did not provide a useful functional split boundary.

Core 007 asks:

> Can bounded write-demand/interference geometry provide a better mitosis boundary than semantic address identity, and can that functional identity be recovered from inference-visible representations after the split?

For frozen hidden state `h`:

\[
z=U^Th
\]

and frozen-foundation projected loss signal:

\[
u_t=U^T\frac{\partial L}{\partial h_t},
\]

the write-demand signature is:

\[
G=\frac1T\sum_tu_tz_t^T.
\]

Discovery compared:

1. `semantic_singleton` — Core 006 control;
2. `activation_community` — activation-subspace geometry;
3. `write_community` — write-demand geometry;
4. `interference_cut` — direct symmetric cross-write damage with deterministic balanced cut.

The frozen winner is `interference_cut`.

## Frozen confirmation gates

Every amended confirmation seed must independently satisfy the existing gates:

- registered regression <= `0.50x` unsafe;
- new-learning gain >= `0.80x` replay;
- candidate gain > no-growth certificate baseline;
- median split-conflict reduction >= `0.30` and greater than the same-seed Core 006 semantic split;
- spawned Cells <= `0.50 * 32 addresses`;
- at least four later transactions reuse children;
- oracle/deploy z-mode agreement >= `0.70`;
- deploy heldout NLL within `2%` of oracle functional routing;
- at least one non-zero heldout causal-ablation signal.

No gate was changed after the retired attempt.

Only after all `80721/80722/80723` checkpoints complete with matching identities may the reporter emit:

```text
FUNCTIONAL_BOUNDARY_MECHANISM_SUPPORTED
FUNCTIONAL_BOUNDARY_MECHANISM_NOT_SUPPORTED
```

Until then the only allowed status is:

```text
CONFIRMATION_INCOMPLETE
```

with `scientific_decision=false`.

## Final resumable infrastructure

The canonical confirmation path is:

```text
orchestrate_core_validation_007_confirmation.py
    -> run_core_validation_007_confirmation_seed.py  (fresh process / one seed)
    -> report_core_validation_007_confirmation.py
    -> publish_core_validation_007.py
```

The seed runner:

- accepts one amended seed;
- verifies protocol, amendment, winner, data manifest and scientific-code identity;
- atomically writes `confirmation/seeds/seed-<seed>.json` immediately after success;
- writes a structured failure record for catchable exceptions;
- skips a matching completed checkpoint;
- refuses to overwrite a checkpoint from another scientific identity.

The orchestrator:

- validates CUDA and GitHub write access **before** opening a new seed;
- runs every seed in a fresh Python/CUDA child process;
- captures per-seed stdout/stderr logs;
- records process-level failures such as SIGKILL/OOM when Python cannot write its own traceback;
- regenerates the partial/final report after every seed;
- commits and pushes canonical partial artifacts after every seed;
- stops after failure without losing completed checkpoints;
- hydrates pushed checkpoints in a fresh Kaggle session and resumes from the first unfinished seed.

## Publication path

Core 007 now reuses the same authentication mechanism as the repository's established Kaggle publisher (`scripts/research/publish_experiment_results.py`):

- Kaggle Secret name: `GITHUB_TOKEN`;
- token loaded directly through `kaggle_secrets.UserSecretsClient` when not already in the environment;
- temporary `GIT_ASKPASS` supplies `x-access-token` / token credentials;
- `GIT_TERMINAL_PROMPT=0`;
- an authenticated `git push --dry-run` is performed before GPU work;
- the token is not persisted in the Git remote or repository config.

This replaces the ad-hoc `GH_TOKEN` path used by the first Core 007 notebook.

## Kaggle execution — one cell

Use:

```text
research/notebooks/04-continual-learning-core/
core-validation-007-functional-boundary-discovery.ipynb
```

Before running it, configure:

```text
GITHUB_TOKEN
```

in Kaggle Secrets with Contents read/write permission for `ArcheLabs/mini-cells`. `HF_TOKEN` is optional.

The notebook contains one execution cell. It fresh-clones the latest research branch, installs dependencies and runs:

```bash
python scripts/research/orchestrate_core_validation_007_confirmation.py \
  --branch codex/core-validation-007-functional-boundary-discovery \
  --secret-name GITHUB_TOKEN \
  --push-results
```

No separate token-loading or publish cell is required. Re-running the same notebook after interruption is the intended resume procedure.

## Interpretation boundary

A positive result would establish only that, under frozen Pythia representations and a frozen coarse router, bounded functional-interference geometry can provide a better mitosis boundary and that the resulting identity is sufficiently predictable from inference-visible `z`.

It would not establish safe nonlinear foundation updates, learned global-router drift, reconstruction of certificates for opaque historical checkpoints, or full-scale autonomous CLM growth.
