# PD-001 — Parameter × adaptive path depth

Status: **PROPOSED_NOT_REGISTERED**. Engineering fixtures only. **NO_SCIENTIFIC_RESULT_YET**.

This additive extension tests whether a smaller jointly trained hard-path model can approach an independently trained standard MoE. It does not reinterpret original 006 or CLM Native results. The original 006 gate remains in force; both the parent gate review and PD registration are required. No Cell-only or Router-only pretraining exists.

## CPU engineering entry

From the repository root:

```bash
python research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/run_pd001.py --mode audit --output /tmp/pd-audit
python research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/run_pd001.py --mode params --all-arms --output /tmp/pd-params
python research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/run_pd001.py --mode dry-run --arm P50-A32 --output /tmp/pd-dry
python research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/run_pd001.py --mode preflight --output /tmp/pd-gate0
python -m pytest -q tests/research/path_first_pd001
ruff check --config research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/ruff.toml research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/implementation research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/run_pd001.py tests/research/path_first_pd001
```

Use fresh output directories; evidence cannot be silently overwritten. Gate 0 constructs the actual model inventories and executes all ten full-size joint CPU synthetic steps, plus tiny mechanism fixtures. Seed 17 and artificial token IDs are engineering fixtures only, not consumed research seeds.

Import [notebook.ipynb](notebook.ipynb) into the existing `bingjiangli/minicells` editor. Run all cells. Default mode is `auto`: CPU preflight, then registration/authorization checks, then the registered core experiment matrix. A failed check exits with an error. The current unregistered proposal stops before training; there is no manual switch between two phases. No approval or assets are fabricated, and no secrets are read.

## Model and training

Each live token executes exactly one selected operation per decision: Cell, CTX_READ or STOP. The same Cell can execute consecutively without an artificial repeat cap. STOP performs no Cell/read; upper-bound termination emits no fake STOP probability. Padding is -1 and stopped positions receive no more updates. All hidden-state dependencies remain differentiable across 64 real steps.

J0 uses terminal CE plus whole-path actor/critic. J1 independently samples two complete trajectories at unchanged weights and uses detached leave-one-out terminal CE+cost returns. The task gradients and actor gradients average over trajectories and target tokens, never divide by path length. The same optimizer updates Cells, Connection, memory, STOP and Head; critic inputs are detached. The critic remains an auxiliary trained diagnostic in J1, not its actor baseline. Repeated executions and both rollouts are charged.

M0 directly reuses `exp006.baseline.MoELM`, including causal attention and native Top-2 load balancing. At the 25% tier, width changes from 384 to 256: this is an explicit representation-capacity confound. The constant residual scale 0.125 and 64-entry step capacity are shared across PD arms, avoiding an implicit alpha change between F16/F32.

## Registered operations (currently blocked)

`train`, `evaluate`, `profile`, `intervene`, `confirmation` and `publish` refuse before asset/checkpoint/secret access. After explicit approved registration, use the same entry with `--config REGISTERED_JSON --source-commit FULL_SHA --arm ARM --seed REGISTERED_SEED --role development --assets BUNDLE --output RUN`; `--run RUN` selects final checkpoint evaluation/profile/interventions. Confirmation training requires `--confirmation-assets SEALED_BUNDLE`, loaded only after the final checkpoint hash is fixed. No Python fixture helper is a GPU authorization route.

`pd001.orchestration.matrix` supports independent per-device queues and returns only after all workers finish; profiling is a separate serial phase. There is no autonomous GPU launch. Publication reuses the historical publisher from an isolated current-main checkout, staging metadata only, not historical workflow commits or weights.

## Evidence limits

Inference counters charge selected module rows, repeated affine calls, edge-prior products, dense masked QK/AV and the tied output head. The training dispatcher observes mm/addmm/bmm in forward and backward. Full non-GEMM and fused SDPA backward accounting remain **INCOMPLETE_ACCOUNTING**, not zero. Exact total-FLOPs matching cannot be claimed until this is completed/reviewed. Hardware profiles distinguish warm prefix cache and first-invocation cold model; the latter is not an OS hardware-cache flush.

Interventions separate downstream policy replanning from fixed-action replay. Statistics use paired whole-story bootstrap and a one-sided upper bound; single-seed diagnostics cannot establish H1. MoE baseline stability, seed consistency and statistical power are registration requirements.

See [SPEC_PD1.md](SPEC_PD1.md), [registration](registration/README.md), [parameter inventory](reports/parameters.json), and [engineering report](reports/ENGINEERING.md).
