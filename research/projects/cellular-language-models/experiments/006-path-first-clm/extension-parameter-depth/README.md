## Performance repair — full retraining remains blocked

Use [performance-notebook.ipynb](performance-notebook.ipynb) for the bounded synthetic Kaggle benchmark and checkpoint rehearsal. See [PERFORMANCE_REPAIR.md](PERFORMANCE_REPAIR.md). This repair does not approve a 5,000-step run; the previous launcher source and checkpoints are historical evidence.

# PD-001 — Parameter × adaptive path depth

Execution: **automatic development**. Click Run All once; assets, hashes, seed and execution identity are generated internally. The historical formal proposal remains unregistered; development execution does not require manual registration. **NO_SCIENTIFIC_RESULT_YET** until a real run supplies evidence.

This additive extension tests whether a smaller jointly trained hard-path model can approach an independently trained standard MoE. It does not reinterpret original 006 or CLM Native results. Original 006 remains unchanged. Following the user-requested workflow simplification, PD development does not require parent/PD formal registration; independent confirmation remains isolated. No Cell-only or Router-only pretraining exists.

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

Import [notebook.ipynb](notebook.ipynb) into the existing `bingjiangli/minicells` editor and click Run All. The program automatically prepares pinned training assets, checks duplicates, trains the tokenizer, records hashes/config/source and selects a seed. It checks the existing GITHUB_TOKEN before training, runs the four core development arms with independent workers, then profiles and analyzes serially and returns small metadata to GitHub. HF_TOKEN is optional. No manual hash fields, mode switch or registration form is required. Repeating Run All resumes the same source-bound run; complete checkpoints are reused. Confirmation is not part of this flow.

The declared proposal budget (10M target tokens per core arm) is retained. A development run records the first-occurrence story-selection policy and its actual row provenance; it is not presented as a formally registered raw-row study. GPU work occurs only when the user launches the Notebook, not during implementation tests.

## Model and training

Each live token executes exactly one selected operation per decision: Cell, CTX_READ or STOP. The same Cell can execute consecutively without an artificial repeat cap. STOP performs no Cell/read; upper-bound termination emits no fake STOP probability. Padding is -1 and stopped positions receive no more updates. All hidden-state dependencies remain differentiable across 64 real steps.

J0 uses terminal CE plus whole-path actor/critic. J1 independently samples two complete trajectories at unchanged weights and uses detached leave-one-out terminal CE+cost returns. The task gradients and actor gradients average over trajectories and target tokens, never divide by path length. The same optimizer updates Cells, Connection, memory, STOP and Head; critic inputs are detached. The critic remains an auxiliary trained diagnostic in J1, not its actor baseline. Repeated executions and both rollouts are charged.

M0 directly reuses `exp006.baseline.MoELM`, including causal attention and native Top-2 load balancing. At the 25% tier, width changes from 384 to 256: this is an explicit representation-capacity confound. The constant residual scale 0.125 and 64-entry step capacity are shared across PD arms, avoiding an implicit alpha change between F16/F32.

## Automatic identity and reusable entries

`--mode auto --devices auto --output RUN` performs the entire development flow. Its execution config, source and asset identities are written into RUN, and worker/evaluation/publication entry points check these records automatically. An incomplete/corrupt bundle fails; the program does not fabricate successful audits. Direct `train/evaluate/profile/publish` without a prepared identity still fail clearly.

`pd001.orchestration.matrix` starts independent per-device queues, resumes checkpoints and returns only after all workers finish. Profiling is serial afterward. Publication reuses the historical Askpass publisher from current main, staging result metadata only. Weights and source corpus stay in Kaggle output; chunked logs avoid oversized blobs.

Independent confirmation retains its separate data/approval/checkpoint isolation. The automatic development branch explicitly rejects every confirmation-role entry. See the repository [workflow policy](../../../../../governance/automatic-development-workflow.md) and root AGENTS.md for future experiments.

## Evidence limits

Inference counters charge selected module rows, repeated affine calls, edge-prior products, dense masked QK/AV and the tied output head. The training dispatcher observes mm/addmm/bmm in forward and backward. Full non-GEMM and fused SDPA backward accounting remain **INCOMPLETE_ACCOUNTING**, not zero. Exact total-FLOPs matching cannot be claimed until this is completed/reviewed. Hardware profiles distinguish warm prefix cache and first-invocation cold model; the latter is not an OS hardware-cache flush.

Interventions separate downstream policy replanning from fixed-action replay. Statistics use paired whole-story bootstrap and a one-sided upper bound; single-seed diagnostics cannot establish H1. MoE baseline stability, seed consistency and statistical power are registration requirements.

See [SPEC_PD1.md](SPEC_PD1.md), [registration](registration/README.md), [parameter inventory](reports/parameters.json), and [engineering report](reports/ENGINEERING.md).
