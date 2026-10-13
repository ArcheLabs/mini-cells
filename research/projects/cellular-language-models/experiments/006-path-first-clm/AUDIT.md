# Read-only historical audit

Base: `codex/research-project-first-layout`,
`0c1886052973547be1a8beded4c44c9dac1538c2` (2026-10-13).
User specification SHA256:
`7083993b084111ccdd928ee5789e21f05f6bce35beb31b1b2aae72354627f095`.

## What was checked

| Historical line | Implementation / primary record | Supported boundary and lesson |
|---|---|---|
| Native M1 | `src/minicells/native_clm_v0.py`; `research/projects/cellular-language-models/context/06-native-clm/hypothesis/M1_CLOSURE.md` | 12,154,368 parameters, eight Cells/top-2, next-token training. Six fixed Transformer blocks and a parallel cellular layer do not establish path-first computation. |
| Constructive 004 | `src/minicells/constructive_clm_004.py`; `research/projects/cellular-language-models/experiments/constructive-clm-004-model-level-multicell-computation/validation/` | Controlled residual operators support simultaneous/sequential composition and order recovery. This is not a language-level path-learning result. |
| Constructive 005 | `src/minicells/constructive_clm_005.py`; corresponding `research/projects/cellular-language-models/experiments/constructive-clm-005-scaffold-removal/validation/` | Registered learned routing/write/growth control passes in its controlled world. Protected geometry and operator family remain supplied. |
| TextNCA | `src/minicells/sparse_cellular_textnca.py`, `src/minicells/overcomplete_cellular_textnca.py`; archived routing protocols/results and specification evidence | Recurrent state and conditional computation are reusable ideas. Sparse quality and dense scaffold handoff failures warn against substituting local fit for whole-trajectory quality. |
| Native M2/M3 | `research/governance/audits/CLM_CAPABILITY_CEILING.md`, `RESEARCH_LEDGER.md` | Retention/growth boundaries remain not supported; optimizer transaction mechanics do not establish safe continual language learning. |
| Hybrid 005 | `implementation/exp005/models.py`, `routing.py`, `graph.py`, `training.py`, `profiling.py`, `checkpoint.py`, `publish.py` under 005 | G0 uses four local hops inside fixed attention layers, constrained candidates, backup gradients and external probe refreshes. These mechanisms are deliberately absent from 006. |

005 published development evidence was read from result commit
`96bedff200736767091376082ced1b335a1552c6`, source
`7b5affd42efe9d341a9c2cbf73511f67adcedd38`, seed `50713001`:
M0/P0 final development PPL 10.12889/9.96952; evaluation GEMM per token
47,722,496/35,287,040; isolated cached FP32 decode median 9.41335/45.13193 ms.
Lower arithmetic did not deliver lower measured latency. G0 early PPL values in
user logs are incomplete development observations, not completed confirmation.
TextNCA primary decisions were also read directly: routing Validation 001 reports
`CLM_PROGRAM_SPARSITY_QUALITY_FAILURE` (quality ratios 2.264–2.365); 001b reports
`CLM_NO_QUALITY_SAFE_SPARSITY` with quality-safe k=8 and zero successful replicates;
v2 closed-loop 001b reports `CLMV2_CLOSED_LOOP_HANDOFF_FAILURE`. Sources are the
respective `research/projects/cellular-computing/experiments/*/artifacts/decision.json`.
The local imitation diagnostic (MSE about .052–.055, cosine .970–.972) coexists with
roughly 1.12x direct-handoff PPL. These are precise historical failure boundaries.

006 retrains its own M0; it never substitutes those old numbers for a control.

The unrelated existing local 005 diagnosis report is left untouched and excluded
from the 006 commit. Historical protocols, results, consumed seeds and all old
implementations are protected by a CI diff against the base commit. Only a new 006
entry is appended to the root registry; its previous entries must remain identical.

## Reuse and new research logic

| Capability | Reused source |
|---|---|
| FFN Cell and standard top-2 expert dispatch | `experiments.common.models` canonical facade |
| Batch slicing / JSON writing | `experiments.common.train` |
| Atomic optimizer/RNG checkpoint | Read-only `exp005.checkpoint`, with explicit path RNG and global NumPy state added in a 006 wrapper |
| Observed linear GEMM hooks | Read-only `exp005.accounting.ObservedGemmFlops`, with 006 context/prior/head products added |
| Story-cluster paired bootstrap | Read-only `exp005.report.paired_bootstrap_delta` |
| Safe result branch, Git preflight, Askpass and push | Read-only `exp005.publish.prepare_result_branch`, `publish_core_validation_007`, `publish_experiment_results` |

New logic is restricted to the path-first model, whole-trajectory objective,
006-specific causal cache, interventions, independent assets and review gates.
The common historical dataset loader is intentionally not called: it opens the
validation prefix used by 005. 006 asset preparation uses only explicitly supplied
train-split Parquet rows, with its own identity and disjointness checks.

No GPU was used, no real 006 corpus was prepared, and no confirmation data was read.
Gate 0 fixtures establish engineering properties only. Final scientific claims
remain constrained by the capability ceiling and the original spec.
