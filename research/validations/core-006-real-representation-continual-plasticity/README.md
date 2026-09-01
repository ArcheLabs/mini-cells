# Core Validation 006 — Real-Representation Continual Plasticity

Status: **REAL_REPRESENTATION_CONTINUAL_PLASTICITY_NOT_SUPPORTED**

Formal result: **0/3 seeds passed** (`80611`, `80612`, `80613`). The frozen gates were not retuned after formal opening.

## Decision question

Can bounded dependency-aware subspace state support replay-free continual learning on real pretrained language-model representations without rapid saturation or growth explosion?

Core Validation 006 used:

- frozen `EleutherAI/pythia-160m@step143000`;
- pinned `DKYoon/SlimPajama-6B@1224c66add28b96ab045cd1058e795e8d3595485` real text from seven sources;
- 64-D projected Pythia hidden states;
- 32 fixed K-means addresses across eight base Cells;
- linear writable Cells with per-address covariance certificates;
- four variants: `unsafe`, `certificate_no_growth`, `certificate_mitosis`, and `replay`;
- dependency-partitioned clone-and-move mitosis;
- real next-token NLL, registered-history regression, effective-rank, dependency, reuse, and causal-ablation diagnostics.

## Formal provenance

- code commit: `4d5fdddeef7aa3b665846c7ba3c268206d133294`
- code tree: `02ab41489bbba4f956aade6f962d1a83829273d8`
- protocol SHA-256: `d278e038b92abaa5ca85f72abbc0041b25f962da79f3b2a05a0ad49e42eb9a7d`
- data manifest SHA-256: `d098f9172083b8de9f825b66de5277dde5b6ea0581b3a950b8f76e4f443546cc`
- resolved model SHA: `b56d9bee36300031aeea723b73c4d62ac7fa71a2`
- GPU: Tesla T4, CUDA 12.8
- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- elapsed formal runtime: 143.95795154571533 s

## Formal gate summary

| seed | regression / unsafe | gain / replay | midstream rank fraction | reuse ratio | split conflict reduction | spawned/address fraction | child reuse | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 80611 | 0.2712 | 0.8526 | 0.3281 | 1.8083 | 0.1495 | 0.6875 | 48 | FAIL |
| 80612 | 0.3455 | 0.8861 | 0.3281 | 1.3032 | 0.0000 | 0.6563 | 49 | FAIL |
| 80613 | 0.2808 | 0.8384 | 0.2813 | 1.5108 | 0.0000 | 0.6563 | 52 | FAIL |

All three seeds passed the following registered gates:

- no learner replay for the candidate;
- real representation did not immediately fill the 64-D Cell space;
- functional reuse grew;
- registered-history regression was reduced relative to `unsafe`;
- new-learning gain remained at least 0.80x `replay`;
- mitosis improved plasticity relative to `certificate_no_growth`;
- spawned children were reused;
- non-zero heldout causal signals existed.

All three seeds failed bounded growth because spawned Cells covered about 65.6–68.8% of the 32 addresses, above the frozen 50% maximum.

All three seeds also failed the split-conflict-relief requirement. Median conflict reduction was `0.149531...`, `0`, and `0`, below the frozen minimum `0.15`.

## Scientific interpretation

The formal hypotheses resolved as:

```text
bounded_certificate_reduces_registered_forgetting_without_replay = true
real_hidden_states_have_reusable_functional_geometry = true
dependency_partitioned_mitosis_restores_plasticity = false
growth_remains_bounded_and_reused = false
```

The negative result is therefore narrower than a complete rejection of the certificate mechanism.

The real Pythia representation did **not** rapidly saturate: midpoint 99%-energy rank occupied only about 28–33% of the 64-D Cell space. Functional reuse also increased, and replay-free certificate writes retained roughly 84–89% of replay new-learning gain while reducing registered forgetting to roughly 27–35% of the `unsafe` baseline.

The failed mechanism was the proposed **address-based mitosis boundary**. Moving a routing address into a cloned child usually did not materially reduce the parent's protected functional conflict. This indicates that fixed semantic/routing addresses are not equivalent to independent writable functional subspaces.

The actionable No-Go is therefore:

> **Dependency/routing address is not a sufficient functional boundary for bounded mitosis.**

A future continuation should derive Cell boundaries from functional-subspace geometry itself (for example covariance/eigenspace overlap or principal angles), rather than assuming routing addresses are the correct split units.

## Artifact completeness

The Kaggle formal run completed and printed the formal decision and full gate summary, but the session was closed before the generated result directory was published.

Canonical artifacts recovered from the completed run are stored under:

```text
artifacts/experiments/core-validation-006-real-representation-continual-plasticity/
```

Recovered:

- `decision.json`
- `gate-summary.csv`
- frozen `protocol.json`
- `RECOVERY_NOTES.md`

Detailed files such as `raw.json`, transaction/rank/split records, heldout/causal CSVs, plots, and the hidden-state cache were not available after session closure and have deliberately not been reconstructed.
