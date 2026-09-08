# Granite Hybrid CLM v0.1

Status: `ENGINEERING_MILESTONE_GPU_PENDING`

This milestone is the first practical migration path from a mature pretrained MoE into a CLM-compatible model without retraining the foundation or requiring full post-hoc Cell decomposition.

## Definition

The model is intentionally hybrid:

```text
Granite 3.1 1B-A400M Base
        |
        | immutable legacy substrate
        v
  frozen transformer
        |
        +-----------------------------+
        | CLM evolution layer         |
        |                             |
        | independent address gates   |
        | shadow residual transforms  |
        | commit / rollback lifecycle |
        +-----------------------------+
```

The legacy Granite experts are **not** declared native Cells. They remain frozen legacy computation. New knowledge is represented by persistent CLM Cell artifacts.

## Why this exists

`CLM Conversion Kill Test 001` rejected the first post-hoc cellization scheme. Its dominant failures were coupled route/write formation, competitive top-1 routing, non-local writes and non-invariant child spawn. At the same time, local Cell writes were highly plastic and rollback remained exact.

This milestone keeps the useful property (small writable residual transforms) and removes the known structural hazards:

1. Cell allocation is explicit per mutation packet. Autonomous global semantic partitioning is out of scope.
2. Address and write training are separate phases.
3. Gates are independent sigmoids; Cells never compete through a shared softmax/top-1 denominator.
4. An allocated Cell is invisible to production until `commit_cell_`.
5. Shadow mode is the only path for training an uncommitted transform.
6. Address read occurs before all write sites, so committed Cells cannot perturb the representation used for future applicability gates.
7. Branch composition is artifact/manifest union, not weight addition.

## Engineering target

A successful run must demonstrate on the pinned Granite foundation:

- exact compatibility before the first commit;
- 50 sequential synthetic knowledge mutations;
- semantic candidate-choice acquisition for each committed mutation;
- at least 98% final semantic-choice retention across all committed facts;
- no foundation parameter updates;
- a contextual child update that learns a `v2` value while preserving the parent `v1` path;
- rollback by uncommitting the child;
- versioned Cell artifacts;
- manifest-level fork/merge;
- checkpoint reload with behavior parity.

This is deliberately an engineering milestone, not a new formal scientific kill test. Thresholds can be tuned during implementation until the runner is stable; once a public v0.1 checkpoint is selected, its exact protocol and artifacts should be frozen separately.

## Core files

```text
src/minicells/hybrid_clm.py
scripts/research/granite_hybrid_clm_v01/dataset.py
scripts/research/granite_hybrid_clm_v01/run_milestone.py
tests/test_hybrid_clm.py
```

## GPU run

```bash
python scripts/research/granite_hybrid_clm_v01/run_milestone.py \
  --device cuda:0 \
  --facts 50
```

For a low-cost preflight:

```bash
python scripts/research/granite_hybrid_clm_v01/run_milestone.py \
  --device cuda:0 \
  --facts 3 \
  --address-steps 80 \
  --transform-steps 12 \
  --output-dir results/granite-hybrid-clm-v0.1-smoke
```

## JAM demonstration

The controlled 50-fact run is the engineering acceptance surface because failures are easy to localize. Once it passes, the same Cell lifecycle should be applied to `research/datasets/jam-knowledge-v0.1` for the public `Granite Hybrid CLM learns JAM` demonstration.

JAM learning is not a requirement that the old Granite experts become native Cells. The public claim is narrower and mechanically true if the milestone passes:

> A frozen pretrained Granite MoE can acquire persistent new JAM knowledge through a CLM evolution layer while preserving the legacy foundation and allowing versioned rollback.

Progressive legacy-expert offloading is Milestone 2 and is explicitly not a dependency of this milestone.
