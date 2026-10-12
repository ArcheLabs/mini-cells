# Experiment 020 — Capability Tissue Specificity

## Motivation

Experiments 017–019b established several properties separately:

- capability learning can be causally concentrated in newborn phenotype;
- sleeping newborn tissue protects old behavior;
- fully active trained tissues are broadly useful;
- however, Experiment 019b showed that the matching skill tissue is often not uniquely better than wrong trained tissues.

The unresolved question is therefore upstream of recruitment:

> Do different capabilities form functionally distinct tissues at all?

Without capability identity, no recruitment rule can select a meaningful specialist.

## Phase A — zero-training 019b baseline

020 first re-analyzes the existing 019b full-activation response matrix.

For input family `a` and tissue trained on family `b`:

```text
U_ab = mean[L_a(e=0) - L_a(T_b, e=1)]
```

Specificity is:

```text
S_a = U_aa - mean_{b != a} U_ab
S_a_norm = S_a / (|U_aa| + eps)
```

The strict margin is:

```text
M_a = U_aa - max_{b != a} U_ab
```

This baseline performs no training and records how weak/strong tissue identity already was under the 019 one-cell localized-learning protocol.

## Phase B — matched fixed-geometry comparison

Two donor geometries are trained from exactly the same stable-019 Phase-1 checkpoint:

### one-cell

```text
old-parent <-> c1
```

### three-cell-chain

```text
old-parent <-> c1 <-> c2 <-> c3
```

Controls shared across arms:

- same Phase-1 state per replicate;
- same 6 skill families;
- same skill corpus and validation corpus;
- same per-step batch schedule;
- same 200 optimization steps;
- same optimizer and LR;
- same frozen shared genome;
- old phenotype fully protected;
- only newborn `cell_memory` receives gradients;
- no autonomous fork/connect/prune after initial allocation;
- no router, recruitment gate, task ID, or expert ID is used in inference.

The first newborn is identical between the two arms. The three-cell condition adds a deterministic protected chain using an orthogonal second perturbation and a third continuation. Thus the intended experimental variable is tissue capacity/geometry.

## Evaluation

Every trained tissue is evaluated on every skill family using the same held evaluation examples as Experiment 019/019b.

Outputs include:

- full 6x6 cross-skill utility matrix per geometry;
- matching-vs-wrong specificity;
- normalized specificity;
- strict margin against the best wrong tissue;
- matching tissue rank on replicate-mean utility;
- example-level matching-tissue top-1 accuracy;
- language retention;
- tissue ablation causal fraction;
- transplant recovery;
- old-memory drift.

A stronger skill gain alone is not specificity. If all `U_ab` rise together, the experiment does not count that as capability identity.

## Pre-registered family PASS

For a skill family and geometry:

```text
mean normalized specificity >= 0.10
matching tissue ranks #1 in >= 2/3 replicate-mean matrices
example-level matching-tissue top-1 accuracy >= 0.50
```

Integrity is evaluated independently. At least 2/3 replicates must satisfy all three:

```text
language retention ratio <= 1.10
tissue causal fraction >= 0.90
transplant recovery >= 0.90
```

A general geometry signal requires at least 4/6 skill families.

## Decision statuses

- `MULTICELL_TISSUE_SPECIFICITY_SIGNAL`
  - three-cell specificity PASS >=4/6 and joint specificity+integrity PASS >=4/6.
- `SPECIFICITY_WITH_INTEGRITY_COST`
  - three-cell specificity emerges but retention/localization/transplant integrity does not.
- `FIXED_ONECELL_TISSUE_SPECIFICITY_SIGNAL`
  - the matched fixed one-cell arm itself reaches >=4/6 joint PASS.
- `PARTIAL_MULTICELL_TISSUE_SPECIFICITY`
  - three-cell improves the family pass count but does not reach the general threshold.
- `NO_CAPABILITY_TISSUE_IDENTITY`
  - neither geometry produces a meaningful capability-specific diagonal structure.

## Interpretation

If three-cell succeeds while one-cell fails, the result supports the hypothesis that a capability requires a small coherent tissue rather than a single generic residual-adapter cell.

If both fail, recruitment remains deferred. The next experimental variable should be the learning objective or allowed phenotype/genome degrees of freedom, not another gate or routing heuristic.

## Run

020 requires the local stable-019 checkpoints and the 019b response-curve baseline in the same Kaggle workspace (or merged 019b curated artifacts).

```bash
python -m pytest tests/research/02-self-organization/test_language_tissue_specificity.py -q
python scripts/research/run_capability_tissue_specificity.py
```

The runner uses up to two GPUs concurrently. Each of the 36 trained donors is checkpointed independently, so interrupted runs reuse completed donors.
