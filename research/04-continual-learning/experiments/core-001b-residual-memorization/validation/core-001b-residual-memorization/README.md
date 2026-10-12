# Core Validation 001b — Generalization vs Residual Memorization

## Question

Core Validation 001 produced a stable result across three seeds:

1. the first curriculum phase was memorized;
2. the late model generalized modular addition to held-out pairs;
3. a small set of late Fourier modes was sufficient for high held-out accuracy;
4. however, removing only the top three modes did not always destroy old-example accuracy enough to satisfy the frozen `memorization_cleanup` gate.

The unresolved ambiguity is:

> Does the accuracy that survives removal of the dominant Fourier modes come from residual memorization of old examples, or from additional generalized computation distributed across other modes?

Core Validation 001b tests only that ambiguity.

## What 001b does not change

001b does not introduce a new learner.

The Kaggle notebook first reruns the existing frozen Core Validation 001 training code:

```text
scripts/research/run_core_validation_001.py
```

with the same:

- modular-addition curriculum;
- balanced commutative random-label control;
- three seeds;
- model;
- optimizer;
- no replay;
- no growth.

The parent oracle is skipped during this stage only to avoid training it twice. Core 001b then loads the newly produced early and late checkpoints and performs the extended intervention.

Therefore any difference between 001 and 001b is diagnostic, not algorithmic.

## Intervention

For modulus 31 there are 15 non-DC conjugate Fourier-frequency pairs.

For each late checkpoint:

1. measure energy in every pair;
2. rank all 15 pairs from highest to lowest late embedding energy;
3. for every `k = 0..15`:
   - **exclusion:** remove the top `k` pairs while retaining DC;
   - **restriction:** retain only the top `k` pairs plus DC;
4. evaluate the same intervention on:
   - early seen vs early unseen examples;
   - late old vs late held-out examples.

The early model is a positive-control assay for memorization. At `k=0`, seen accuracy should be high while unseen accuracy is low.

The late old and held-out sets are random partitions of the same modular-addition function. Formal coupling gates use **output-class-balanced accuracy** so incidental class-count imbalance cannot masquerade as a membership effect. Raw accuracy is retained only as a secondary diagnostic. If the surviving computation is generalized, the two partitions should degrade together as Fourier pairs are removed. If old examples retain a material advantage over held-out examples, that is the operational signature of residual memorization.

## Frozen primary gates

A primary seed passes only if all of the following hold.

### Parent preconditions

The corresponding Core Validation 001 run must already satisfy:

- early memorization;
- late generalization;
- generalizing-circuit gate.

### Assay sensitivity

At zero removal:

```text
early seen accuracy - early unseen accuracy >= 0.50
```

This prevents a negative residual-memory result from being accepted when the assay never demonstrated that it could expose memorization.

### Synchronized late degradation

Across the full `k=0..15` exclusion curve:

```text
corr(old balanced accuracy, heldout balanced accuracy) >= 0.95
```

### No material membership advantage

Across the same curve:

```text
mean |old balanced accuracy - heldout balanced accuracy| <= 0.05
max(old balanced accuracy - heldout balanced accuracy, 0) <= 0.10
```

The second condition specifically prevents a transient old-only plateau from being hidden by averaging.

### Endpoint

After all 15 non-DC pairs are removed:

```text
old balanced accuracy <= 0.15
heldout balanced accuracy <= 0.15
```

This verifies that the sweep actually destroys the informative computation rather than ending while substantial function remains.

## Oracle validity

The cumulative-replay oracle is retrained inside 001b because the full model is needed for all 16 intervention points.

The same coupling assay is applied to oracle seen vs held-out examples. If the oracle fails the preregistered coupling/endpoint criteria, the 001b diagnostic is invalid and the experiment cannot report a positive result.

This directly addresses the main weakness exposed by Core Validation 001: a diagnostic must not classify a clearly generalized oracle as residual memorization merely because one small subset of modes was removed.

## Random-label control

The balanced commutative random-label arm remains a memorization control.

It must:

- remain valid under the parent Core 001 control criterion;
- produce zero Core 001b false positives.

Because it lacks the late-generalization parent precondition, it cannot satisfy the complete 001b hypothesis merely by producing numerically similar degradation curves.

## Formal decision

The positive status is:

```text
NO_MATERIAL_RESIDUAL_MEMORIZATION_DETECTED
```

and requires:

- all three modular-addition seeds pass;
- all random-label controls remain valid;
- zero random-label false positives;
- oracle assay valid.

Otherwise:

```text
RESIDUAL_MEMORIZATION_OR_INCONCLUSIVE
```

## Interpretation limit

A positive result does **not** mean that all historical parameter traces were erased.

It means something narrower and falsifiable:

> Under a complete cumulative Fourier-removal intervention, the late model shows no material class-balanced behavioral advantage for training-membership examples over held-out examples, while the same model family clearly exhibits such an advantage in its early memorization state.

That would support the claim that the surviving late behavior is primarily shared generalized computation rather than a privileged old-example lookup path.

Only after that ambiguity is resolved should MiniCells-specific mechanisms for active consolidation, role transfer, recycling, or growth be tested.
