# MiniCells Core Validation 001 — Knowledge Subsumption and Computational Reorganization

## Question

This experiment tests one claim only:

> Can cumulative experience replace memorizing computation with a reusable generalizing circuit while preserving previously acquired behavior?

The experiment does **not** test growth. It disables mitosis, tissue formation, hierarchy, apoptosis, message passing, routing growth, and parameter expansion. A positive result is not evidence that MiniCells uniquely causes grokking.

## Why this comes before growth

The continual-learning update is

\[
M_t + D_{t+1} \rightarrow M_{t+1}.
\]

Internal knowledge is not assumed to be append-only. An early model may fit examples through memorizing computation; later experience may support a reusable rule that subsumes those mappings. The behavior must remain correct, but the historical internal representation is allowed to disappear.

The experimental pattern follows the modular-addition grokking literature, especially the distinction between a memorizing circuit, a forming generalizing circuit, and later cleanup of memorizing components described by Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* (ICLR 2023).

## Task

Primary task:

\[
y=(a+b) \bmod 31.
\]

Negative control: a deterministic random **commutative** mapping over the same `(a,b)` inputs. It is constructed so that:

- `label(a,b) == label(b,a)`, matching the exact symmetry of the model input;
- every output class occurs exactly 31 times over the 31 x 31 ordered input pairs;
- the mapping is random and therefore does not contain the modular-addition rule.

This matters because an asymmetric random-label control would be structurally unrepresentable by a model whose input is `embedding(a) + embedding(b)`, creating a trivial negative control.

The full 31 x 31 pair set is shuffled deterministically for each seed. Four disjoint training phases consume 10%, 20%, 20%, and 20% of the pairs. The remaining ~30% is held out for the entire run. During a phase, the optimizer receives only that phase's examples; old examples are never replayed.

## Model

The formal model is intentionally conventional and fixed so the experiment does not smuggle in a MiniCells growth mechanism:

- one shared learned number embedding;
- a factored one-hidden-layer MLP;
- ReLU activation;
- tied input/output embedding;
- 128 hidden neurons;
- the hidden layer is partitioned into 16 fixed groups of 8 neurons only for causal ablation.

These groups are called `cells` in the instrumentation, but the experiment does **not** assume that one fact maps to one cell or that the groups have persistent semantic identity.

## Why Fourier interventions

Modular addition has a known periodic generalizing solution. Grokked networks can represent the rule through a small number of Fourier modes rather than through independent fact memorization.

At the late checkpoint, the experiment ranks conjugate Fourier-frequency pairs by energy in the learned number embedding and freezes the top three pairs as the candidate generalizing circuit frequencies.

The same frequency set is then applied to both early and late checkpoints:

- **restricted model**: retain DC plus only the selected key frequency pairs;
- **excluded model**: remove only those key frequency pairs and retain the rest.

This gives a direct operational test of circuit formation and cleanup:

- if the early excluded model still fits early examples, memorizing computation exists outside the future generalizing frequencies;
- if the late restricted model handles old and held-out examples, the compact generalizing circuit has become sufficient;
- if the late excluded model can no longer handle old examples, old behavior has become dependent on the new generalizing circuit rather than the old memorizing remainder.

This is a task-specific mechanistic test, not a universal definition of knowledge.

## Secondary causal-cell diagnostic

For example `x` and fixed hidden cell `i`, responsibility is the positive increase in per-example NLL after ablating that cell:

\[
R_{i,x}=\max(0, L_x(M^{-i})-L_x(M)).
\]

The top cells form a causal path fingerprint. Mean pairwise Jaccard overlap is reported at early and late checkpoints as a secondary measure of reuse. It is **not** part of the formal pass/fail gate because the fixed hidden partition is arbitrary.

## Frozen gates

A primary-seed run passes only when all four gates pass.

### G1 — Early memorization

At the captured early checkpoint:

- first-phase accuracy >= 0.98;
- all not-yet-seen examples have accuracy <= 0.20.

This confirms that the run actually contains a memorization state before generalization.

### G2 — Late generalization with behavioral retention

At the final checkpoint:

- old-phase accuracy >= 0.90;
- current-phase accuracy >= 0.90;
- permanently held-out accuracy >= 0.90.

Because old examples are not replayed, the final computation must cover earlier behavior as well as genuinely unseen pairs.

### G3 — Generalizing circuit sufficiency

Using only the late-selected key Fourier pairs:

- restricted old accuracy >= 0.80;
- restricted held-out accuracy >= 0.80.

This requires a compact reusable circuit to be sufficient for both retained and generalized behavior.

### G4 — Memorization cleanup / subsumption

Using the complementary excluded model:

- early excluded seen accuracy >= 0.80;
- late excluded old accuracy <= 0.30;
- late excluded held-out accuracy <= 0.20.

The first condition establishes that early success did not already depend on the eventual generalizing frequencies. The late conditions establish that, after learning, removing the generalizing circuit destroys both the retained old behavior and unseen generalization: the new circuit has subsumed the old behavior.

## Negative-control validity

A random-label run is allowed to count as a negative control only if it is demonstrably learnable by the same system:

- it must satisfy the same early-memorization gate;
- its final current-phase accuracy must be >= 0.90.

If a control does not meet these conditions, the whole formal experiment is invalid rather than treating the control's failure to form a generalizing circuit as evidence for the hypothesis.

## Replication rule

Formal seeds are 73101, 73102, and 73103.

The experiment reports `KNOWLEDGE_SUBSUMPTION_SUPPORTED` only if:

1. **all three** modular-addition runs pass all four gates;
2. **all three** random-label controls satisfy the control-validity requirement; and
3. **zero** random-label controls produce a false positive under the same four formal gates.

There is no majority-vote rescue and no post-hoc threshold tuning in the frozen v1 protocol.

An oracle modular-addition reference is trained on the cumulative union of training phases. It is descriptive only and does not affect the decision.

## Interpretation

A positive result supports this narrow statement:

> Under this controlled continual curriculum, a fixed neural system can preserve old behavior while shifting from memorizing computation toward a reusable generalizing circuit that subsumes the old behavior.

It does **not** establish that:

- MiniCells is superior to a conventional Transformer or MLP;
- dynamic growth is necessary;
- cells permanently own knowledge;
- all continual-learning problems admit this kind of compression;
- SGD always finds a mathematically shortest path.

A negative formal result rejects the frozen Core Validation 001 v1 mechanism test. The correct next action is to identify which gate failed, not to add growth or tissue mechanisms automatically.

## Run

Formal Kaggle/GPU run:

```bash
python scripts/research/run_core_validation_001.py
python scripts/research/report_core_validation_001.py
```

CPU smoke run:

```bash
python scripts/research/run_core_validation_001.py --smoke --device cpu --skip-oracle
python scripts/research/report_core_validation_001.py
```

Outputs are written to:

```text
results/core-validation-001-knowledge-subsumption/
```

The Kaggle notebook is `research/notebooks/04-continual-learning-core/core-validation-001-knowledge-subsumption.ipynb` and its final cell publishes curated formal results to `kaggle/core-validation-001-knowledge-subsumption-results` using the existing Kaggle secret `GITHUB_TOKEN`.
