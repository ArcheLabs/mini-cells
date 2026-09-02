# Constructive CLM-001 — Learned Coordinate Formation

Status: **PROTOCOL_FROZEN_UNRUN**

## Why this experiment exists

The frozen [CLM Feasibility Evidence Map](../CLM_FEASIBILITY_EVIDENCE_MAP.md) shows that MiniCells already has direct evidence for growth-restored plasticity, replay-free Cell-local protection, independently mutable routed state, real-representation reusable geometry, and a simple causal foundation write interface.

The remaining first-order native-CLM gap is narrower:

> Can **Cell coordinates and read keys themselves be learned from continual experience**, instead of being assumed to exist naturally inside a pretrained LLM or being supplied as semantic labels?

This experiment therefore does **not** re-run Core 004 or Core 005. It intentionally removes certificates, Pythia and language-scale confounds and isolates the missing constructive bridge.

## Hidden world

A deterministic synthetic world contains six hidden reusable functional factors. Each factor has:

- a context prototype in a 16-D read space;
- an effect atom in a 12-D write/effect space.

Transactions contain one or two factors. Inputs and targets are noisy averages of the corresponding hidden prototypes/atoms. The learner never receives the factor IDs.

The curriculum has three regimes:

```text
warmup
  -> factor introduction
  -> late consolidation/recombination
```

A novel factor first appears as an unlabeled singleton transaction, then reappears in singleton and compositional contexts. The late region contains no new factors and therefore tests whether growth stops when reusable coordinates are sufficient.

## Learner

Each active Cell stores:

```text
Cell = (read_key, effect_value, usage, born_at)
```

Read is sparse:

```text
x -> cosine(read_key) -> TopK=2 -> softmax weights -> weighted Cell effects
```

Growth uses only current `(x, y)` fit quality and routing confidence. A newborn Cell is initialized from current transaction statistics and enters the same sparse routing system. Reuse updates are local key/value EMA writes on currently routed examples only.

No task label, factor ID, replay buffer, old sample, Pythia gradient, certificate or semantic address is passed into the learner.

This is deliberately a **scaffolded constructive** experiment: growth is still an engineered rule. The question is coordinate/read-key formation, not learned mitosis policy.

## Seed discipline

Development seeds `1001/1002/1003` were observed while implementing and testing the mechanism. They are permanently excluded from the scientific decision.

The untouched formal seeds are:

```text
90111
90112
90113
```

All three formal seeds are required.

## Primary gates

Every formal seed must independently satisfy:

- final active Cells `<= 8` for six hidden factors;
- late-half spawns `<= 1`;
- independent-memory compression `>= 4x`;
- heldout singleton MSE `<= 0.02`;
- heldout pair-composition MSE `<= 0.03`;
- singleton route recall `>= 0.90`;
- pair route recall `>= 0.85`;
- mean learned effect-atom cosine `>= 0.90`;
- mean matched read-key cosine `>= 0.85`;
- at least five of six hidden factors covered by learned Cells;
- pair MSE `<= 0.5x` a shuffled-address control.

Hidden factor identities are used only for post-hoc route/alignment metrics.

## Causal control

`shuffled_address` preserves effect targets and transaction count while permuting transaction contexts. It breaks stable read->effect structure without reducing memory capacity.

A positive main result must substantially outperform this control. This guards against the trivial interpretation that Cells are only memorizing effect vectors while the read geometry is irrelevant.

## Run

Development diagnostic only:

```bash
python scripts/research/run_constructive_clm_001.py --seed 1001
```

Frozen formal run:

```bash
python scripts/research/run_constructive_clm_001.py --formal
```

Outputs are written to:

```text
artifacts/experiments/constructive-clm-001-learned-coordinate-formation/
```

## Interpretation boundary

A positive result supports only:

```text
structured continual pressure
  -> reusable learned Cell coordinates
  <-> deployable learned read keys
  -> bounded growth in the controlled stream
```

It does **not** establish language-scale CLM, asymptotically sublinear growth, learned growth policy, certificate integration, endogenous foundation plasticity or JAM execution.

Those are deliberately left to Constructive CLM-002+ so that the project does not repeat already-settled mechanisms or combine too many unknowns in one decision.
