# CLM Conversion Kill Test 001 — Mature MoE Functional Cellization

Status: **PROTOCOL v1.2 FROZEN — GPU FORMAL EXECUTION PENDING**

## Question

Can a frozen mature MoE grow persistent, semantically addressable, independently mutable functional Cells without assuming that pretrained Expert or channel boundaries are already Cells?

This is the first direct bridge from the closed Constructive CLM mechanism chain into a real pretrained token-predictive MoE.

The frozen foundation is:

- `ibm-granite/granite-3.1-1b-a400m-base`
- revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`

The foundation is never trainable.

## Why this test exists

Earlier MiniCells evidence established two facts that must be reconciled:

1. Core 002/009C did not find a useful natural sparse/local Cell ontology in pretrained models.
2. Constructive CLM-001 through CLM-005 showed, under registered controlled worlds, that addressable coordinates can form, grow, accept protected writes, compose, and be governed by learned control.

JAM Knowledge Mutation 001 then showed that a mature Granite MoE is locally plastic and can acquire substantial domain behavior, but a fixed Expert/channel write boundary did not produce reliable semantic correction.

Conversion 001 therefore inserts the missing stage:

```text
mature frozen MoE
        ↓
zero-init CLM compatibility shell
        ↓
functional Cell formation
        ↓
semantic local mutation
        ↓
contextual child growth
        ↓
branch mutation / merge / rollback
```

It does **not** try to Cellize all pre-existing Granite knowledge.

## Cell substrate

A Cell is defined operationally as:

```text
Cell = semantic read key + cross-layer low-rank residual transform + mutable state
```

The registered substrate writes at decoder layers `7 / 15 / 23`.

Routing is read once at layer 7 and reused at later write sites in the same forward pass. This deliberately avoids defining each write site as an independent Cell and tests a minimal cross-layer functional operator.

All up-projection factors initialize to exactly zero. Before learning:

```text
converted_model(x) == frozen_foundation(x)
```

up to the measured foundation forward-repeatability floor.

## Controlled knowledge

The test uses deterministic fictional names and nonce codes. The foundation cannot receive credit for prior world knowledge.

The training world contains:

- 12 fictional entities;
- 6 fictional protocol codes;
- 3 fictional region codes;
- entity → protocol facts;
- protocol → region facts.

Training, checkpoint selection, and final heldout evaluation are deliberately separated:

- 36 training rows provide the formation pressure;
- 12 separate validation rows are used only to select the formation checkpoint under the frozen history-KL ceiling;
- final direct, negation, relation, routing, and candidate-choice metrics are never used for checkpoint selection.

The final heldout probes use different wording and separate direct semantic recall, negation correction, two-hop relation use, and routing consistency across four paraphrases per entity.

### v1.2 semantic anti-shortcut gate

Reference-NLL improvement alone is not sufficient. For every semantic heldout question, the formal runner teacher-forces **every registered nonce candidate against the same question** and compares answer-token NLL with EOS excluded.

For example, for a question about `Zorven`, the model is scored on all six protocol candidates:

```text
QX-17  LM-42  VR-08  PN-63  TK-51  HF-29
```

A semantic-choice row is correct only if the registered reference value has **strictly lower answer-token NLL than every incorrect candidate**. Ties fail. Because EOS is excluded, a generic termination or code-format shortcut cannot satisfy this gate.

The dataset generator is frozen by its Git blob identity in `protocol.json`. Formal results from a different generator or protocol hash must not be aggregated.

## Registered phases

### P0 — compatibility

Install the zero-init substrate and compare logits with the frozen foundation. This must remain within measured repeatability plus the frozen excess tolerance.

### P1 — Cell formation

Train only the CLM substrate. The foundation remains frozen.

The router receives no hard Cell ID labels. Pairwise paraphrase consistency, route diversity pressure, answer-token NLL, and a small frozen-foundation KL preservation term provide the formation pressure.

Checkpoint selection uses only the registered validation split. Final heldout NLL and candidate-choice metrics remain untouched until the formation snapshot has been selected.

A formal seed must satisfy both NLL acquisition and candidate-choice accuracy on direct, negation, and relation families.

### P2 — semantic local write

Pick one entity's learned primary Cell. Change that entity's protocol using only that Cell's transform parameters. Its read key remains frozen.

Unseen rewrite wording must improve, the new value must rank first among all six semantic candidates, and unrelated entity mappings must not regress in NLL or candidate-choice accuracy.

### P3 — contextual child growth

Create a contextual conflict (`archive Alpha` vs `archive Beta`).

First run a parent-only update. A valid growth demonstration now requires that this parent-only control **cannot simultaneously satisfy** the registered Alpha-retention, Beta-acquisition, and Beta semantic-choice gates.

Then restore the formation snapshot, spawn one child from the parent, and train only that child and its read key for the Beta context.

The child copies the parent key and transform at birth. Before specialization, spawn must be function-preserving under both reference NLL and candidate-choice margin, while semantic-choice accuracy must remain unchanged.

After specialization, Beta must be semantically acquired, Alpha must remain semantically correct, and Beta prompts must route to the child at the registered rate.

The growth trigger itself is scaffolded. Learned growth control is already a separate Constructive CLM-005 result and is not re-tested here.

### P4 — branch / merge / rollback

Find two entity concepts whose learned primary Cells are distinct. Starting from the same formation snapshot:

1. train branch A by changing only Cell A;
2. train branch B by changing only Cell B;
3. require each standalone branch to satisfy both NLL acquisition and semantic candidate-choice;
4. export the two Cell-scoped mutations;
5. restore the common parent;
6. apply both mutations;
7. require NLL gain retention and semantic candidate-choice retention for both branches;
8. restore the parent snapshot and require exact tensor rollback.

If formation produces fewer than two distinct entity Cell addresses, the branch-composition gate records a scientific FAIL rather than treating the run as an infrastructure error.

## Kill gates

A formal seed fails if any registered gate fails. v1.2 includes:

- zero-init compatibility;
- direct NLL acquisition + semantic candidate-choice;
- negation NLL acquisition + semantic candidate-choice;
- relation NLL acquisition + semantic candidate-choice;
- frozen-foundation preservation;
- cross-paraphrase route agreement;
- route diversity;
- semantic local-write NLL gain + 100% heldout candidate-choice;
- unrelated NLL locality + candidate-choice non-regression;
- function-preserving child spawn;
- growth necessity: parent-only must fail the registered conflict solution;
- Beta acquisition + 100% candidate-choice;
- Alpha retention + 100% candidate-choice;
- child routing;
- two standalone branch acquisitions + 100% candidate-choice;
- merged NLL retention + 100% candidate-choice for both branches;
- exact overlay rollback.

Formal support requires at least 2 of 3 untouched seeds to pass every gate.

## Freeze rule

Formal seeds are:

```text
26090441
26090442
26090443
```

Once the first formal seed starts, no scientific gate, threshold, dataset identity, model revision, seed, training rule, or evaluation rule may change. Any later engineering repair must preserve the registered scientific semantics and be documented explicitly.

## Interpretation boundary

A PASS would support:

> A frozen mature MoE can host a newly formed, semantically addressable, cross-layer plastic Cell substrate that survives local mutation, contextual growth, branch composition and rollback in the registered fictional-knowledge world.

A PASS would **not** establish full Cellization of Granite's existing knowledge, natural Expert/neuron boundaries as Cells, fully learned growth control in a mature LLM, asymptotically sublinear growth, superiority to LoRA or routed adapters, or JAM domain support.

Matched LoRA and static-routed-adapter baselines are explicitly gated behind a successful mechanism bridge. If this test fails, the failure must be classified at the registered boundary rather than repaired by changing thresholds.

## Run

CPU engineering tests:

```bash
python -m pytest -q tests/test_functional_cellization.py \
  tests/test_clm_conversion_kill_test_001.py
```

Formal hosted-GPU execution under the frozen v1.2 protocol:

```bash
python scripts/research/clm_conversion_kill_test_001/run_seed_v12.py \
  --seed 26090441 --device cuda:0
```

The Kaggle notebook under `research/notebooks/07-safe-model-evolution/clm-conversion-kill-test-001-kaggle.ipynb` runs all untouched formal seeds and publishes each completed seed immediately. Recovery skips a seed only when its terminal artifact matches the current frozen protocol SHA-256.
