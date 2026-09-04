# MoE Mutation 001

## Question

Can the byte-identical Granite 3.1 1B-A400M CLM substrate established by **MoE Conversion 001** accept one isolated, addressable mutation that improves a held-out target behavior while preserving bounded control behavior, preserving the target layer's routing decisions, and supporting exact rollback?

This is deliberately narrower than a composability test.

## Frozen base

- model: `ibm-granite/granite-3.1-1b-a400m-base`
- revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- required Conversion 001 manifest identity: `dd2b9c750567ff73b1d48e39eb7d1e1213eea9116a68c5164d023420f5a4d670`
- formal seeds: `26090411`, `26090412`, `26090413`

The original MoE router and all shared/base weights remain frozen. A MoE expert is still **not** defined as a CLM Cell.

## Mutation boundary

The mutation is restricted to the final MoE layer (`layer_index=23`). For each formal seed:

1. Run the frozen 24 training prompts through the untouched base model.
2. Inspect the target layer's last-token Top-K routing decisions.
3. Select the expert with the highest routing count; break ties by the lowest expert index.
4. Discover the target layer's two packed expert tensors at runtime.
5. Map each runtime tensor to its canonical Conversion 001 packed tensor by exact expert-slice shape.
6. Train only the selected expert slice in those two tensors.

Therefore a mutation artifact contains exactly two canonical addresses of the form:

```text
<packed-input-tensor>::expert[e]
<packed-output-tensor>::expert[e]
```

All other slices, router weights, attention weights, embeddings, normalization parameters, and shared parameters remain frozen.

## Task

The branch task is intentionally synthetic. It is not used to claim semantic knowledge acquisition.

Train prompts and held-out prompts share the frozen template:

```text
CLM mutation calibration record A-<seed_mod>-<index> resolves to
```

The train split uses indices `00..23`; held-out uses `24..31`.

The target is one token. From the frozen candidate word list in `protocol.json`, retain candidates that tokenize to exactly one token and choose the token with the **lowest mean base probability on the training prompts**. This creates a deterministic low-prior target without changing the protocol after results are observed.

Eight unrelated natural-language prompts are frozen as controls.

## Training

- dtype: FP32
- steps: 40
- batch size: 4
- learning rate: 0.05
- optimizer: manual masked SGD
- maximum selected-slice gradient norm: 1.0
- no weight decay

The packed parameter may receive a full tensor gradient, but the update operation is applied only to `parameter[selected_expert]`. No optimizer state is created for untouched experts.

## Per-seed gates

A seed passes only if all frozen gates pass:

- train next-token NLL gain >= `0.20`
- held-out next-token NLL gain >= `0.10`
- control mean KL(base || mutated) <= `0.05`
- control next-token argmax identity >= `0.875`
- target-layer router Top-K identity == `1.0`
- rollback max absolute logit error <= `1e-6`
- mutation delta norm > 0
- exactly two canonical mutation addresses
- base Conversion 001 manifest identity matches the frozen identity

The target-layer routing gate is intentionally strong: the router is evaluated before the selected expert computation in that layer, so changing the selected expert slice must not change that layer's routing decisions.

## Formal decision

- **SUPPORTED**: at least 2 of 3 formal seeds pass.
- **REJECTED**: 2 or more formal seeds fail.
- Otherwise the formal run is incomplete.

A supported result permits only the following narrow claim:

> An unchanged converted Granite MoE substrate supports a first isolated, addressable expert-slice mutation with measurable held-out gain, bounded control drift, unchanged target-layer routing, and exact rollback under the frozen protocol.

It does **not** establish multi-mutation composition, mergeability, continual learning, autonomous Cell discovery, arbitrary safe editing, or that an MoE expert is a Cell.

## Execution boundary

The canonical implementation lives under `scripts/research/moe_mutation_001/`. Hosted notebooks are launchers only. Formal Kaggle execution must use `notebooks/kaggle/moe-mutation-001.ipynb`, run one seed at a time, and publish each completed seed immediately before continuing.
