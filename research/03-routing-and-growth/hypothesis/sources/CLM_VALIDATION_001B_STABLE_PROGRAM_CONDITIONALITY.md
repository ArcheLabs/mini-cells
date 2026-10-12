# CLM Validation 001b — Stable Program Conditionality

## Question

Can an exactly converted TextNCA learn genuinely local, state-dependent sparse selection of its
shared FFN program shards while preserving language-model quality better than matched static
pruning and sample-shuffled routing?

This validation changes no CLM architecture. Cell activation is exactly one throughout. It does
not use phenotype, lifecycle, topology, capability labels, task labels, or semantic experts.

## Why Validation 001 was inconclusive

Validation 001 successfully established real-checkpoint dense equivalence. Its soft routing stages,
however, remained in the dense basin: the registered 0.75 and 0.50 targets produced approximately
0.9995 and 0.996 program activity. Subsequent hard phases were abrupt pruning. Dynamic mask
variation was not measured in the Dynamic branch, and the whole-executor ratio threshold of 0.5
was unreachable while the GRU remained fully active. The original result remains unchanged; the
formal diagnosis is `CONTINUATION_DID_NOT_LEAVE_DENSE_BASIN`.

## Preregistered continuation

After Stage 0 real-checkpoint equivalence, only receptor program-output biases are reset to a
zero-mean deterministic perturbation of scale `1e-4`. Cell-logit bias and all executor/backbone
parameters remain untouched. Hard top-8 therefore executes every shard and remains dense in the
forward pass, while its straight-through gradient learns program preferences around unsaturated
logits.

One AdamW optimizer and one continuous cosine scheduler are retained across:

| Stage | K | Tokens |
|---|---:|---:|
| Router warmup | 8 | 250K |
| B | 7 | 250K |
| C | 6 | 375K |
| D | 5 | 500K |
| E | 4 | 500K |

After every stage, Dynamic PPL is compared with the same student checkpoint in dense routing mode.
Reduction stops at the first ratio above 1.03. The last passing value is that replicate's `K*`.

## Conditionality and controls

At each replicate's `K*`, an independent calibration split selects the global Static top-K mask.
The formal holdout evaluates Dense, Dynamic, Static, and three preregistered sample permutations of
the exact Dynamic masks. Shuffling changes only the sample assignment; time, position, program
counts, usage, and compute remain intact.

Conditionality means both routing variation and causal matching value. The formal sample metric
first averages each sample's masks over recurrent steps and positions, then measures adjacent-sample
L1 distance over programs. Position and temporal variation are reported separately.

## Success criterion

At least two of three replicates must simultaneously satisfy:

- `K* <= 6` and Dynamic/dense PPL ratio `<= 1.03`;
- sample routing variation `>= 0.05`;
- normalized Dynamic NLL advantage over Static `>= 0.002`;
- normalized Dynamic NLL advantage over Shuffled `>= 0.002`;
- receptor/dense-executor cost `<= 0.05`.

The whole-executor ratio is reported but is not a program-sparsity gate. Thresholds are fixed before
the GPU run and must not be changed after observing results.
