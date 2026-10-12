# CLM v2 Validation 001b — Closed-Loop Scaffold Handoff

## Question

Can the overcomplete conditional program bank take over the recurrent TextNCA dynamics through a genuine closed-loop scaffold homotopy, rather than being required to survive direct `alpha=1 -> alpha=0` replacement before the homotopy starts?

The CLM v2 architecture is unchanged. Cell activation remains exactly one. The experiment changes only the validation/training curriculum.

## Motivation from Validation 001

Validation 001 reached local relative MSE of approximately `0.052-0.055` and local cosine similarity of approximately `0.970-0.972` after 1M imitation tokens, but direct zero-scaffold counterfactual PPL remained approximately `1.12x` the dense teacher. The run stopped before any `alpha=0.75 -> 0.50 -> 0.25 -> 0` handoff stages.

That stopping rule contradicted the purpose of the scaffold homotopy. One-step local approximation on dense trajectories does not imply closed-loop recurrent equivalence after abrupt scaffold removal. Small local errors change recurrent state, which changes later local perceptions and therefore changes the distribution on which the conditional branch must operate.

Validation 001b preserves the direct `alpha=0` counterfactual as telemetry only and allows the registered homotopy to perform the distributional handoff it was designed for.

## Architecture lock

No changes are permitted to:

- `overcomplete_cellular_textnca.py`;
- `textnca_to_clm_v2.py`;
- program-bank width/count;
- receptor architecture;
- top-k semantics;
- cell activation;
- phenotype, topology, lifecycle or semantic labels.

The fixed sparse branch remains:

- shared hidden width: 128;
- 12 independent conditional programs;
- conditional expert hidden width: 64;
- initial `K=6`;
- total FFN genome capacity: `1.75x` dense;
- active K6 FFN capacity: `1.00x` dense.

## Stage 0 — scaffold parity

The Experiment 006 `minicells-v2-10m.pt` checkpoint is converted to CLM v2. At `alpha=1`, the CLM main recurrent path must remain equivalent to the original TextNCA. Failure stops the replicate.

## Stage 1 — local approximation

The inherited TextNCA backbone and dense scaffold are frozen. Only the shared program, conditional programs and local receptor train at:

- `alpha=1`;
- `K=6`;
- up to two 500K-token blocks.

The local gate is preregistered as:

- validation relative MSE `<= 0.10`;
- validation cosine similarity `>= 0.95`.

The direct `alpha=0` PPL ratio is recorded as `zero_scaffold_ppl_ratio_telemetry` but **does not gate entry to the homotopy**.

## Stage 2 — closed-loop homotopy

With `K=6`, the current student trajectory is trained sequentially at:

`alpha = 0.75 -> 0.50 -> 0.25 -> 0.00`.

Each stage uses 250K tokens and the existing CE + frozen-teacher KL + on-policy local imitation loss. The dense scaffold is frozen but is evaluated on the **current mixed-trajectory local perception**, so local imitation follows the student distribution as the scaffold weight falls.

Each stage records PPL before training at the new alpha and after training.

A stage passes only if both:

- safety: `PPL_after / PPL_teacher <= 1.20`;
- recovery: `PPL_after / PPL_before <= 1.01`.

These are continuation safety gates, not the final scientific quality criterion.

## Stage 3 — scaffold-free consolidation

After successful `alpha=0`, the sparse backbone is unfrozen except for the dense scaffold, which remains frozen and absent from the main forward path. K6 consolidation runs for 500K tokens.

The strict handoff criterion is restored here:

`PPL_alpha0_K6 / PPL_teacher <= 1.03`.

Passing this gate establishes `CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL`.

## Stage 4 — active-compute reduction

Only after successful scaffold-free K6 consolidation does the run test:

`K=5 -> K=4 -> K=3`.

Each stage uses 375K tokens. Reduction stops at the first stage whose PPL exceeds `1.03x` the consolidated K6 reference. The last passing K is the replicate's quality-safe K.

## Stage 5 — causal routing controls

At the quality-safe K, the formal holdout evaluates:

- Dense scaffold counterfactual;
- Dynamic local routing;
- matched global Static routing;
- three sample-Shuffled versions of the exact Dynamic masks.

Program conditionality requires at least two of three replicates to satisfy:

- scaffold-free handoff succeeded;
- quality-safe `K <= 5`;
- final PPL / teacher PPL `<= 1.03`;
- sample routing variation `>= 0.05`;
- normalized Dynamic NLL advantage over Static `>= 0.002`;
- normalized Dynamic NLL advantage over Shuffled `>= 0.002`;
- receptor cost `<= 5%` of dense FFN-equivalent compute.

## Official diagnoses

- `CLMV2_SCAFFOLD_EQUIVALENCE_FAILURE`
- `CLMV2_CLOSED_LOOP_LOCAL_APPROXIMATION_FAILURE`
- `CLMV2_CLOSED_LOOP_HANDOFF_FAILURE`
- `CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL`
- `CLMV2_CONDITIONAL_CAPACITY_WITHOUT_CAUSAL_ROUTING`
- `CLMV2_PROGRAM_CONDITIONALITY_SIGNAL`

## Interpretation discipline

A successful handoff would show that an overcomplete, locally routed program bank can replace the dense recurrent FFN scaffold after trajectory-aware continuation. It would not by itself establish semantic experts, tissues, capabilities, phenotype differentiation, cell sleep, growth, fork or topology evolution.
