# CLM-0.3c — Counterfactual Mitosis

## Question

Can a trained CLM estimate the *real future value* of untying one lineage before committing a birth?

CLM-0.3c keeps the CLM-0.3 growth operator frozen. It does not redesign hierarchical lineage routing, exact parent duplication, geometry initialization, optimizer-state inheritance, or the birth-equivalence gate.

The experiment tests a different proposition:

> A birth should be treated as relaxing the parameter-tying constraint between two prospective child regions. WHEN and WHERE are therefore the same decision: grow only when the best candidate has positive counterfactual marginal value relative to continuing without growth.

## Prior evidence

CLM-0.3 and CLM-0.3b established that executed births can be function preserving and that newborns can acquire traffic, diverge from their parent, and develop statistically detectable causal specialization. They did not establish a practically large advantage over a matched no-growth continuation, and heuristic parent scores did not reliably beat random selection.

CLM-0.3c therefore does not introduce another direct `pressure` heuristic as the decision rule. An analytic score is used only as a diagnostic/screening proxy. The formal decision is based on short counterfactual continuations from an identical checkpoint.

## Frozen substrate

- Base release: CLM-0.1
- Base SHA-256: `87d36c408ae3873ffd567ebf17050661b42ddae2c8d5d1bab84b2c27c3c7e7a0`
- Recurrent substrate: unchanged GRUCell core
- Root topology: 3 stages × 4 CLM-0.1 experts
- Birth: exact `child = deepcopy(parent)`
- Child competes only inside the parent lineage
- Geometry initialization: deterministic cosine k-means, k=2
- Optimizer: AdamW; child inherits parent moments; split router starts fresh
- Scheduler: continuous global continuation schedule
- Objective: `CE + 0.5 * KL(student || frozen CLM-0.1)`
- Root balance auxiliary loss: `0.0`

## Formal replicates

Seeds remain paired with prior CLM experiments:

- 55031
- 55032
- 55033

Each replicate is one self-contained counterfactual experiment.

## Decision checkpoint

Each replicate first trains a fixed-4 CLM from the released CLM-0.1 checkpoint for exactly **1.5M continuation tokens**.

The resulting serialized model, optimizer, scheduler, RNG state, and data schedule form the immutable decision checkpoint. Every shadow branch is reconstructed from this exact checkpoint.

CLM-0.3c intentionally does not require a saturation gate. CLM-0.3b showed that saturation is neither necessary nor sufficient for deciding whether new capacity has a better marginal return than existing capacity.

## Analytic split-regret proxy

Calibration uses the 16 training microbatches immediately preceding the decision checkpoint.

For each active lineage `i`, the calibration records routed perceptions and the gradient of the expert parameters for each microbatch. A deterministic two-cluster cosine partition is fitted to the lineage perceptions.

For microbatch `b`, let:

- `g_ib`: expert gradient divided by the number of routes through lineage `i`
- `n_ib0`, `n_ib1`: routed perceptions assigned to the two prospective child regions

The prospective region gradients are estimated as route-count-weighted means:

`g_i0 = sum_b n_ib0 * g_ib / sum_b n_ib0`

`g_i1 = sum_b n_ib1 * g_ib / sum_b n_ib1`

Let `v_i` be the parent Adam second moment. The Adam-metric disagreement is:

`D_i = mean((g_i0 - g_i1)^2 / (sqrt(v_i) + eps))`

With lineage utilization `U_i` and region masses `pi_i0`, `pi_i1`, the preregistered analytic proxy is:

`R_i = U_i * pi_i0 * pi_i1 * D_i`

`R_i` is interpreted as a local **split regret / untying-value proxy**: the amount of learning pressure hidden by forcing two prospective child regions to share the same parameters.

The proxy is *not* allowed to decide the formal birth by itself.

## Counterfactual shadow probes

At 1.5M, every active root lineage is probed from the same checkpoint.

Formal branches:

1. `no_growth`
2. one `split_i` branch for every one of the 12 CLM-0.1 lineages

Each branch receives the exact same next **100K training tokens**, validation schedule, LR schedule, and starting optimizer/RNG state.

Every split branch must pass the existing `CLM_GROWTH_EQUIVALENCE` gate before its first optimizer step.

The probe endpoint is evaluated on 32 validation batches (32,000 target tokens).

For candidate `i`:

`V_i(100K) = NLL_no_growth - NLL_split_i`

and the relative value is:

`Q_i(100K) = V_i(100K) / NLL_no_growth`

A deterministic paired 2,000-sample batch bootstrap produces a 95% interval for `Q_i`.

## Formal WHEN + WHERE policy

The counterfactual policy chooses the candidate with the largest 95% lower confidence bound:

`i* = argmax_i LCB95[Q_i(100K)]`

Then:

- if `LCB95[Q_i*] > 0`: action = `GROW(i*)`
- otherwise: action = `NO_GROW`

This is the formal unification of WHEN and WHERE.

## Long-horizon confirmation

Selection by a 100K shadow continuation can overfit short-term optimization dynamics. Therefore the selected candidate and a fresh no-growth control are both reconstructed again from the original 1.5M decision checkpoint and continued for **500K tokens**.

The confirm endpoint uses the same 32 validation batches and paired bootstrap:

`Q_i*(500K) = (NLL_no_growth_500K - NLL_split_i*_500K) / NLL_no_growth_500K`

The 500K continuation is an evaluation of the 100K decision, not an input to that decision.

## Formal decisions

### 1. Counterfactual birth equivalence

All 36 executed probe births (12 candidates × 3 replicates), plus the three confirmation births, must pass the existing exact parity gate.

Status:

`CLM_COUNTERFACTUAL_GROWTH_EQUIVALENCE`

### 2. Split-regret predictive signal

For every replicate, compute Spearman correlation between analytic `R_i` and realized 100K `Q_i` across all 12 candidates.

Signal requires `rho >= 0.30` in at least 2/3 replicates:

`CLM_SPLIT_REGRET_PREDICTIVE_SIGNAL`

This is a secondary endpoint; failure does not invalidate counterfactual selection.

### 3. Counterfactual decision calibration

The 100K policy is considered correctly calibrated for a replicate when:

- it chooses `GROW` and the selected candidate has `LCB95[Q_i*(500K)] > 0`, or
- it chooses `NO_GROW` and the selected candidate has `UCB95[Q_i*(500K)] <= 0`.

An interval crossing zero is inconclusive, not a success.

Signal requires at least 2/3 calibrated replicates:

`CLM_COUNTERFACTUAL_DECISION_SIGNAL`

### 4. Confirmed marginal capacity value

Independent of the 100K action, record whether the probe-selected candidate has a strictly positive 500K lower confidence bound.

At least 2/3 gives:

`CLM_COUNTERFACTUAL_CAPACITY_VALUE_SIGNAL`

### 5. Practical growth effect

The prior 0.5% practical threshold remains a separate endpoint. A replicate passes when the selected split has at least 0.5% lower PPL than the 500K no-growth control.

At least 2/3 gives:

`CLM_COUNTERFACTUAL_PRACTICAL_GROWTH_SIGNAL`

A statistically positive but smaller result must not be promoted to this status.

## Evidence and provenance

Every worker records:

- immutable Git commit and tree SHA
- clean tracked tree requirement
- CLM-0.1 SHA
- corpus/tokenizer hashes
- replicate seed
- training and validation schedule hashes
- decision checkpoint token
- probe and confirmation horizons
- analytic candidate table
- all birth parity records
- no-growth and candidate per-batch NLLs
- paired bootstrap intervals
- policy decision
- 500K confirmation

The publisher rejects mixed training commits/trees.

## Out of scope

CLM-0.3c does not yet implement:

- repeated autonomous births
- persistent online shadow branches
- expert death/merge
- learned probe horizons
- explicit hardware/energy pricing in the decision objective
- 2D topology
- multimodality
- online continual learning

If the counterfactual decision is calibrated, the next stage may turn the one-shot decision into a repeated local growth controller.