# CLM-0.3b — Marginal Growth Utility

## Question

Can a trained CLM that has entered a diminishing-return continuation regime gain measurable utility from one function-preserving lineage birth, and can a marginal-capacity WHERE signal choose a better parent than random selection?

CLM-0.3b does **not** change the CLM-0.3 growth operator. Hierarchical lineage routing, exact parent duplication, geometry initialization, optimizer-state inheritance, and birth equivalence remain fixed.

## Why 0.3b exists

CLM-0.3 established 6/6 pressure-growth birth equivalence, but did not establish growth utility or pressure selection. The fixed-4 control was still learning strongly at 1.5M continuation tokens, the second birth had only half the age of the first, and the preregistered pressure score `U * (1 + G)` was empirically dominated by utilization because gradient disagreement had low cross-expert variance.

0.3b isolates the unresolved variables:

1. Enter a preregistered diminishing-return regime before birth.
2. Permit exactly one birth.
3. Give every newborn the same 1M-token age window.
4. Replace pressure selection with an objective-sensitivity-based marginal score.
5. Remove the auxiliary root-router balance term from all formal arms.
6. Use a much larger causal validation sample and deterministic bootstrap confidence intervals.
7. Pin code semantics across resume.

## Frozen substrate

- Base release: CLM-0.1
- Base SHA-256: `87d36c408ae3873ffd567ebf17050661b42ddae2c8d5d1bab84b2c27c3c7e7a0`
- Recurrent substrate: unchanged GRUCell core
- Hierarchical lineage routing: unchanged
- Birth: `child = deepcopy(parent)`
- Child can only split traffic inside its parent lineage
- Geometry split: deterministic cosine k-means, k=2
- Optimizer: AdamW; existing state preserved; child inherits parent moments; split router receives fresh state
- Scheduler: continuous global continuation schedule; never restarted

## Formal matrix

Three paired replicates, seeds inherited from CLM-0.3:

- 55031
- 55032
- 55033

Arms:

1. `fixed4`
   - no birth
   - matched continuation through the same saturation and post-saturation token boundary
2. `marginal_growth`
   - one birth at the saturation boundary
   - parent selected by the 0.3b marginal-capacity score
3. `random_growth`
   - one birth at the same saturation boundary
   - parent selected randomly from the same eligible candidates
   - same deterministic geometry split

The old CLM-0.3 pressure score is still recorded for every candidate, but it is not a formal arm because 0.3 already tested it directly.

## Objective

All formal arms use:

`CE + 0.5 * KL(student || frozen CLM-0.1 teacher)`

Root-router auxiliary balance weight is fixed to `0.0` in the formal 0.3b matrix. This avoids using an objective that actively flattens utilization while simultaneously treating utilization as evidence of growth demand.

## Diminishing-return gate

Evaluation cadence: every 100K training tokens.

Formal validation size: 32 batches × 8 sequences × 125 target tokens = 32,000 target tokens per evaluation.

The earliest saturation boundary is 1.5M continuation tokens. The latest allowed pre-birth boundary is 3.0M tokens.

At each evaluation at or after 1.5M:

1. Take the latest five PPL observations (400K-token span).
2. Fit a least-squares line to `log(PPL)` versus units of 100K tokens.
3. Project the implied improvement over the next 500K tokens.
4. Declare the diminishing-return regime only when projected improvement is <= 2.0%.
5. The latest PPL may not be >1% worse than the first point in the window, preventing a temporary regression from masquerading as saturation.

The 2.0% gate is intentionally distinct from the final growth-utility success threshold. CLM-0.3 still showed substantial ordinary continuation gains near 1.5M tokens; requiring only <=0.5% projected continuation gain would make the regime gate unnecessarily close to a hard plateau and could prevent the experiment from testing marginal capacity at all. The formal utility endpoint below remains a stricter requirement that growth beat the matched fixed4 control by at least 0.5%.

If no diminishing-return boundary exists by 3.0M tokens, the replicate is recorded as `NO_SATURATION_REGIME`. No forced birth is allowed, and growth-utility claims are invalid for that matrix.

The pre-birth trajectory is identical across paired arms by construction; aggregation requires both the detected token and all pre-birth PPL observations to match across the three arms of a replicate.

## WHERE — marginal-capacity score

Calibration uses the 16 training microbatches immediately preceding birth.

For each eligible parent lineage `i`, record:

- `U_i`: routed utilization
- `G_i`: legacy gradient disagreement, retained only for diagnosis
- `P_old = U_i * (1 + G_i)`: legacy CLM-0.3 pressure
- `F_i`: Fisher-like gradient energy normalized by routed mass
- `S_i`: mean absolute weight-gradient saliency
- `D_i`: deterministic two-cluster cosine geometry separation

Definitions implemented in `research/minicells/growth_marginal.py`:

`F_i = mean_b ||g_i,b||^2 / max(U_i, eps)`

`S_i = mean_b mean_params |theta_i * g_i,b|`

`D_i = (1 - cos(c0, c1)) / 2`

The preregistered parent score is:

`M_i = sqrt(F_i * S_i) * (0.5 + 0.5 * D_i)`

The sensitivity terms determine objective importance. Geometry is deliberately bounded to a 0.5–1.0 multiplier: it can favor a lineage that admits a clean local split, but cannot make an objectively irrelevant lineage important by itself.

Eligibility still requires at least 512 routed perceptions.

## One-birth rule

0.3b permits exactly one birth. This removes the age confound in 0.3.

After birth, every formal worker continues exactly 1,000,000 training tokens. Newborn diagnostics are evaluated at matched ages:

- 0
- 100K
- 250K
- 500K
- 1M

## Newborn causal utility

The existing merge-back intervention remains:

- dynamic: route normally through parent/child lineage
- merge-back: force child-routed states back to the parent

Formal causal evaluation uses all 32 validation batches. In addition to the point estimate, 0.3b performs a deterministic 1,000-sample batch bootstrap and records a 95% interval for the relative merge-back penalty.

A positive lower CI bound at age 1M is treated as strong evidence that the newborn has independent causal utility. It is a secondary endpoint and is not used to redefine birth equivalence.

## Formal decisions

### 1. Paired pre-birth equivalence

For every replicate, all three arms must have matching evaluation token sets and numerically matching PPL values through the detected diminishing-return boundary:

`CLM_PAIRED_PREBIRTH_EQUIVALENCE`

A mismatch aborts formal aggregation rather than being interpreted as a growth effect.

### 2. Saturation regime

All three paired replicates must detect the same diminishing-return boundary across their three arms:

`CLM_SATURATION_REGIME_ESTABLISHED`

Otherwise:

`NO_SATURATION_REGIME`

### 3. Growth equivalence

All six growth-arm births (marginal + random, 3 replicates each) must pass the existing exact birth gate:

`CLM_GROWTH_EQUIVALENCE`

### 4. Marginal growth viability

All three marginal-growth newborns must, at age 1M:

- receive non-zero traffic
- have non-zero parameter divergence from the parent
- have non-zero split entropy

Causal sign is not used as a binary viability gate because 0.3 showed that a small validation sample can flip a near-zero merge-back estimate.

### 5. Marginal growth utility

For each replicate, compare final PPL at newborn age 1M against the paired fixed4 worker at the exact same global token count.

A replicate passes when:

`PPL_marginal / PPL_fixed4 <= 0.995`

Signal requires at least 2/3 replicates:

`CLM_MARGINAL_GROWTH_UTILITY_SIGNAL`

This 0.5% improvement endpoint is unchanged by the 2.0% diminishing-return gate.

### 6. Marginal selection

Compare marginal growth directly against random growth at matched replicate and token count.

Signal requires marginal PPL < random PPL in at least 2/3 replicates:

`CLM_MARGINAL_SELECTION_SIGNAL`

### 7. Strong causal utility

At age 1M, the merge-back bootstrap 95% lower bound must be >0 in at least 2/3 marginal-growth replicates:

`CLM_NEWBORN_CAUSAL_UTILITY_SIGNAL`

## Provenance and resume

Every worker records:

- exact Git commit
- Git tree SHA
- tracked-tree dirty flag
- CLM-0.1 SHA
- corpus/tokenizer hashes
- schedule hash
- validation schedule hash
- replicate seed
- all fixed formal experiment parameters encoded by the immutable training commit

These semantics are stored in checkpoints. Resume rejects any mismatch. A bug fix that changes the code commit therefore requires a new formal run rather than silently continuing an old one.

The publisher additionally requires all nine workers to report the same training commit and tree SHA before it will create a result branch.

## Out of scope

CLM-0.3b does not attempt:

- automatic production WHEN
- repeated progressive births
- expert death/merge
- 2D topology
- multimodality
- online continual learning
- sub-dense FLOP claims
- full balance-mechanism research

The next step after 0.3b depends on the result. Repeated growth should only resume after marginal utility is established under a diminishing-return regime.
