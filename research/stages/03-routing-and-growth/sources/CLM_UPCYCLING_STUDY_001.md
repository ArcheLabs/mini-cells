# CLM Upcycling Study 001 — Inherit Then Differentiate

## Scientific motivation

CLM v1 showed that arbitrary hidden-channel segmentation is function-preserving only when all shards remain active; removing even one of eight shards caused an approximately 7.5% PPL regression. The result is interpreted as **arbitrary decomposition does not imply substitutability**.

CLM v2 replaced shards with an overcomplete random conditional program bank. Local imitation learned a strong one-step approximation, and closed-loop homotopy recovered a substantial fraction of the direct scaffold-removal gap, but scaffold-free K6 plateaued near 1.098x teacher PPL. This isolates a new bottleneck: **knowledge inheritance into randomly newborn programs**.

The next question is therefore:

> Can a routed cellular model inherit the pretrained dense local computation exactly first, and differentiate only afterward?

This study tests that question without capability labels, semantic expert labels, phenotype, cell sparsity, or topology changes.

## Core invariant

For every NCA stage, create `M=4` full-width routed experts by exact copying of the pretrained dense FFN:

\[
E_1 = E_2 = E_3 = E_4 = F_D.
\]

Use hard local top-1 routing:

\[
e^*(p)=\arg\max_e R_e(p),
\qquad
F_U(p)=E_{e^*(p)}(p).
\]

At initialization, for every local perception `p` and every possible route:

\[
F_U(p)=F_D(p).
\]

Therefore dense-to-routed conversion is exactly function-preserving **without a scaffold and without imitation**.

Total expert capacity is 4x the original dense FFN, while active expert capacity per local update remains 1x dense FFN.

## Formal arms

All arms begin from the same Experiment 006 `minicells-v2-10m.pt` checkpoint and receive the same continuation examples in the same order within each replicate.

### Dense continued

The original TextNCA is continued for 1M tokens. This controls for the benefit of simply training the checkpoint longer.

### Copy-Random

Four exact dense FFN copies are created per stage. The local router uses four randomly initialized learnable cosine prototypes. Initial routed execution is exactly equal to the pretrained dense model because all experts are identical.

### Copy-Geometry

The expert initialization is identical to Copy-Random. The only difference is router initialization.

For each frozen teacher stage, collect unlabeled local perceptions

\[
p_i^\tau = \operatorname{norm\_ffn}(h_i^\tau + \Delta_i^\tau)
\]

from a dedicated calibration subset. Run deterministic cosine k-means with `K=4`. The resulting centroids initialize the four router prototypes for that stage.

No token, task, capability, or semantic labels enter clustering.

Thus the primary geometry comparison isolates:

\[
\text{random local partition}\quad\text{vs}\quad\text{pretrained local-state partition}.
\]

It does **not** conflate router geometry with different expert functions.

## Router

Both routed arms use the same strictly pointwise router:

\[
R_e(p)=s\,\frac{p}{\|p\|}\cdot\frac{c_e}{\|c_e\|},
\qquad s=4.
\]

Forward routing is hard top-1. Backward routing uses a straight-through softmax estimator.

The router cannot inspect batch aggregates, sequence aggregates, task IDs, labels, or nonlocal cell states.

## Training

Each formal arm receives 1,000,000 continuation tokens in four 250K blocks.

- optimizer: AdamW
- learning rate: `1e-4`
- betas: `(0.9, 0.95)`
- weight decay: `0.1`
- gradient clipping: `1.0`
- teacher: frozen Experiment 006 TextNCA
- distillation weight: `0.5`
- upcycled-router balance weight: `0.01`

All inherited backbone parameters and routed experts may continue training. The dense arm uses the same optimizer schedule, except it has no routing-balance term.

## Geometry calibration

- four cosine clusters per NCA stage;
- at most 8192 local-perception samples per stage;
- dedicated deterministic validation starts distinct from formal scoring starts;
- clustering is label-free;
- cluster occupancy and normalized occupancy entropy are recorded.

## Mandatory Stage-0 parity

Before continuation, both Copy-Random and Copy-Geometry must independently satisfy:

\[
|PPL_U/PPL_D-1|\le 10^{-5},
\]

\[
\max|\Delta\text{logits}|\le5\times10^{-5},
\]

\[
\max|\Delta h|\le10^{-6}.
\]

Any parity failure yields:

`CLM_UPCYCLING_EQUIVALENCE_FAILURE`.

## Final routing controls

After 1M continuation tokens, each routed arm is evaluated on an independent formal validation split.

### Dynamic

Normal local top-1 routing.

### Static

For each recurrent route slot, choose the most-used expert on an independent calibration split and freeze that expert choice for all formal-evaluation samples and positions. This is a stronger control than a single global expert because it allows deterministic stage/step preferences but no input-dependent routing.

### Shuffled

Record dynamic masks, then permute masks only across the batch/sample dimension. Token position, recurrent step, active expert count, expert usage, and compute are preserved. Three deterministic permutations are averaged.

## Metrics

### Quality relative to matched dense continuation

\[
Q=\frac{PPL_{dynamic}}{PPL_{dense\ continued}}.
\]

Quality-safe threshold:

\[
Q\le1.03.
\]

### Routing variation

Formal sample-level route variation must satisfy:

\[
D_{sample}\ge0.05.
\]

Position and recurrent-step variation are reported separately.

### Causal routing value

\[
A_{static}=\frac{L_{static}-L_{dynamic}}{L_{dense}},
\]

\[
A_{shuffled}=\frac{L_{shuffled}-L_{dynamic}}{L_{dense}}.
\]

Both must satisfy:

\[
A\ge0.002.
\]

### Utilization

Normalized hard-usage entropy must be at least:

\[
H_{usage}\ge0.80.
\]

This prevents a nominal MoE success from being explained by collapse to one expert.

### Differentiation telemetry

The study reports mean pairwise expert cosine similarity and pairwise relative L2 distance after every 250K continuation block. These are diagnostics, not semantic expert labels.

## Aggregate statuses

### `CLM_UPCYCLING_EQUIVALENCE_FAILURE`

Any Stage-0 routed conversion loses exact pretrained function.

### `CLM_UPCYCLING_QUALITY_FAILURE`

Neither Copy-Random nor Copy-Geometry is quality-safe in at least 2/3 replicates.

### `CLM_UPCYCLING_QUALITY_SIGNAL`

At least one upcycling method is quality-safe in at least 2/3 replicates, but no method satisfies the preregistered causal-routing criteria in at least 2/3 replicates.

This is already a positive knowledge-inheritance result: a routed model replaced the dense FFN without the approximately 10% random-newborn handoff penalty.

### `CLM_UPCYCLING_CONDITIONALITY_SIGNAL`

At least one upcycling method satisfies, in at least 2/3 replicates:

- quality ratio <= 1.03;
- sample variation >= 0.05;
- static advantage >= 0.002;
- shuffled advantage >= 0.002;
- usage entropy >= 0.80.

### Geometry advantage

`geometry_advantage=true` only when Copy-Geometry reaches the full conditionality signal in >=2/3 replicates while Copy-Random does not. Geometry superiority is deliberately secondary to the primary upcycling question.

## Interpretation boundaries

A positive result may establish:

- exact function-preserving dense-to-routed cellular upcycling;
- successful pretrained-knowledge inheritance;
- expert differentiation under route-conditioned experience;
- causal local-state-dependent routing if the controls pass.

It must not be described as capability discovery, semantic expert formation, tissue formation, phenotype formation, self-growth, or cell specialization.

## Historical random-newborn control

The preceding CLM v2 result is retained as historical evidence rather than rerun as a formal arm:

- random newborn overcomplete programs;
- local imitation succeeded strongly;
- closed-loop homotopy reduced the direct scaffold-removal gap;
- final scaffold-free K6 plateaued near 1.098x teacher PPL.

The present experiment asks whether **inherit first, then differentiate** removes that knowledge-transfer bottleneck.
