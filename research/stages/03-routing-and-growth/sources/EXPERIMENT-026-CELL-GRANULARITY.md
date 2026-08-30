# Experiment 026 — 30M Cell Granularity under Local Plasticity

## Question

Does the current CLM fail to form clear cellular phenotypes partly because one program cell is too coarse?

Experiment 026 compares G={1,2,4,8} micro-cells per existing FFN tissue while preserving the same retained 30M TextNCA source, the same fixed 12-root ProgressiveGrowthCLM, the same trainable parameter count and the same age-zero function. Persistent birth is disabled.

A pure exact partition would only regroup hidden units: without a cell-level update rule, all G arms would follow almost the same mathematical training trajectory. Experiment 026 therefore applies one identical **local-plasticity rule** to every arm. G is the resolution at which that rule can act.

The causal question is:

> With capacity, age-zero function, tissue router, data, global LR schedule and mean tissue LR fixed, does finer local adaptation resolution produce additional stable functional differentiation?

## Developmental environment

Every arm starts after the same 100M-token TinyStories source experience and receives 20M continuation tokens from a persistent balanced environment:

- **Story** — TinyStories;
- **Math** — the six-family controlled integer arithmetic generator already used in Experiment 025;
- **Symbolic** — deterministic sequence reverse, sort and elementwise-offset transformations;
- **Facts** — deterministic synthetic key/value QA examples.

The latter three are controlled synthetic environments. They are not broad reasoning or factual-knowledge benchmarks.

Every four training steps contain exactly one step from every domain. The order inside each four-step block is deterministically shuffled. All G arms therefore receive the same domain schedule and the same sampled examples.

Evaluation checkpoints are 0, 1M, 2M, 5M, 10M, 15M and 20M continuation tokens.

## Exact structural control

Each original FFN tissue is decomposed along its hidden dimension:

\[
f(x)=\sum_{i=1}^{G} W_{2,i}\sigma(W_{1,i}x)+b.
\]

At age zero this is the same function as the original FFN up to floating-point summation order. The partition does not add trainable parameters.

Across G={1,2,4,8}, the experiment fixes:

1. source checkpoint and 100M prior experience;
2. 12 root tissues and tissue-level hierarchical router;
3. trainable parameter count;
4. age-zero forward function;
5. continuation data and exact sample schedule;
6. AdamW, global LR schedule, weight decay and clipping;
7. next-token cross-entropy objective;
8. local-plasticity equation and hyperparameters;
9. mean local-plasticity multiplier within each tissue at 1.0;
10. no persistent division.

## Local plasticity

Every micro-cell starts at multiplier:

\[
p_i(0)=1.
\]

After backward, the cell gradient RMS is compared only with its immediate tissue neighbours. Let

\[
r_i=\frac{g_i+\epsilon}{\operatorname{mean}_{j\in N(i)}g_j+\epsilon}.
\]

The unsmoothed target is:

\[
q_i=\operatorname{clip}(r_i^{1/2},0.5,2.0).
\]

Targets are renormalized to tissue mean 1.0, then updated with EMA decay 0.95. The effective cell LR is:

\[
\eta_i(t)=\eta_{global}(t)p_i(t).
\]

The final multipliers are again mean-normalized inside each tissue. Thus finer G does not receive a larger average learning rate; it receives finer **local control resolution**.

For G=1 there is no distinct neighbour comparison. The rule deliberately degenerates to:

\[
p_1(t)=1.
\]

This makes G=1 the non-local-plasticity baseline without changing the global optimizer.

## Measurements

### Performance

At every checkpoint:

- Story/Math/Symbolic/Facts validation NLL and PPL;
- balanced four-domain NLL;
- geometric-mean PPL;
- elapsed time and peak VRAM.

### Functional phenotype

A fixed held-out probe captures the first invocation of each tissue for each domain. For micro-cell i, the RMS contribution across the four domains produces profile \(p_i\).

Specialization is:

\[
D_i=1-\frac{H(p_i)}{\log 4}.
\]

`0` is domain-uniform; values closer to `1` are more selective.

Because a finer partition can reveal heterogeneity even before training, the formal comparison does **not** use absolute final specialization. For each arm:

\[
\Delta D_G=D_G^{final}-D_G^{age0}.
\]

The causal quantity is then:

\[
\Delta\Delta D_G=\Delta D_G-\Delta D_{G=1}.
\]

The experiment also records:

- output-projection gradient phenotype cosine/conflict across domains;
- profile cosine stability versus the same cell at age zero;
- within-tissue profile cosine redundancy;
- plasticity mean/variance;
- diagnostic stress.

Diagnostic stress is not used to alter topology or trigger division.

## Preregistered decision

A G>1 arm qualifies only if, at 20M continuation tokens:

1. its median specialization **gain** exceeds the G=1 gain by at least **0.05**;
2. its balanced NLL ratio versus G=1 is at most **1.02**;
3. its mean cell-profile cosine versus age zero is at least **0.80**.

If at least one finer arm qualifies and parameter/function parity checks pass:

`GRANULARITY_DIFFERENTIATION_SIGNAL`

Otherwise:

`NO_GRANULARITY_DIFFERENTIATION_SIGNAL`

A positive result means finer local adaptation resolution produced measurable additional stable phenotype differentiation under this frozen environment. It does not prove that the selected G is globally optimal.

## Formal outputs

The result bundle is `results/experiment-026-cell-granularity/` and contains the frozen `protocol.json`, run provenance, per-arm worker summaries and diagnostics, combined trajectory/final tables, `decision.json`, and three figures:

- `performance-by-granularity.png`;
- `differentiation-by-granularity.png`;
- `granularity-frontier.png`.

The frozen protocol is copied into the result directory **before training**, avoiding the publication-provenance failure found in Experiment 025.

## Kaggle target

Target: Tesla T4 ×2, 8-hour hard wall budget, 30-minute finalization reserve. Two arms run concurrently; incomplete arms rotate through 2.25-hour worker slices and resume automatically.

Canonical command:

```bash
python scripts/research/run_experiment_026_cell_granularity.py \
  --total-wall-hours 8 \
  --finalization-reserve-minutes 30 \
  --worker-slice-hours 2.25
```

Canonical notebook:

`research/notebooks/03-routing-and-growth/experiment-026-cell-granularity.ipynb`

It is intended for unattended **Save Version → Run All** and publishes only after every arm and the formal report complete.

## Interpretation boundary

Experiment 026 tests **granularity as the resolution of local adaptation**. It does not test autonomous mitosis, apoptosis, endogenous routing, learned topology or full NCA self-organization.

If 026 is positive, a clean next experiment can enable stress-driven function-preserving micro-cell fission while freezing the 026 environment and local-plasticity rule. If 026 is negative, adding division would be premature: the finer cells would not yet have demonstrated useful developmental differentiation.
