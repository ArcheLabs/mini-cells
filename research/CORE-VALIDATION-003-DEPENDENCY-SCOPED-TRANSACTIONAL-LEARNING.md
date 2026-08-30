# Core Validation 003 — Dependency-Scoped Transactional Continual Learning

## Research transition

Core Validation 002–002C narrowed the write-addressability question:

- a single learned latent often carries a strong functional signal,
- wider sparse assemblies do not improve it,
- but few-shot precise write estimation remains unstable.

003 therefore stops asking whether every new fact can be converted into a perfect parameter delta. It tests a different systems hypothesis:

\[
\boxed{
\text{local candidate training}
\rightarrow
\text{dependency-scoped validation}
\rightarrow
\text{atomic commit / rollback}
}
\]

The primary question is:

\[
\boxed{
\textbf{Can sparse routing turn global continual-learning safety into a local validation problem?}
}
\]

Prior 002-series decisions remain frozen and cannot be rescued by 003.

## Five hypotheses

### H1 — structural locality

For frozen routing and frozen shared parameters, let \(B_t\) be the set of expert blocks modified by transaction \(t\). For a historical input \(x\),

\[
R(x)\cap B_t=\varnothing
\]

should imply, up to the explicit numerical tolerance used by the evaluator,

\[
f_{\theta_t'}(x)=f_{\theta_t}(x).
\]

This is stronger than a statistical locality claim: the modified parameters are absent from the input's computation path.

### H2 — dependency-scoped validation

The old-regression set is

\[
V(B_t)=\{x\in V_{\rm old}:R_{\theta_t}(x)\cap B_t\neq\varnothing\}.
\]

The commit decision sees only \(V(B_t)\) plus an independent new-data validation set. A hidden evaluator also runs the full old set. The primary safety metric is

\[
\mathrm{FSR}
=
P(\text{local PASS}\land\text{global FAIL}\mid\text{local PASS}).
\]

### H3 — transactional benefit

`local_always` and `local_tx_frozen` use the same local update rule. The only mechanism change is commit policy. 003 asks whether rejecting locally unsafe candidates lowers cumulative positive old-regression damage without discarding most useful new learning.

### H4 — granularity benefit

At granularity \(g\),

\[
N_{\rm experts}=4g,
\qquad
d_{\rm expert}=64/g.
\]

Each context still routes to two blocks. Hidden-width-dependent expert parameter budget is therefore held approximately constant while individual state blocks shrink.

The main scope metric is

\[
\Gamma(g)
=
E_t\frac{|V(B_t)|}{|V_{\rm global}|}.
\]

The experiment does **not** hide the fact that logical active expert compute also falls as \(g\) increases. Instead it reports the resulting cost/plasticity frontier.

### H5 — stable routing stress

A separate `local_tx_router_drift` control perturbs the router after local training but before validation. The dependency set remains the pre-update set. This is deliberately a causal stress test, not a model of a specific router optimizer.

If stale routing causes scope-external changes or false-safe commits, it supports the architectural requirement:

\[
\boxed{
\text{stable addressing plane}+\text{mutable expert state plane}
}
\]

H5 is diagnostic only and does not veto the frozen-router primary mechanism.

## Favorable synthetic world

There are 64 contexts. Sixteen are immutable anchor contexts; 48 receive continual updates.

Every context begins as a convex composition of two functions selected from four shared nonlinear functional families:

\[
y_c(u)
=
\alpha_c f_{a(c)}(u)
+
(1-\alpha_c)f_{b(c)}(u).
\]

Each mutable context additionally has a private residual function \(r_c(u)\). Before continual learning its amplitude is zero. A transaction changes

\[
\lambda_c
\leftarrow
\lambda_c+0.35
\]

and the new truth is

\[
y_c'(u)=y_c(u)+\lambda_c r_c(u).
\]

The same 48 mutable contexts are traversed in two shuffled epochs, giving 96 transactions and two updates per mutable context.

Each transaction receives exactly 32 training examples. The optimizer reuses that finite set for 24 steps; it does not silently sample additional training data.

## Structured sparse routing

Experts are grouped by the four shared functional families. Granularity adds independently routed replicas within each family.

A context has a fixed random routing hash. For each of its two functional families it selects one replica, so every forward pass uses exactly two expert blocks.

This is intentionally favorable. 003 is not testing whether a language model can discover these domains by itself; it asks whether the proposed local-validation mechanism works **once sparse functional routing exists**.

## Variants

### `standard_moe_always`

Updates shared trunk/head plus routed experts and always commits. This is the contextual end-to-end MoE baseline.

### `local_always`

Freezes shared state and routing, updates only the two routed expert blocks, and always commits.

### `local_tx_frozen`

Uses the same local candidate update as `local_always`, but commits only if:

\[
G_{\rm new}\ge\tau_{\rm new}
\]

and

\[
D_{\rm local}\le\epsilon_{\rm old}.
\]

The full-history evaluator is never consulted by the commit rule.

### `local_tx_router_drift`

Uses the same local candidate, then perturbs the router before validation. The local dependency set remains the stale pre-update set.

## Evaluation

For old behavior,

\[
D(S)
=
\frac{
\mathrm{MSE}_{\rm after}(S)-\mathrm{MSE}_{\rm before}(S)
}{
E_S[y^2]
}.
\]

For new learning, normalization uses the energy of the actual incoming change:

\[
G_{\rm new}
=
\frac{
\mathrm{MSE}(f_{\rm before},y_{\rm new})
-
\mathrm{MSE}(f_{\rm candidate},y_{\rm new})
}{
E[(y_{\rm new}-y_{\rm old})^2]
}.
\]

This avoids making a small but correctly learned update look weak merely because the pre-existing target has much larger energy.

The experiment records:

- false-safe rate,
- dependency coverage,
- structural escape rate,
- routing drift rate,
- acceptance rate,
- cumulative positive global regression damage,
- cumulative committed new-learning gain,
- final anchor-context normalized MSE,
- final mutable-context normalized MSE,
- candidate-state fraction,
- normalized state + validation cost proxy per accepted update.

## Frozen formal gates

A granularity \(g>1\) passes a seed only if all of the following hold:

- base normalized MSE \(\le 0.10\),
- false-safe rate \(\le 0.01\),
- mean dependency coverage \(\le 0.20\),
- maximum structural escape rate \(=0\),
- acceptance rate \(\ge 0.50\),
- cumulative positive global-regression damage \(\le 0.70\times\) `local_always`,
- committed new-learning gain \(\ge 0.60\times\) `local_always`,
- dependency coverage \(\le 0.50\times\) the \(g=1\) coverage.

Formal seeds are:

```text
80301
80302
80303
```

The final positive decision requires **the same fixed granularity** to pass every gate on all three seeds.

Formal statuses are only:

```text
DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_SUPPORTED
DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED
```

Smoke runs emit only `SMOKE_ONLY`.

## Interpretation

A positive result supports the mechanism claim that sparse routing can convert continual-learning regression safety into a dependency-scoped transactional problem in this favorable world.

It does not establish:

- that natural-language MoE routing already has these dependency properties,
- that router learning itself is solved,
- that the chosen granularity is language-scale optimal,
- that full global evaluation can be removed without maintaining high-quality historical probes,
- or that JAM is required for the neural mechanism.

The JAM-relevant consequence would instead be architectural: model updates can be represented as speculative changes to small state units, locally evaluated, then atomically committed or discarded.
