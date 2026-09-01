# MiniCells Continual-Learning Research Roadmap

Status: **frozen after Core 009A right-collapse bridge**  
Baseline evidence commit: `656317e8708f7fc7f64fcf438c299e93f15b8e82`

## Mission

The current research program no longer assumes that knowledge is naturally partitioned into semantic Cells, hard functional modes, or a small fixed matrix dictionary.

The working product hypothesis is now:

\[
\boxed{
\text{slow/frozen foundation}
+
\text{fast persistent effect memory}
+
\text{learned addressability}
+
\text{safety certificates}
+
\text{adaptive growth}
}
\]

The program must establish this hypothesis through a causal evidence funnel. A later stage is not opened when an earlier prerequisite fails.

## Evidence already established

1. **Replay-free safety is possible in a local subspace.** Core 005/006 support compressed dependency certificates of the form \(\Delta A Q=0\), with growth restoring plasticity when safe directions are exhausted.
2. **Discrete Cell boundaries are not supported.** Core 006 semantic splits and Core 007 write/interference modes did not produce stable bounded functional identities.
3. **A small fixed matrix basis is not supported.** Core 008 and its postmortem show that individual writes are nearly rank-1 while a small shared matrix basis does not cover heldout writes.
4. **Raw two-sided factor geometry is supported.** Core 009A formally supported a budget-matched `(left=56,right=8)` factorization on untouched confirmation seeds.
5. **The apparent right-side collapse is mainly a representation carrier.** The 009A bridge reproduced the source result exactly, but centering/mean-direction removal destroyed the near-one-dimensional right spectrum and whitening made it nearly isotropic. The raw right PC1 is almost identical to the train-token activation mean direction. Sequence averaging is not the cause.

The resulting decomposition to test is:

\[
G_i = a_i \mu^\top + E_i,
\]

where \(\mu\) is a common activation carrier, \(a_i\) is a context-dependent output-effect vector, and \(E_i\) is the non-carrier residual.

## Evidence funnel

### Core 009B-1 — Carrier Causal Sufficiency

Question:

> Does the common-carrier component \(G_\parallel=Grr^\top\), using the **same step size** as the full write, retain most of the actual causal target improvement without increasing unrelated harm?

Required before any effect-memory claim.

Variants:
- full: \(\Delta A=-\eta G\)
- carrier: \(\Delta A=-\eta Grr^\top\)
- residual: \(\Delta A=-\eta G(I-rr^\top)\)

The carrier \(r\) is the normalized mean projected activation fitted on training tokens only.

Discovery may choose only a perturbation magnitude from a frozen grid using **full-write stability/linearity only**. Carrier/residual outcomes are forbidden from influencing the scale lock.

If negative: stop the shared-carrier effect-memory route and redesign the functional metric around residual/context-specific effects.

### Core 009B-2 — Persistent Effect Geometry

Opened only if 009B-1 is positive.

Define:

\[
a_i = G_i r.
\]

Test whether heldout effects admit reusable coordinates:

\[
a_i \approx V\beta_i.
\]

Primary evidence:
- heldout effect reconstruction by dimension;
- online growth curve \(K(N)\);
- new coordinates per 100 writes;
- comparison with independent one-write-per-memory storage;
- dense vs sparse coefficients only after low-dimensional reuse is established.

The critical scalability signal is sublinear growth, not a selection/reuse counter.

If negative: persistent effect dictionaries are not a scalable CLM substrate.

### Core 009B-3 — Deployable Effect Addressability

Opened only if 009B-2 is positive.

Training oracle:

\[
\beta_i = V^\top a_i.
\]

Deployable requirement:

\[
f_\theta(x_i)\rightarrow \hat\beta_i
\]

using inference-visible context only. No gradient, target label, future token, oracle mode, or replay target may enter the router.

Models are tested in increasing capacity:
1. ridge;
2. small MLP;
3. tiny attention router only if required.

Primary metric is causal effect recovery, not coefficient MSE:

\[
\frac{\text{target gain from }V\hat\beta}
{\text{target gain from }V\beta}.
\]

If negative: the memory is not deployably addressable and is not a CLM.

### Core 010 — Certified Persistent Effect Memory

Opened only if all of Core 009B is positive.

For effect coordinates \(V\) and historical coefficient dependencies \(\beta_j\):

\[
Q_\beta=\operatorname{basis}\operatorname{span}\{\beta_j\},
\qquad
\Delta VQ_\beta=0.
\]

Compare:
- unsafe;
- certificate/no growth;
- certificate + growth;
- replay oracle.

Growth is triggered only when the desired write is infeasible under existing dependency constraints.

Primary evidence:
- forgetting;
- plasticity/replay;
- certificate rank and remaining safe dimension;
- \(K(N)\);
- new coordinates per 100 writes.

### Core 011 — Transfer and Product Validation

Opened only if Core 010 is positive.

Test:
- another dense modern LM;
- at least one MoE;
- multiple insertion layers;
- longer continual streams;
- comparison with LoRA/adapters, replay, and conventional MoE adaptation.

A Pythia-only positive is not sufficient for product architecture.

## Product threshold

The CLM core representation is considered experimentally supported only when the following chain is confirmed on untouched seeds:

\[
\begin{aligned}
G_i &\approx a_i\mu^\top && \text{causally sufficient}\\
a_i &\approx V\beta_i && \text{compact/reusable}\\
\beta_i &\approx f_\theta(x_i) && \text{deployably addressable}\\
\Delta VQ_\beta &=0 && \text{safe}\\
P_{\rm free}\approx0 &\Rightarrow \text{growth} && \text{plastic}\\
K(N) &\ll N && \text{scalable}
\end{aligned}
\]

Before this chain is complete, repository documentation must use **candidate CLM substrate** rather than “confirmed CLM architecture”.

## Experimental discipline

- Every confirmatory result uses untouched seeds.
- Discovery and confirmation are separated by a committed lock artifact.
- Failed scientific gates are preserved; no silent seed replacement.
- Diagnostic bridges never overwrite source scientific decisions.
- Causal interventions take priority over normalized geometry when they disagree.
- Whole-model NLL equality is not used as proof of functional equivalence in a weak-effect regime.
- No router/certificate/growth mechanism is added before the representation prerequisite it depends on is confirmed.
