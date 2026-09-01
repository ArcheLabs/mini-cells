# MiniCells Continual-Learning Research Roadmap

Status: **causal funnel frozen; updated after Core 009B-1 positive**  
Current evidence frontier: `CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`  
Current experiment: **Core 009B-2 — Persistent Effect Geometry**

## Mission

The research program no longer assumes that knowledge is naturally partitioned into semantic Cells, hard functional modes, or a small fixed matrix dictionary.

The working product hypothesis is:

\[
\boxed{\text{slow/frozen foundation}+\text{fast persistent effect memory}+\text{learned addressability}+\text{safety certificates}+\text{adaptive growth}}
\]

The program must establish this hypothesis through a causal evidence funnel. A later stage is not opened when an earlier prerequisite fails. Until the full chain is established, repository documentation must use **candidate CLM substrate**, not “confirmed CLM architecture”.

## Evidence established so far

1. Core 005/006 support replay-free local subspace certificates and growth-restored plasticity.
2. Core 006/007 do not support semantic or hard discrete functional Cell boundaries.
3. Core 008 rejects a small fixed shared matrix basis; individual writes are nevertheless nearly rank-1.
4. Core 009A supports raw asymmetric factor geometry.
5. The 009A right-collapse bridge shows the dominant right direction is mainly the train-token activation mean/representation carrier, not sequence averaging.
6. Core 009B-1 formally supports **carrier causal sufficiency**: with `rho=0.01` locked using full-write discovery only, 3/3 untouched confirmation seeds passed. Carrier-only writes preserved roughly 97.6–98.2% of full target gain, residual-only writes contributed roughly 1.9–2.5%, and excess unrelated harm was negligible.

The active decomposition is therefore:

\[
G_i\approx a_i\mu^\top,\qquad a_i=\hat G_i r\in\mathbb R^{64}.
\]

The next question is whether the effect vectors form a compact reusable persistent coordinate system.

## Evidence funnel

### Core 009B-1 — Carrier Causal Sufficiency

**Status: formally supported.**

Result: `CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`. This opens 009B-2.

### Core 009B-2 — Persistent Effect Geometry

**Status: current frontier.**

Test whether heldout effects admit reusable coordinates:

\[
\boxed{a_i\approx V\beta_i}.
\]

Two independent requirements are mandatory.

**Offline compact shared geometry:** fit an origin-preserving shared linear subspace from train effects only. Primary evidence is heldout normalized effect residual by dimension, train→heldout generalization gap and covariance energy. A full 64-dimensional basis is not evidence; only dimensions <=32 can unlock confirmation.

**Online low-growth span:** start with no coordinates and add one only when the current effect cannot be reconstructed below the frozen residual threshold. Primary evidence is `K(N)`, final coordinate count, new coordinates per 100 writes, late-half growth, independent-memory compression `N/K`, heldout residual and robustness across multiple frozen stream orders.

The critical scalability signal is **compact heldout reconstruction plus late growth collapse**, not a raw reuse count and not the trivial 64-dimensional ambient ceiling.

If negative, persistent effect dictionaries are not a scalable CLM substrate and 009B-3 is not opened.

### Core 009B-3 — Deployable Effect Addressability

Opened only if 009B-2 is positive.

Training oracle: \(\beta_i=V^\top a_i\).

Deployable requirement: \(f_\theta(x_i)\to\hat\beta_i\) using inference-visible context only. Gradients, target labels, future tokens, oracle modes and replay targets are forbidden router inputs.

Test ridge, then a small MLP, then tiny attention only if required. Primary metric is causal effect recovery, not coefficient MSE.

### Core 010 — Certified Persistent Effect Memory

Opened only if all of Core 009B is positive.

For effect coordinates `V` and historical coefficient dependencies `beta_j`:

\[
Q_\beta=\operatorname{basis}\operatorname{span}\{\beta_j\},\qquad \Delta VQ_\beta=0.
\]

Compare unsafe, certificate/no-growth, certificate+growth and replay oracle. Growth is triggered only when the desired write is infeasible under existing dependency constraints.

### Core 011 — Transfer and Product Validation

Opened only if Core 010 is positive. Test another dense modern LM, at least one MoE, multiple insertion layers, longer streams and comparison with LoRA/adapters, replay and conventional MoE adaptation. A Pythia-only positive is not sufficient for product architecture.

## Product threshold

The CLM core representation is considered experimentally supported only when untouched confirmation establishes:

\[
\begin{aligned}
G_i&\approx a_i\mu^\top &&\text{causally sufficient}\\
a_i&\approx V\beta_i &&\text{compact/reusable}\\
\beta_i&\approx f_\theta(x_i) &&\text{deployably addressable}\\
\Delta VQ_\beta&=0 &&\text{safe}\\
P_{\rm free}\approx0&\Rightarrow\text{growth} &&\text{plastic}\\
K(N)&\ll N &&\text{scalable}
\end{aligned}
\]

Current position:

\[
\boxed{\underbrace{G_i\rightarrow a_i\mu^\top}_{\text{supported}}\Longrightarrow\underbrace{a_i\rightarrow V\beta_i}_{\text{009B-2}}}
\]

## Experimental discipline

- Every confirmatory result uses untouched seeds.
- Discovery and confirmation are separated by a committed lock artifact.
- Failed scientific gates are preserved; no silent seed replacement.
- Diagnostic bridges never overwrite source scientific decisions.
- Causal interventions take priority over normalized geometry when they disagree.
- No router/certificate/growth mechanism is added before the representation prerequisite it depends on is confirmed.
- Ambient-dimensional saturation is not called bounded growth.
- Selection/reuse counters are never primary scalability evidence.
- Online results must be robust to more than one frozen stream order.
