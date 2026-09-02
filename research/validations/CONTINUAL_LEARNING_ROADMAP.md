# MiniCells Continual-Learning Research Roadmap

Status: **two-track roadmap; Constructive CLM is the native-CLM main line**  
Frozen evidence map: [`CLM_FEASIBILITY_EVIDENCE_MAP.md`](CLM_FEASIBILITY_EVIDENCE_MAP.md)  
Current constructive experiment: **Constructive CLM-001B — Latent Coordinate Discovery under Superposition**  
Current foundation-interface diagnostic: **Core 009D — Compositional Operator Geometry**

## Mission

The product path is:

\[
\boxed{
\text{pretrained LLM}
\rightarrow
\text{external CLM layer}
\rightarrow
\text{hybrid CLM}
\rightarrow
\text{endogenous/native CLM}
}
\]

The research program separates two questions:

1. **Foundation Interface Research** — what writable structure already exists in a mature pretrained LLM and what is the cheapest causal interface for an external CLM layer?
2. **Constructive CLM Research** — can persistent sparse functional coordinates, read addressability and growth be learned during continual experience even when the pretrained foundation does not already contain a deployable natural Cell ontology?

A negative result in Track A does not stop Track B.

The canonical evidence-reuse and no-repeat rules are frozen in the CLM Feasibility Evidence Map. New experiments must cite the missing integration claim they test rather than re-proving already-settled components.

---

# Track A — Foundation Interface Research

## Purpose

Track A characterizes a mature LLM as a substrate for an external CLM layer. It is not responsible for proving that a native CLM exists naturally inside the checkpoint.

The surviving product-level hypothesis is:

\[
\boxed{
\text{slow/frozen foundation}
+
\text{persistent mutable modules}
+
\text{learned addressability}
+
\text{safety certificates}
+
\text{adaptive growth}
}
\]

## Evidence established

1. Core 005 formally supports replay-free Cell-local subspace certificates and reusable growth in the registered linear-writable world.
2. Core 006 shows that real Pythia hidden states do not immediately saturate, that functional reuse exists, and that replay-free certificates reduce registered forgetting while retaining roughly 0.84–0.89x replay new-learning gain.
3. Core 006 also rejects the assumption that routing/semantic address is automatically the correct functional mitosis boundary; address-based splitting produced poor conflict relief and excessive growth.
4. Core 008 rejects a small fixed shared matrix basis while showing that individual normalized writes are nearly rank-1.
5. Core 009A formally supports asymmetric factorized functional geometry.
6. The 009A right-collapse diagnostic shows that the dominant right side is largely a common representation/activation carrier.
7. Core 009B-1 formally supports carrier causal sufficiency: carrier-only writes retain roughly 97.6–98.2% of full target gain at the locked causal scale, while the residual contributes only a few percent.
8. Core 009B-2 discovery finds no viable compact persistent global carrier-effect subspace under the frozen model; confirmation is forbidden.
9. Core 009C discovery finds neither the frozen sparse overcomplete carrier-effect dictionary nor centroid-routed local subspace geometry; confirmation is forbidden.

The resulting interface interpretation is:

\[
\boxed{
\text{the pretrained model exposes a simple useful write interface}
\neq
\text{the pretrained model exposes a ready-made Cell ontology}
}
\]

## Core 009D — Compositional Operator Geometry

009D remains useful and may be completed. It asks whether the **full normalized write operator** retains reusable joint structure that was lost by the carrier projection.

A positive 009D result may improve:

- external CLM write compression;
- initialization of constructive coordinates;
- operator-level composition priors;
- later deployable router design.

A negative 009D result closes the tested natural operator-organization hypothesis only. It does **not** stop Constructive CLM.

## Track-A stop boundary

Do not continue indefinitely searching for near-equivalent hidden natural Cell ontologies. New Track-A work must have a concrete product/interface payoff, such as:

- lower write state;
- better causal fidelity;
- cross-model/layer stability;
- deployable context-to-write prediction;
- lower inference/write cost.

---

# Track B — Constructive CLM Research

## Core question

\[
\boxed{
\text{Can continual pressure produce a reusable, addressable, protected and growing functional coordinate system?}
}
\]

This track does not require:

\[
\exists\text{ a natural global Cell partition inside the pretrained LLM}.
\]

Instead, the system is allowed to **learn the coordinate system itself**.

## Reused mechanism stack

Track B treats the following as reusable components rather than new research questions:

```text
sparse routed mutable state
+
transactional candidate/commit lifecycle
+
growth as a plasticity escape
+
replay-free Cell-local protection
+
foundation writable interface
```

Constructive CLM-001 now adds a first formal constructive component:

```text
structured continual experience + singleton exposure
  -> learned Cell keys/effects
  -> deployable read addressability
  -> pair composition
  -> growth stops after factor coverage
```

The active missing bridge is whether that result survives removal of the clean singleton prototype scaffold.

## G1a — Addressable learned coordinate formation

Status: **SUPPORTED**.

Experiment:

**Constructive CLM-001 — Learned Coordinate Formation**

Formal result:

```text
status = LEARNED_COORDINATE_FORMATION_SUPPORTED
seeds = 90111 / 90112 / 90113
Cells = 6 / 6 / 6
pair route recall = 1.0 / 1.0 / 1.0
late spawns = 0 / 1 / 1
```

This establishes that reusable Cell keys/effects can form without task/factor labels in the registered controlled world.

It does **not** establish latent factor discovery under superposition because every hidden factor first had singleton exposure.

Do not rerun the singleton world as a new core validation.

## G1b — Latent coordinate discovery under superposition

Current experiment:

**Constructive CLM-001B — Latent Coordinate Discovery under Superposition**

Decision question:

> Can reusable latent Cell keys/effects be recovered when no hidden factor is ever presented alone, and can those recovered Cells address completely unseen compositions from `x` alone?

The registered 001B world deliberately removes singleton exposure while keeping the discovery problem identifiable and falsifiable:

- six correlated, non-orthogonal latent context/effect factors;
- training contains only equal-weight pair superpositions;
- three pair types are held out entirely;
- no singleton or triple appears during training;
- learner receives no factor labels, pair labels or hidden factor count;
- online transaction means form mixture prototypes;
- pairwise prototype geometry induces a learned overlap graph;
- largest maximal cliques recover latent star incidence;
- the learned incidence system is solved for latent Cell keys/effects;
- heldout inference uses `x` only;
- heldout pairs use coefficients not seen during discovery;
- every triple is unseen during training and tested compositionally.

Controls compare against nearest transaction memory and a shuffled effect-address baseline.

A positive result supports G1b **only for this registered additive pair-superposition family**. It does not establish arbitrary blind source separation, unknown arity or nonlinear composition.

## G2 — Long-horizon growth law

Open only after G1b positive.

Primary question:

\[
\boxed{K(N)=o(N)?}
\]

The first practical standard is weaker but measurable:

\[
\frac{K(N)}{N}\downarrow,
\qquad
P(\text{spawn}\mid t)\downarrow,
\qquad
\text{reuse rate}\uparrow.
\]

Constructive CLM-002 must vary stream length and order and report:

- `K(N)`;
- `K(N)/N`;
- late-window spawn probability;
- Cell lifetime;
- reuse count distribution;
- heldout performance as memory grows.

Do not infer asymptotic sublinearity from a short fixed stream.

## G3 — Learned coordinates + existing protection

Constructive CLM-003 integrates the learned coordinate/read system with the already-supported Core-005 certificate logic.

This is an **integration** test, not a new certificate-principle test.

Required comparison:

```text
learned coordinates + unsafe writes
learned coordinates + certificate/no-growth
learned coordinates + certificate/growth
replay oracle
```

Primary question:

> Does protection preserve the learned coordinate system without collapsing plasticity or forcing near-linear growth?

## G4 — Multi-Cell composition

001 and 001B contain controlled algebraic composition tests, but G4 remains a model-level execution question.

The model must activate multiple learned Cells on unseen combinations and demonstrate that the combined behavior is not merely nearest-memory retrieval.

Required metrics include:

- unseen combination accuracy/loss;
- cross-Cell destructive interference;
- route support recovery;
- superposition/composition residual;
- compute versus active Cell count.

Representation-level operator composition from 009D is informative but is not a substitute for model-level Cell composition.

## G5 — External -> endogenous transition

Only after the constructive core is stable.

Remove scaffolding one component at a time:

```text
prototype / relational-discovery scaffold
  -> learned router
  -> learned write controller
  -> learned growth controller
  -> slow-plastic foundation
  -> endogenous cellular model
```

Each removal is accepted only if it preserves or improves:

- new learning;
- old retention;
- growth efficiency;
- routing generalization;
- composition;
- compute/storage cost.

This creates a continuous product-to-native path rather than a second unrelated model project.

---

# Product Track — External CLM Layer

Product feasibility is a separate decision from Native CLM feasibility.

The first product can remain:

\[
\boxed{
\text{mature frozen/slow LLM}
+
\text{persistent sparse mutable Cells}
+
\text{router}
+
\text{certificate}
+
\text{growth}
+
\text{version/rollback lifecycle}
}
\]

It may use engineered LoRA/rank-1/low-rank Cells and a practical router. Native-CLM research is allowed to fail without invalidating this product.

The product benchmark must eventually compare against:

- RAG/external memory;
- ordinary LoRA/PEFT;
- continual adapters;
- replay;
- adapter banks / MoE-style routing.

The differentiating value must include parameterized behavior/skill adaptation, not factual storage alone.

---

# Experimental discipline

- Frozen formal seeds are never silently replaced.
- Discovery/diagnostic results are not promoted to formal confirmation claims.
- Negative natural-geometry results do not automatically become Native-CLM No-Gos.
- Hidden semantic/task/factor labels may be used for post-hoc evaluation only unless explicitly registered as learner input.
- An existing supported mechanism is not re-tested unless a new integration variable is named in advance.
- Growth evidence must report state growth and reuse, not only task accuracy.
- Short-stream bounded growth is not described as asymptotically sublinear.
- Coordinate recovery is not sufficient without deployable read addressability.
- Read addressability is not sufficient without write safety and long-run growth control.
- A positive 001B result must not be generalized beyond its registered pair-superposition scaffold.
- Rank-1-per-write compression is not evidence of cross-write Cell reuse.
- Pythia-only evidence is not sufficient for product generalization.
- The formal research stop rule is the one frozen in the CLM Feasibility Evidence Map.

## Immediate order

```text
1. Constructive CLM-001 = formally supported; freeze it as G1a parent evidence.
2. Run Constructive CLM-001B on untouched formal seeds 90211/90212/90213.
3. 009D may proceed in parallel as a non-blocking Foundation Interface diagnostic.
4. If C-CLM-001B is positive -> C-CLM-002 long-horizon growth law.
5. Then integrate Core-005 protection in C-CLM-003.
6. Then model-level composition and external->endogenous transition.
```
