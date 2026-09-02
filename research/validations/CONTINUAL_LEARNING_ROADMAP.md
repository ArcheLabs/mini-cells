# MiniCells Continual-Learning Research Roadmap

Status: **two-track roadmap; Constructive CLM is the native-CLM main line**  
Frozen evidence map: [`CLM_FEASIBILITY_EVIDENCE_MAP.md`](CLM_FEASIBILITY_EVIDENCE_MAP.md)  
Current constructive experiment: **Constructive CLM-002 — Long-Horizon Structure-Tracking Growth Law**  
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

The program separates:

1. **Foundation Interface Research** — characterize the cheapest useful writable interface already present in a mature LLM.
2. **Constructive CLM Research** — learn persistent sparse functional coordinates, routing, growth and eventually protection/endogenous control even when the pretrained checkpoint does not expose a natural Cell ontology.

A negative result in Track A does not stop Track B.

---

# Track A — Foundation Interface Research

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

## Reusable evidence

- Core 005: replay-free Cell-local subspace certificates and reusable growth are formally supported in the registered linear-writable world.
- Core 006: real Pythia hidden states retain useful replay-free plasticity, while routing/semantic address is not a sufficient mitosis boundary.
- Core 008: a small fixed shared matrix basis is rejected; individual normalized writes are nearly rank-1.
- Core 009A: asymmetric factorized functional geometry is formally supported.
- Core 009B-1: carrier-only writes retain roughly 97.6–98.2% of registered full-write target gain.
- Core 009B-2 / 009C: the tested pretrained effect representation does not expose the desired compact persistent sparse/local Cell ontology.

Therefore:

\[
\boxed{
\text{useful pretrained write interface}
\neq
\text{ready-made natural Cell ontology}
}
\]

Core 009D may continue in parallel as a non-blocking operator-geometry diagnostic.

---

# Track B — Constructive CLM Research

## Core question

\[
\boxed{
\text{Can continual pressure produce a reusable, addressable, protected and growing functional coordinate system?}
}
\]

Track B reuses:

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

and now also reuses two formal constructive results.

## G1a — Addressable learned coordinate formation

Status: **SUPPORTED**.

Experiment: **Constructive CLM-001 — Learned Coordinate Formation**

Formal result:

```text
status = LEARNED_COORDINATE_FORMATION_SUPPORTED
seeds = 90111 / 90112 / 90113
Cells = 6 / 6 / 6
pair route recall = 1.0 / 1.0 / 1.0
late spawns = 0 / 1 / 1
```

Boundary: the registered world gave every hidden factor clean singleton exposure. Do not rerun this scaffold as a new core validation.

## G1b — Latent coordinate discovery under superposition

Status: **SUPPORTED**.

Experiment: **Constructive CLM-001B — Latent Coordinate Discovery under Superposition**

Formal result:

```text
status = LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED
seeds = 90211 / 90212 / 90213
training singleton count = 0
latent Cells = 6 / 6 / 6
pair route recall = 1.0 / 1.0 / 1.0
triple route recall = 1.0 / 1.0 / 1.0
```

001B shows that the registered additive pair-superposition scaffold can recover latent Cell keys/effects without any singleton training and can route unseen pair/triple compositions from `x` alone.

Boundary: this is not arbitrary blind source separation, unknown nonlinear mixing, or language-scale latent discovery. Do not rerun a near-equivalent pair-superposition world as a new core validation.

## G2 — Long-horizon structure-tracking growth law

Status: **ACTIVE**.

Experiment: **Constructive CLM-002 — Long-Horizon Structure-Tracking Growth Law**

The new question is no longer whether Cells can form. It is whether learned state scales with reusable structure rather than the transaction stream itself.

Define:

- `N`: transactions processed;
- `M(N)`: true reusable latent vocabulary exposed by the registered world;
- `K(N)`: committed learned Cells.

The registered finite-horizon world grows its latent vocabulary approximately as:

\[
M(N)-M_0 \propto (N-N_0)^{0.60}
\]

across checkpoints:

```text
N = 256 / 512 / 1024 / 2048 / 4096
```

CLM-002 requires:

```text
K(N) ≈ M(N) << N
K(N)/N ↓
windowed spawn rate ↓
late reuse rate ↑
```

while preserving controlled pair/triple addressability and early-factor retention.

Critical anti-degeneracy guards:

- no `max_cells` hard cap exists;
- Cell count must track oracle latent count at every checkpoint;
- final `K=M=30`;
- growth must still occur after 90% of the horizon;
- refusing to grow is therefore not a positive result.

CLM-002 reuses the 001B relational mechanism only for the initial six-Cell bootstrap. Streaming growth uses an engineered residual/probation controller and therefore does **not** claim an endogenous growth policy.

A positive result is finite-horizon scaling evidence only. It must not be described as an asymptotic proof of `K(N)=o(N)`.

## G3 — Learned coordinates + existing protection

Next if G2 is positive.

Constructive CLM-003 integrates learned coordinates/growth with the already-supported Core-005 certificate logic.

This is an integration test, not a new certificate-principle test.

Required comparison:

```text
learned coordinates + unsafe writes
learned coordinates + certificate/no-growth
learned coordinates + certificate/growth
replay oracle
```

Primary question:

> Does replay-free protection preserve the learned coordinate system without collapsing plasticity or forcing near-linear growth?

## G4 — Model-level multi-Cell composition

001/001B contain controlled algebraic composition tests, but G4 remains a model-level execution question.

Required metrics should include:

- unseen combination accuracy/loss;
- cross-Cell destructive interference;
- route support recovery;
- composition residual;
- compute versus active Cell count.

Core 009D is informative but cannot substitute for model-level Cell composition.

## G5 — External -> endogenous transition

Only after the constructive core is stable.

Remove scaffolding one component at a time:

```text
prototype / relational / residual-growth scaffolds
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

---

# Product Track — External CLM Layer

Product feasibility remains separate from Native CLM feasibility.

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

The product benchmark must eventually compare against RAG/external memory, ordinary LoRA/PEFT, continual adapters, replay, and adapter-bank/MoE routing.

---

# Experimental discipline

- Frozen formal seeds are never silently replaced; any observed seed is permanently excluded from later confirmation.
- Discovery/diagnostic results are not promoted to formal confirmation claims.
- Negative natural-geometry results do not automatically become Native-CLM No-Gos.
- Hidden semantic/task/factor labels may be used for post-hoc evaluation only unless explicitly registered as learner input.
- An existing supported mechanism is not re-tested unless a new integration variable is named in advance.
- Growth evidence must report state growth and reuse, not only task accuracy.
- A short/finite stream is not described as an asymptotic theorem.
- Coordinate recovery is not sufficient without deployable read addressability.
- Read addressability is not sufficient without write safety and long-run growth control.
- Positive 001B evidence must not be generalized beyond its registered additive pair-superposition scaffold.
- Positive 002 evidence, if obtained, must not be generalized to a learned growth policy.
- Rank-1-per-write compression is not evidence of cross-write Cell reuse.
- Pythia-only evidence is not sufficient for product generalization.
- The formal research stop rule is the one frozen in the CLM Feasibility Evidence Map.

## Immediate order

```text
1. G1a / CLM-001 = SUPPORTED.
2. G1b / CLM-001B = SUPPORTED.
3. Run CLM-002 on untouched formal seeds only after development validation.
4. Core 009D may proceed in parallel and remains non-blocking.
5. If CLM-002 is positive -> integrate Core-005 protection in CLM-003.
6. Then model-level composition and external->endogenous scaffold removal.
```
