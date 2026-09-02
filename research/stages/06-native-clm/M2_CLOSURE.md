# Native CLM v0 — M2 Formal Closure

## Decision

```text
status               NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
scientific_decision  false
formal seeds         73211 / 73212 / 73213
protocol SHA-256     4af6bc61355a7fb1aab8f47acb2a68838b430fe3b5474c059c3c3284420e6a00
data manifest SHA    a9bf79f9a53cd031fa3703322a1ad6ac11d9663c44ce2bc9671da16d0e81ca61
M1 checkpoint SHA    91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 is a valid frozen negative result. The three formal seeds were completed under the
registered fixed-topology protocol. The result must not be converted into a positive
claim by changing the A-retention threshold or reusing these seeds after tuning.

## Registered question

Starting from the exact trained M1 checkpoint, can protected sparse Cell-local updates
learn the sequential B -> C -> D language stream with zero learner replay while
retaining prior behavior better than unsafe writes?

The registered substrate was intentionally narrow:

- 12,154,368 parameters;
- 8 persistent Cells;
- 2 active Cells/token;
- one Cellular Layer;
- frozen shared substrate;
- frozen learned router and route keys;
- Cell operator weights only are writable;
- growth disabled;
- protected arm uses certificate-nullspace projected Cell gradients;
- unsafe arm uses the identical Cell-local write stream without projection.

## Gate result

All registered gates passed on all three seeds except one:

```text
exact_same_m1_checkpoint             PASS 3/3
cell_only_writes                     PASS 3/3
shared_and_router_frozen             PASS 3/3
replay_free_stream                   PASS 3/3
fixed_topology                       PASS 3/3
sparse_cell_execution                PASS 3/3
protected_phase_plasticity           PASS 3/3
protected_absolute_A_retention       FAIL 3/3
unsafe_interference_exposed          PASS 3/3
protected_retention_advantage        PASS 3/3
protected_plasticity_preserved       PASS 3/3
```

The failed gate was pre-registered as:

```text
final protected TinyStories-A regression <= 20%
```

Observed A regression was approximately 43.6–44.0% on every formal seed.

## Aggregate formal metrics

| metric | protected | unsafe |
|---|---:|---:|
| mean new-domain plasticity | 0.3560 | 0.3694 |
| mean forgetting | 0.2115 | 0.2790 |
| A / TinyStories regression | 0.4387 | 0.5228 |
| B / WikiText forgetting | 0.0342 | 0.0588 |
| C / Python-code forgetting | 0.1616 | 0.2553 |

Additional aggregate comparisons:

```text
protected retention advantage       0.06748  (~6.75 percentage points)
relative mean-forgetting reduction  24.19%
protected / unsafe plasticity       96.38%
A-regression reduction vs unsafe    0.08415  (~8.41 percentage points)
```

Per-seed headline values:

| seed | protected forgetting | unsafe forgetting | retention advantage | protected A regression |
|---:|---:|---:|---:|---:|
| 73211 | 0.212006 | 0.279022 | 0.067016 | 0.436378 |
| 73212 | 0.211121 | 0.279085 | 0.067965 | 0.439569 |
| 73213 | 0.211381 | 0.278853 | 0.067473 | 0.440100 |

Protected phase gains were also stable and comfortably above the registered 5% gate:

```text
B / WikiText       mean gain 0.38044
C / Python code    mean gain 0.51366
D / Dolly          mean gain 0.17392
```

## Scientific interpretation

M2 rejects the claim that the registered **fixed 8-Cell topology + frozen read-address
geometry + current bounded certificate** is sufficient for replay-free long-horizon
language retention.

It does **not** show that protection is ineffective. The causal intervention is strong
and extremely consistent across all three seeds:

1. protected writes retain essentially the same new-domain plasticity as unsafe writes;
2. protected writes reduce overall forgetting by roughly 24%;
3. protected writes reduce old-domain regression on A, B, and C;
4. nevertheless the oldest M1 behavior (A) still regresses by roughly 44%, far above
   the registered 20% ceiling.

The strongest current diagnosis is therefore a **capacity / topology / certificate
coverage limit**, not a failure to learn new domains and not a failure of the
certificate intervention to have causal effect. With only eight fixed operators and a
frozen router, every new domain must reuse the same read/write address set. Protection
reduces destructive overlap, but the registered system cannot preserve enough of the
oldest token-predictive function through three sequential shifts.

This diagnosis remains a hypothesis until the next registered experiment separates
fixed-topology saturation from certificate-representation limits.

## What must not happen next

Do not:

- raise the 20% A-retention threshold after seeing this result;
- tune on seeds 73211/73212/73213 and call a rerun untouched confirmation;
- create a cosmetic M2B that only changes learning rate, step count, or certificate
  rank to turn this frozen negative into a positive;
- claim replay-free continual language is already solved.

## Next milestone

The clean next experiment is **M3 — growth-restored continual language**.

M3 should use the M2 result as the fixed-topology baseline and introduce dynamic Cell
growth as the principal new mechanism. The preferred causal design is:

```text
same canonical M1 checkpoint
same registered B -> C -> D corpus stream
same zero learner replay requirement
same sparse Cell execution objective
same protected-write mechanism
new untouched formal seeds
+
dynamic child creation / reuse under conflict or protected-capacity pressure
```

Where possible, M3 should keep all other M2 quantities fixed. It should test whether
new functional capacity can absorb later domains without forcing unsafe overwrite of
the oldest M1 behavior.

A positive M3 would support the stronger interpretation already suggested by the
constructive sequence: protection reduces interference, while growth is required when
protected reusable capacity is insufficient.

## Publication note

The first M2 publisher invocation encountered a Hugging Face `403 Forbidden` while
attempting the first LFS checkpoint upload. This was an infrastructure permission
failure, not a scientific-run failure: all formal seeds and the registered decision had
already completed locally.

The publisher is now designed so lightweight scientific evidence can still be pushed
to Git when Hugging Face binary publication is unavailable. Future canonical notebooks
also preflight model-repository write access before expensive formal training.
