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

Protected retention advantage was about 6.75 percentage points; mean forgetting fell by about 24% relative to unsafe while preserving about 96% of unsafe new-domain plasticity. Protection therefore had a strong causal effect, but the registered fixed eight-Cell system was insufficient for the absolute retention requirement.

Per-seed protected / unsafe forgetting and A regression:

```text
73211  0.212006 / 0.279022   A=0.436378
73212  0.211121 / 0.279085   A=0.439569
73213  0.211381 / 0.278853   A=0.440100
```

## Interpretation boundary

M2 rejects the claim that the registered fixed 8-Cell topology + frozen read-address geometry + current bounded certificate is sufficient for replay-free long-horizon language retention.

It does not show that protection is ineffective. The strongest next causal test is whether new Cell capacity can absorb later-domain writes rather than forcing continued reuse of the same protected operators.

The original local M2 lightweight artifacts and six end-state checkpoints were lost after the Kaggle session terminated before publication completed. The formal console output and this closure preserve the decision and registered headline metrics, but later work must not pretend the missing binary artifacts still exist.

## No-repeat rule

Do not change the M2 20% gate, tune on 73211/73212/73213 and call the result untouched, or create a cosmetic M2B. A rerun of those seeds would be reproduction / artifact reconstruction only.

Next milestone: **M3 — Growth-Restored Continual Language** with new formal seeds and a matched fixed-topology control in the same newly pinned data snapshot.
