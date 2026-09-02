# Constructive CLM-003 — Protected Learned/Growing Cells

Status: **PROTOCOL FROZEN — FORMAL SEEDS UNRUN**

## Question

Constructive CLM-001/001B established controlled learned Cell coordinates. Constructive CLM-002 established finite-horizon structure-tracking growth. Core Validation 005 separately established replay-free subspace-certified writes and mitosis in an explicit-routing linear-writable world.

CLM-003 asks the first integration question:

> Can learned/growing Cell coordinates host replay-free protected writes without either destructive forgetting or a return to transaction-linear growth?

The new variable is therefore **certificate integration with learned hierarchical routing and context-keyed mitosis**. This experiment does not re-prove the Core-005 certificate principle.

## Reused evidence

- Core 005: `SUBSPACE_CERTIFIED_MITOSIS_SUPPORTED`.
- Constructive CLM-001B: `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED`.
- Constructive CLM-002: `LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED`, formal result commit `243ac29c38f79d0f4c82723f13ecf8e093b6eee1`.

## Structural bridge

Each seed first reuses the CLM-002 mechanism to obtain 12 learned root Cell keys:

```text
001B six-factor relational bootstrap
  -> CLM-002 residual/probation streaming growth
  -> 12 learned root coordinates
```

This 1024-transaction bridge is only an integration precondition. It is not a new G2 growth-law experiment.

Required bridge diagnostics:

```text
root Cells = 12
covered posthoc factors = 12
duplicate assignments = 0
mean matched root-key cosine >= 0.985
```

Hidden factor IDs are used only after learning for scoring.

## Protected write world

Each learned root develops three recurring functional modes.

```text
root mode 0  ── learned root route key
mode 1       ── local context perturbation
mode 2       ── another local context perturbation
```

The top-level router always uses the learned CLM-002 root keys. Within one root lineage, the router uses the root key plus any child keys that have actually been created.

No mode label, correct child address, or novelty flag is passed to the learner.

Each mode owns two linear write blocks. Mode 0 first fills the root Cell's registered activation certificate. Mode 1 and mode 2 then present conflicting writes on already-protected activation directions. The new contexts are separately identifiable, but the current root cannot safely overwrite its protected behavior.

For `certificate_growth`, an infeasible write therefore produces:

```text
current context
  -> existing root lineage
  -> certificate says unsafe
  -> spawn child
  -> child.route_key = normalized current context mean
  -> empty W / empty Q
  -> commit current behavior
```

Old examples are never given to the certificate learner. Historical contexts must continue routing to their old root/child on their own.

## Core-005 reuse

CLM-003 imports the already-tested Core-005 primitives directly:

```python
constrained_update(...)
extend_basis(...)
```

The registered constraint remains:

\[
\Delta WQ = 0.
\]

`Q` is Cell-local bounded state built only from current committed activations.

## Variants

### `unsafe`

Always apply an unconstrained least-squares write to the currently routed Cell. No certificate and no mitosis.

Purpose: forgetting control.

### `certificate_no_growth`

Use the Core-005 certificate, but reject infeasible writes.

Purpose: expose the stability/plasticity limit.

### `certificate_growth`

Use the Core-005 certificate and spawn a context-keyed child when the current routed Cell is infeasible.

Learner history accesses must remain exactly zero.

### `replay_growth_oracle`

Use the same hierarchical routing and growth lifecycle, but reconstruct the protected activation span from retained historical activations for the currently routed Cell.

Purpose: full-history integration oracle, not a product mechanism.

## Registered scale

```text
learned roots                       12
functional modes per root           3
true functional Cells              36
acquisition writes                 72
long-tail reuse writes            720
total protected-write transactions 792
```

A correct protected-growth run should end with exactly two children per learned root:

```text
12 roots + 24 children = 36 functional Cells
```

and then perform the entire 720-transaction tail without additional spawning.

This makes the anti-degeneracy target explicit:

\[
K = 36 \ll N_{write}=792.
\]

## Formal gates

Every formal seed must satisfy all registered gates. Important groups are:

1. **Structural bridge**
   - 12 learned roots;
   - unique factor alignment;
   - mean root-key cosine >= 0.985;
   - all mode contexts select the intended learned root.

2. **Controls**
   - unsafe must measurably forget;
   - certificate/no-growth must retain history but lose at least half the replay-oracle acquisition gain.

3. **Replay-free protection**
   - certificate-growth historical replay/sample/label accesses = 0;
   - final historical regression MSE <= `1e-10`;
   - cumulative positive historical regression <= `1e-9`;
   - acquisition gain >= 0.98x replay-growth oracle.

4. **Integration agreement**
   - certificate-growth and replay-growth oracle safe/grow decisions agree on >=99% of acquisition writes;
   - all registered conflict acquisitions are rescued by growth;
   - all registered safe extensions commit without growth.

5. **Routing and bounded growth**
   - exact final mode-route accuracy >=99%;
   - modes 1/2 reuse their existing children >=99% in the tail;
   - no tail spawns;
   - final Cell count exactly 36;
   - every lineage contains exactly three functional Cells.

6. **Behavior**
   - certificate-growth and replay oracle final all-mode MSE <= `1e-8`;
   - write-transaction / final-Cell compression >=20x.

See [`protocol.json`](protocol.json) for the canonical machine-readable thresholds.

## Seed discipline

Development-only seeds:

```text
401
402
403
```

They are permanently excluded from formal confirmation once observed.

Untouched formal seeds frozen after implementation:

```text
90511
90512
90513
```

The runner rejects formal seeds through `--seed`.

## Fast smoke

This checks only the protection/lineage mechanism with synthetic orthogonal root anchors. It does not run the CLM-002 bridge:

```bash
python scripts/research/run_constructive_clm_003.py --smoke
```

## Development run

```bash
python scripts/research/run_constructive_clm_003.py --seed 401
```

Expected top-level status:

```text
DEVELOPMENT_RUN
scientific_decision = false
```

## Formal run

Only after smoke/development review:

```bash
python scripts/research/run_constructive_clm_003.py --formal
```

Formal positive status:

```text
PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED
```

Artifacts are written to:

```text
artifacts/experiments/constructive-clm-003-protected-growing-cells/
  decision.json
  gate-summary.csv
  variant-summary.csv
  RESULTS.md
```

## Interpretation boundary

A positive result would establish only the registered controlled integration:

> learned root coordinates + context-keyed lineage routing + Core-005 replay-free certificates can retain old behavior and achieve replay-oracle-level new-learning gain while functional mitosis tracks recurring context modes rather than individual writes.

It would **not** establish:

- arbitrary Transformer write safety;
- a fully learned router;
- a learned/endogenous growth controller;
- arbitrary functional-boundary discovery;
- simultaneous multi-Cell computation;
- language-scale continual learning;
- foundation-model plasticity;
- JAM execution.

If positive, the next main experiment is **Constructive CLM-004 — model-level multi-Cell computation**. Do not create a cosmetic 003B certificate rerun unless CLM-003 identifies a distinct integration failure.
