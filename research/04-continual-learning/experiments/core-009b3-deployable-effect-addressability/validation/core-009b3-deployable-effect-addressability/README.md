# Core Validation 009B-3 — Deployable Effect Addressability

Status: `PREPARED_WAITING_FOR_009B2_LOCK`

009B-3 is fully specified and implemented, but **cannot run** until Core 009B-2 publishes:

- `scientific_decision=true`
- `supported=true`
- `status=PERSISTENT_EFFECT_GEOMETRY_SUPPORTED`
- a committed `basis-lock.json` with `locked_dimension <= 32`.

The parent is bound by a committed `parent-lock.json` before any 812xx seed may run.

## Question

Given a compact effect basis \(V\), can inference-visible prefix context predict coefficients

\[
\hat\beta=f_\theta(x)
\]

such that

\[
\hat a=V\hat\beta
\]

recovers most oracle causal write benefit and generalizes when a source is excluded from router training?

The oracle is evaluator-only:

\[
\beta^*=V^\top a,\qquad a= \hat G r.
\]

## Leakage boundary

Router input is only the first half of projected activation tokens \(z=U^\top h\).

Forbidden router inputs:

- labels / next tokens;
- gradients or \(q\);
- \(\hat G\);
- \(a\) or \(\beta^*\);
- hidden/token states after the prefix boundary;
- replay targets or oracle route IDs.

## Discovery

Seeds: `81201`, `81202`.

Discovery uses **effect-space metrics only** and may choose one learned router:

1. ridge on prefix mean;
2. 2-layer 64-hidden MLP on prefix mean;
3. tiny 2-query attention pool over prefix tokens, only when simpler families fail.

Baselines are mean effect and nearest-neighbor retrieval. They cannot be selected as the learned router.

Discovery must pass both IID and source-heldout/OOD gates on both seeds. Causal NLL is forbidden from router selection.

## Confirmation

Seeds: `81211`, `81212`, `81213`.

Confirmation inherits `rho=0.01` from Core 009B-1. For each target, oracle, deploy, mean and NN variants use the exact same target-specific \(\eta\).

Primary evidence:

- actual causal target NLL gain recovery versus oracle;
- correct-direction fraction;
- source-heldout gain recovery;
- excess unrelated harm;
- margins over mean and nearest-neighbor baselines.

Possible final statuses:

- `DEPLOYABLE_EFFECT_ADDRESSABILITY_SUPPORTED`
- `EFFECT_ADDRESSABILITY_DOES_NOT_GENERALIZE`
- `EFFECT_ADDRESSABILITY_NOT_SUPPORTED`

Only the first status opens Core 010.

## Execution order

After Core 009B-2 is formally positive:

```bash
python scripts/research/prepare_core_validation_009b3_parent.py \
  --branch codex/core-validation-009b3-deployable-effect-addressability \
  --push-results
```

Then run discovery from a checkout containing the committed parent lock:

```bash
python scripts/research/orchestrate_core_validation_009b3.py \
  --phase discovery \
  --branch codex/core-validation-009b3-deployable-effect-addressability \
  --device cuda \
  --push-results
```

Only if discovery publishes a committed `router-lock.json`:

```bash
python scripts/research/orchestrate_core_validation_009b3.py \
  --phase confirmation \
  --branch codex/core-validation-009b3-deployable-effect-addressability \
  --device cuda \
  --push-results
```
