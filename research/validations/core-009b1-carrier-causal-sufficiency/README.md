# Core Validation 009B-1 — Carrier Causal Sufficiency

## Purpose

Core 009A established a formal positive for raw two-sided factor geometry. Its diagnostic bridge then showed that the apparently one-dimensional right side is almost exactly the train-token activation mean direction and disappears after centering/mean-direction removal/whitening.

009B-1 asks the next causal question:

> Is that common carrier merely a geometric anisotropy, or does it actually carry nearly all of the useful causal write effect at its natural magnitude?

This experiment is the first stage of the frozen continual-learning roadmap in `research/validations/CONTINUAL_LEARNING_ROADMAP.md`.

## Causal decomposition

For each sequence:

\[
G = \operatorname{mean}_t[(U^\top \partial L/\partial h_t) z_t^\top],
\qquad
\hat G = G/\|G\|_F.
\]

Fit only on training tokens:

\[
r = \frac{\operatorname{mean}_{train}(z)}
{\|\operatorname{mean}_{train}(z)\|}.
\]

Then:

\[
G_\parallel=\hat G rr^\top,
\qquad
G_\perp=\hat G(I-rr^\top).
\]

The interventions are:

\[
\Delta A_{full}=-\eta\hat G,
\]

\[
\Delta A_{carrier}=-\eta G_\parallel,
\]

\[
\Delta A_{residual}=-\eta G_\perp.
\]

**Carrier and residual are never renormalized.** The same target-specific \(\eta\) is used for all three variants.

## Discovery

Seeds:

- `81001`
- `81002`

Discovery computes **full writes only**. It selects the largest viable target hidden perturbation ratio:

```text
0.0003, 0.001, 0.003, 0.01
```

The selected ratio must independently on both discovery seeds satisfy:

- full-write descent fraction >= 0.95;
- median normalized target NLL gain >= 1e-5;
- median half-step linearity error <= 0.25;
- p90 half-step linearity error <= 0.50.

Carrier/residual results are not computed and cannot affect scale selection.

If no ratio is viable, stop. Do not run confirmation.

## Confirmation

Untouched seeds:

- `81011`
- `81012`
- `81013`

For each heldout target, compare full / carrier / residual using the locked scale.

Peers:
- one deterministic same-source peer, diagnostic only;
- up to six deterministic different-source peers, one per other source.

Formal per-seed gates:

- full descent fraction >= 0.95;
- carrier descent fraction >= 0.90;
- median carrier/full target gain >= 0.90;
- median residual/full target gain <= 0.20;
- median carrier excess unrelated positive harm / full target gain <= 0.10.

All three confirmation seeds must pass.

Positive status:

`CARRIER_CAUSAL_SUFFICIENCY_SUPPORTED`

Negative status:

`CARRIER_CAUSAL_SUFFICIENCY_NOT_SUPPORTED`

## Interpretation

A positive result means the common carrier component retains most of the actual frozen-model causal write benefit at its original magnitude.

It does **not** establish:
- reusable effect coordinates;
- sparse effect coefficients;
- inference-time addressability;
- safety certificates;
- bounded growth;
- a complete CLM.

A positive 009B-1 opens only **009B-2 Persistent Effect Geometry**.

A negative result stops the shared-carrier effect-memory route and requires a residual-aware functional hypothesis.

## Kaggle workflow

Requirements:

- GPU enabled;
- Internet enabled;
- `GITHUB_TOKEN` secret;
- optional `HF_TOKEN` for higher Hugging Face Hub rate limits.

### 1. Discovery

```bash
python scripts/research/orchestrate_core_validation_009b1.py \
  --phase discovery \
  --branch codex/core-validation-009b1-carrier-causal-sufficiency \
  --device cuda \
  --push-results
```

If and only if the final discovery decision says:

```json
{"confirmation_allowed": true}
```

the publisher commits `scale-lock.json`.

Refresh or reclone the branch before confirmation.

### 2. Confirmation

```bash
python scripts/research/orchestrate_core_validation_009b1.py \
  --phase confirmation \
  --branch codex/core-validation-009b1-carrier-causal-sufficiency \
  --device cuda \
  --push-results
```

The orchestrator checkpoints and pushes after every seed. Re-running hydrates matching published seed artifacts and skips completed work.
