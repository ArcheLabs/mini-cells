# CLM Conditionality Validation 002 — Aligned Local Routing

## Status

Preregistered release gate for MiniCells CLM-0.1. This validation does not change the model,
training recipe, router, experts, or the authoritative result of CLM Upcycling Study 001.

## Motivation

Upcycling Study 001 produced a `CLM_UPCYCLING_QUALITY_SIGNAL`: both copied-expert arms beat matched
dense continuation and Dynamic routing beat Static and sample-Shuffled controls by a wide margin.
The stronger conditionality status was blocked only by the original `sample_variation >= 0.05`
criterion.

That metric averaged routing masks over recurrent time and token position before comparing samples.
It can therefore report low variation when different samples use the same global expert proportions
but assign experts to different aligned local states.

Validation 002 replaces that *gate* with an aligned local disagreement metric while retaining the
previous quality, causal-control, and utilization requirements.

## Locked model and training

- Source: Experiment 006 `minicells-v2-10m.pt`.
- Architecture: CLM Upcycling Study 001 copied-expert model.
- Method: `copy_geometry` only.
- Experts: 4 full-width inherited FFNs per NCA stage.
- Router: strictly pointwise local cosine-prototype router.
- Top-k: 1.
- Continuation: exactly 1,000,000 tokens.
- Replicates: 3.
- Every replicate uses the exact Study 001 geometry extraction, model seed, optimizer, and data
  schedule.
- The reproduced final PPL must remain within absolute 0.05 of the already published Study 001
  Geometry PPL for that replicate or the release build aborts.

No task/domain/capability labels are used.

## Aligned route disagreement

For a hard top-1 routing mask, let

`z[t, b, p]`

be the expert selected at recurrent route slot `t`, sample `b`, and token/cell position `p`.

For each fixed `(t, p)`, compute the fraction of distinct sample pairs whose selected expert differs.
The final score is the mean over all `(t, p)` and validation batches:

`D_aligned = E[t,p,b!=b'] [ 1(z[t,b,p] != z[t,b',p]) ]`.

This preserves local position and recurrent-time identity. It does not reward mere changes over time;
it asks whether different input samples induce different computation at the same aligned local slot.

## Causal controls

The previous controls remain authoritative:

1. **Dynamic** — normal local routing.
2. **Static** — route-slot-specific calibration templates independent of the current input sample.
3. **Shuffled** — Dynamic masks are reassigned across samples while preserving route count,
   position, recurrent slot, utilization, and active compute.

The Shuffled control remains the strongest causal test: if the state-to-route match is irrelevant,
reassigning masks among samples should not materially hurt loss.

## Preregistered strong signal

`CLM_LOCAL_CONDITIONALITY_SIGNAL` requires at least 2/3 replicates to satisfy all of:

- `PPL_dynamic / PPL_dense_continued <= 1.03`;
- `D_aligned >= 0.10`;
- normalized Dynamic advantage over Static `>= 0.002`;
- normalized Dynamic advantage over Shuffled `>= 0.002`;
- normalized usage entropy `>= 0.80`.

Normalized advantage is `(NLL_control - NLL_dynamic) / NLL_dense_continued`.

If fewer than 2/3 replicates pass, the result is
`CLM_LOCAL_CONDITIONALITY_NOT_ESTABLISHED` and CLM-0.1 release export is blocked.

## Interpretation boundary

A PASS supports only the claim that local recurrent state controls causally useful conditional
computation in the locked CLM-0.1 architecture. It does not establish semantic experts, autonomous
growth, phenotype, online self-learning, multimodality, or sub-dense active compute.
