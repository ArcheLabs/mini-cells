# Core 008 Preflight — Artifact Sufficiency Audit

Status: **ARTIFACT_AUDIT_COMPLETE / EXACT_COUNTERFACTUAL_PENDING_REHYDRATION**

Scientific decision: **false**.

This audit does not alter Core 007. It asks only whether the already-published 80721/80722 artifacts can establish that the low oracle-vs-deploy NLL gap reflects functional equivalence rather than a weak-Cell-effect confound.

## Published evidence

| seed | eval mode agreement | eval mode disagreement | oracle NLL | deploy NLL | absolute NLL gap | absolute gap / oracle NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 80721 | 0.285714 | 0.714286 | 3.1878886223 | 3.1878786087 | 1.001358e-05 | 3.141132e-06 |
| 80722 | 0.339286 | 0.660714 | 3.1878855228 | 3.1878859997 | 4.768372e-07 | 1.495779e-07 |

The combination is striking: two thirds or more of heldout mode identities disagree, yet whole-model NLL barely changes.

That fact alone does **not** distinguish:

1. mode/Cell functional redundancy or equivalence; from
2. Cell interventions being too small relative to the frozen Pythia foundation for routing mistakes to move whole-model NLL appreciably.

## What was persisted

The canonical confirmation checkpoint persists:

- candidate aggregate metrics;
- transaction records;
- split records;
- rank records;
- training routing records;
- retention checkpoint records;
- per-Cell heldout causal-ablation records;
- mode metrics;
- router centroids and base assignment.

The final return payload does **not** persist:

- final candidate Cell matrices `A`;
- per-evaluation-sequence projected `z` / hidden states;
- per-evaluation oracle/deploy mode assignments and final owners;
- per-evaluation logits or Cell-output deltas.

Therefore the published artifacts cannot reconstruct:

\[
E_{ij}=\mathbb E_z\|A_i z-A_j z\|^2,
\]

mode-swap logit KL, or NLL regret normalized by local Cell contribution.

## Artifact-only conclusion

`FUNCTIONAL_EQUIVALENCE_ESTABLISHED = false`.

The near-zero whole-model NLL gap must not be cited as evidence that Core 007 modes are functionally equivalent. The required counterfactual state was not persisted. `causal-load.csv` can quantify the scale of per-Cell ablation signals and therefore expose a weak-effect proxy, but it still cannot answer the route-level counterfactual question.

## Does the lost Kaggle cache block the test?

**No.**

The Core 007 seed runner already defines the cache as optional. It:

1. loads the pinned Pythia revision;
2. deterministically selects the pinned SlimPajama inputs;
3. verifies the exact frozen data-manifest SHA-256;
4. attempts to load `frozen-hidden.pt`;
5. recomputes frozen hidden states when the cache is absent.

The expected manifest remains:

`d098f9172083b8de9f825b66de5277dde5b6ea0581b3a950b8f76e4f443546cc`

So the original closed Kaggle session is not required. What is required for the exact bridge is fresh compute capable of re-extracting the frozen Pythia representations, preferably a GPU session.

## Exact bridge now implemented

The branch contains:

- `scripts/research/analyze_core008_preflight_functional_equivalence.py` — artifact-only audit;
- `scripts/research/run_core008_preflight_functional_equivalence_seed.py` — deterministic rehydration and route-level counterfactual diagnostics for 80721/80722;
- `scripts/research/report_core008_preflight_functional_equivalence.py` — aggregate bridge reporter;
- `protocol.json` — frozen non-confirmatory bridge measurement protocol.

The exact runner first checks that the rehydrated candidate reproduces the canonical Core 007 oracle/deploy NLL and train/eval routing agreement. Only then are the missing diagnostics interpreted.

It measures, per heldout sequence:

- oracle/deploy mode identity;
- final Cell owner identity;
- same-owner mode mismatch;
- hidden Cell-output route distance;
- route distance normalized by Cell-output magnitude;
- logit route distance and normalized logit distance;
- symmetric logit KL;
- absolute NLL route regret;
- NLL route regret normalized by the larger oracle/deploy Cell effect relative to the frozen foundation path.

This is the bridge needed before choosing the Core 008 architecture. It remains a diagnostic analysis on already-observed seeds, not a new scientific confirmation.
