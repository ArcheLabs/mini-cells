# Native CLM v0 — M3L Query-Sketch Gate Diagnostic

- Classification: `QUERY_SKETCH_GATE_NOT_FEASIBLE`
- Scientific decision: `False` (mechanism diagnostic only)
- Valid edges: `24/24`
- Current cosine median AUC: `0.5227`
- Offline oracle median AUC: `0.9281`
- Query-sketch gate median AUC: `0.8968`
- Median normalized oracle recovery: `0.9356`
- Median heldout old FPR: `0.1855`
- Median heldout current TPR: `0.8204`
- Median historical sketch bytes: `27720`

Interpretation:

The offline edge-local oracle is separable, but the registered compact historical sketch loses too much boundary information; improve the address-state representation before a new continual-language formal run.

Boundary: consumed M3R checkpoints only; Native CLM parameters are frozen, old raw queries are used only to construct the historical sketch and heldout evaluator, and the sketch-derived gate itself receives no old token/query replay.
