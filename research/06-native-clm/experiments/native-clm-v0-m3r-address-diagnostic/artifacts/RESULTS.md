# Native CLM v0 — M3R Address Diagnostic

- Classification: `QUERY_GEOMETRY_SEPARABLE`
- Scientific decision: `False` (diagnostic only)
- Valid edges: `24/24`
- Current cosine median AUC: `0.5315`

| feature | median AUC | mean AUC | fraction >= floor |
|---|---:|---:|---:|
| query | 0.9623 | 0.9542 | 1.000 |
| write_input | 0.9726 | 0.9689 | 1.000 |
| write_left | 0.7767 | 0.7795 | 0.333 |
| write_pair | 0.9825 | 0.9797 | 1.000 |
| certificate_residual | 0.8637 | 0.8921 | 0.917 |

Interpretation:

Frozen query geometry contains a stable local boundary; prioritize a learned lineage-local read gate.

Boundary: checkpoint-only offline diagnostic over consumed M3R formal checkpoints; no Native CLM training, routing update, certificate update, growth, or new formal seed consumption occurred.
