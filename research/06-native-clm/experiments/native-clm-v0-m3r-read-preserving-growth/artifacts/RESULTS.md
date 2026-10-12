# Native CLM v0 M3R — Read-Preserving / Lineage-Isolated Growth

- Status: `NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED`
- Scientific decision: `False`
- Protocol SHA-256: `c3e73545899ccf20f54411df701f22dd64b10cb46ff728e862c2d002a94f8627`
- Data manifest SHA-256: `213ddb9d093ea44fd0524e6ba6318f86a61c54270bd5cad6ddeb3233470565b0`
- Formal seeds: `[73611, 73612, 73613]`
- Learner replay: `0 bytes`
- Causal arms: frozen M3 global-pool growth vs lineage-isolated growth

| seed | global A reg | lineage A reg | A advantage | global forgetting | lineage forgetting | max birth drift | A child-share reduction | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 73611 | 0.4967 | 0.4722 | 0.0245 | 0.2207 | 0.2106 | 9.54e-06 | -0.020 | FAIL |
| 73612 | 0.4947 | 0.4691 | 0.0256 | 0.2193 | 0.2089 | 9.54e-06 | -0.020 | FAIL |
| 73613 | 0.4961 | 0.4713 | 0.0248 | 0.2204 | 0.2098 | 1.05e-05 | -0.020 | FAIL |

Boundary: all M3 pressure-controller numerical thresholds are unchanged; M3R changes only read topology and the leaf-only lineage allocation constraint required by that topology.
