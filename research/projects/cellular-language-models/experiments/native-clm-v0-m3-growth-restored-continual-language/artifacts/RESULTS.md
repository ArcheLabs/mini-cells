# Native CLM v0 M3 — Growth-Restored Continual Language

- Status: `NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED`
- Scientific decision: `False`
- Protocol SHA-256: `9bc23cac3cf4e4512f251836e4dd2cd48750b5894565c1a346396df06028f658`
- Data manifest SHA-256: `38197be7396d292106700b94208cdc65cf935809889f3759caa2f2ff5e390e16`
- Formal seeds: `[73411, 73412, 73413]`
- Learner replay: `0 bytes`
- Causal arms: `fixed protected` vs `growth protected`

| seed | fixed A reg | growth A reg | A advantage | fixed forgetting | growth forgetting | final growth Cells | child reuse | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 73411 | 0.4416 | 0.4938 | -0.0522 | 0.2137 | 0.2201 | 16 | 1.000 | FAIL |
| 73412 | 0.4293 | 0.4838 | -0.0545 | 0.2107 | 0.2170 | 16 | 1.000 | FAIL |
| 73413 | 0.4351 | 0.4889 | -0.0539 | 0.2123 | 0.2186 | 16 | 1.000 | FAIL |

Boundary: growth sees current training loss, routing, certificate rank and projected/raw gradient pressure only; no phase/domain/evaluation label controls spawn decisions.
