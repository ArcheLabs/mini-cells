# Comparison with original Titans

Reference: [Titans, sections 3.1–3.3 and 4](https://arxiv.org/html/2501.00663).

| Mechanism | Original Titans | Frozen M3 | MiniCells 0.8 |
| --- | --- | --- | --- |
| Surprise and momentum | Gradient-driven memory recurrence | MATCH: core recurrence | MATCH: reuses M3 |
| Forgetting and step size | Input-dependent controls | SIMPLIFIED: bounded scalars from mean Key | SIMPLIFIED: same controls |
| Deep memory | Nonlinear memory with layer-dependent parameterization | SIMPLIFIED: two affine layers, GELU, residual and LayerNorm | SIMPLIFIED: same memory |
| K/V/Q | Separate projections | MATCH: separate projections, with SiLU and normalized K/Q | DIFFERENT: independent Qwen packets for writes; session hidden for runtime reads |
| Chunk recurrence | Chunk-start gradients with intra-chunk recurrence | DIFFERENT: one averaged gradient and one transition per chunk | DIFFERENT: same transition |
| Static persistent memory tokens | Sequence-level learned parameters | NOT APPLICABLE to standalone KV tests | NOT APPLICABLE to standalone KV tests |
| MAG | Sequence memory integration | DIFFERENT: shared MAG-lite residual | DIFFERENT: average readout plus gated projection |
| Open-set rejection | Outside the cited core recurrence | MISSING in candidate-ranking evaluation | MISSING in pilot |
