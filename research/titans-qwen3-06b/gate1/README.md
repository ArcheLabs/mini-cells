# Gate 1 — Qwen backbone feasibility

Gate 0 implementation pending verification. Do not use the P1.2-TR screening
entry for this experiment: it wraps layer 3, commits after whole segments, and
changes independent projections. This adapter wraps layers[22], keeps native
Q/K norms and RoPE, and delegates address scaling, one-token state updates and
first-token masking to the pinned Titans forward. Segment cache ownership
belongs to the future driver, separate from the adapter's episodic state.

The pinned Titan-Llama initializer calls initialize_kv_memory with K/V weights.
In the identity branch those weights are reshaped but do not fit the map: the
initializer sets earlier matrices to I and the last to a GELU gain times I.
Thus compatible tensor shapes do not demonstrate recovery of native Qwen V
from normalized and rotated K. C/D remain pending compatibility evidence.

P1.5 published result f00e0d4aad67bd8b677b333b582772663786cd5c is the historical
diagnostic reference: swap–random separation begins before LR weighting and the
Step 200 swap update weakens further after clipping. This is single-chunk
evidence, not proof about multi-token momentum or forgetting. Exact checkpoint
lineage verification and native hook, causality, GQA, reset, and gradient
tests remain required before any training. No formal runner is enabled.
