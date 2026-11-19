# Architecture

The controlled recurrence uses a residual 512 → 2048 → 512 GELU MLP
with LayerNorm. Every affine parameter participates in surprise gradients.
The online recurrence reads before writing and maintains separate fast
weights and momentum. Phase 1 uses fixed scalar controls for the
synthetic diagnostic. Adaptive controls and independent Q/K/V projections
belong to the later outer-training stage.
