# D3 · Update

For a segment C use L = sum_i ||M_phi(k_i)-v_i||² / |C|.
At the boundary compute g = grad_phi L, then S' = eta S − theta g and
phi' = (1−alpha) phi + S'. M3 bounds its scalar controls from the mean
normalized key. Momentum begins at zero and is saved with fast weights.

Sequential updates recompute g at every new state. Chunk-start recurrence
evaluates token gradients at the chunk's incoming state and applies a
sequence of transitions. Segment batching averages gradients and applies
one transition. These differ even with the same exposures. Repeated
forgetting multiplies survival factors and repeated momentum changes
the effective coefficients of older gradients.

Keep the segment-mean rule for the integration reference and width-one
writes for Stage 7 packet acquisition. Run fixed-controls comparisons
before attributing any benefit to adaptive controls. Preserve bounded
forgetting as a separately testable factor. Freeze state during reads and
commit all shards atomically. [Titans equations](https://arxiv.org/html/2501.00663).
