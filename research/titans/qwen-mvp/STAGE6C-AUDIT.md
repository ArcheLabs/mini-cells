# Stage 6C initial audit

Stage 6B intercepts native attention with ChunkCache, which returns only
the current call's K/V. Its single-token continuation therefore discards
the entire local history. Its fusion adds a sigmoid-gated residual and
its training performs one historical write followed by one read. Stage
6C calls for retained within-segment KV, convex fusion initialized at
0.1, and gradient flow across at least three 512-token segments.

Transformers 5.3.0 Qwen DeltaNet uses carried recurrent state only on
single-token calls. Feeding subsequent whole segments to the native
forward can reset the recurrent contribution. The integration must
retain the native single-token continuation path or verify an equivalent
backend first. The attention cache cursor is segment-local, while RoPE
position IDs must use the cumulative token position.

SegmentState implements a local cache lifecycle contract with an explicit
partial flush, preserving nonselected caches and recurrent state. Live
Qwen integration, per-layer fusion, differentiable segment training and
the experiment matrix remain pending. The lifecycle tests use synthetic
cache entries and establish only the boundary bookkeeping behavior.
