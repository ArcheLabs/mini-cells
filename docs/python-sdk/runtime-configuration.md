# Runtime configuration

The native FLA backend uses one CUDA GPU. The current runtime reports a clear
error for CPU. `dtype="auto"` uses FP16 for T4 compatibility. Explicit BF16
requires native device support. FP32 is accepted by the loader and awaits real
GPU acceptance. The SDK defaults to a write chunk of 64. T4 rejects larger
explicit chunks up front. BF16-capable GPUs allow larger explicitly requested
native write chunks. Query prefill uses recurrent segments up to 64 tokens.

`max_cells=8` is the initial automatic-read limit. `selection_top_k=2` is
configurable. Diagnostics record scoring, selection, combined fusion/generation
time and peak GPU allocation. The GPU acceptance launcher measures complete
generated queries at 1, 2, 4 and 8 Cells.
