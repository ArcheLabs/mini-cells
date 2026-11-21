# MOM-STATE-DIAG-002 — First observed MoM divergence

Two fresh workers each rebuild twice from empty state at lengths 8 and 72.
The original sorting path and an isolated stable-sort intervention are
reported separately. Actual kernel invocation is checked. Capture layer
inputs, router logits and cutoff ties, selected experts, actual sort
permutations, Q/K/V short convolution inputs/outputs and delta kernel
inputs/outputs. Compare tensor bytes in observed execution order; worker
reports retain hashes and routing indices.

Stable sorting preserves order among equal keys and may change results
even when the original path is deterministic. Compare repeatability
within each arm; an output change across arms is not evidence of a
nondeterminism fix. Tracing adds CUDA allocations and copies; an exact
trace does not rule out drift in the uninstrumented model.

Run kaggle_diag.py on one T4 with Internet and GITHUB_TOKEN. No checkpoint
restore, training or external Cells. DIAG-001 and STATE-001 verdicts remain
unchanged. GPU results pending.
