# Stage 2 engineering retrofit

E2 uses the pinned Qwen post-RoPE Q/K and native V at one selected attention
layer. Each KV head owns one FP32 memory state and serves its native GQA
query group. The per-channel sigmoid fusion enters before the native output
projection and uses the native attention output gate. Empty memory bypasses
the residual. The write commits after the model call returns.

The inner update reuses the frozen M3 functional chunk gradient, momentum
and forgetting. Stage 2 uses detached online updates. The separate Titans
reference under phase1 remains experimental. Qwen continuation calls are
single-token on this pinned backend.

Run e2_t4.ipynb for first-chunk parity, read-before-write comparison, actual
state updates and the selected-layer cache check. Pretrained T4 results and
formal outer training remain pending.
