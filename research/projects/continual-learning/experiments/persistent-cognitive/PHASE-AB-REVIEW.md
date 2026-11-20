# Phase A/B review

Target: Qwen/Qwen3.5-0.8B at revision
2fc06364715b967f1860aea9cf38778875588b17, Transformers 5.3.0, eager.
The pinned config fingerprint is
b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204.
It has hidden size 1024, 24 layers, full attention at 3/7/11/15/19/23,
eight query heads and two KV heads of width 256. The remaining layers
use native DeltaNet. The runner records the actual PyTorch version.

The adapter supports the text CausalLM class. Projection and norm output
hooks consume native tensors, including the gate half of Q projection.
Memory addresses are pre-RoPE. Dynamic fusion consumes per-head Q,
local-output and gated-read RMS features. Memory can be bypassed.

The causal window includes the current token and at most W−1 prior
tokens. The explicit band mask applies during prefill and decode.
Completed forwards retain at most W KV entries per attention layer.
Prefill still has dense eager attention temporaries. Decode requires
explicit absolute position IDs and single-token continuation on this
DeltaNet version. The default window is 512. The smoke uses W=4 to
exercise eviction with a short fixture.

Run phase_ab_t4.ipynb for pretrained within-window parity, native
DeltaNet state parity and bounded continuation. Long-range retrieval
and memory-on pretrained numerical validation remain open.
