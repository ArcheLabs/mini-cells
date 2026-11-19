# Pinned Qwen3.5 attention audit

Model: Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17.
Transformers: 5.3.0, matching the existing runtime adapter.

The text config declares full attention at zero-based layers 3, 7, 11,
15, 19 and 23. Other layers use Gated DeltaNet. Each full attention layer
has eight query heads, two KV heads and head dimension 256. Native
attention splits the query projection into Q and output gate, normalizes
Q/K per head, applies RoPE, updates the layer cache, expands KV for GQA,
and applies the sigmoid output gate before the output projection.

The existing runtime preserves all attention KV within a session and
exports only convolution and recurrent state. The isolated E1 prototype
suppresses KV persistence at one full attention layer. Other layers
retain native caches and supply the position cursor. Continuation uses
single tokens because the pinned DeltaNet only consumes carried state on
the single-token path.

Sources: [pinned config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)
and [pinned model implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
