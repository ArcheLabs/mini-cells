# D1 · Representation

Qwen projects normalized Q and K separately, applies RoPE, and uses eight
query heads over two KV heads of width 256. Each consecutive group of four
queries shares one KV memory. Hybrid6C captures these post-RoPE features,
detaches write K/V, and feeds the queries directly into M3 recall. Native
attention's learned compatibility does not establish compatibility for an
MLP that was fitted on keys alone.

Compare four address maps on identical values and episodes:

| Map | Cost and risk |
| --- | --- |
| Direct normalized native Q/K | No added parameters. Preserves the observed domain mismatch. |
| Shared encoder on canonical hidden packets | One trainable map per shard. Exact same packet yields the same address. Paraphrase alignment remains open. |
| TNT query projection into observed key span | Requires a basis and rank policy. The basis becomes persistent if reads must survive a session reset. |
| Memory-native separate encoders | Can specialize write and query roles. Requires paired training and stronger leakage controls. |

Select shared canonical addressing for Stage 7. Use pre-RoPE features for
portable addresses. Keep native RoPE in local attention. Post-RoPE remains
a frozen control. An absolute phase can bind the address to the source
position. Removing it does not remove contextual variation in hidden states.

Sources: [Qwen implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
[TNT](https://arxiv.org/abs/2511.07343).
