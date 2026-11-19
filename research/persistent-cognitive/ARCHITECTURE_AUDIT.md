# Persistent Cognitive Architecture audit

The baseline for this audit is `c6ea8bc`. Stage 6C uses native post-RoPE
features captured at the attention input, with fusion before the native
output projection. Each of the six full-attention layers (3, 7, 11, 15,
19, 23) has two independent KV shards. Each shard serves four query
heads of width 256. Its memory is 256 → 64 → 256. The six fast tensors
have shapes [1,64,256], [1,64], [1,256,64], [1,256], [1,256] and [1,256].
Momentum mirrors the fast tensors. The Update 64 archive remains the
legacy reference.

Stage 7's frozen read-time SVD probe is a key-span diagnostic. Its output
does not establish the behavior of the TNT training system. The
development persistence fixture also stores reconstruction weights,
packets and expected scores alongside exported shards. Those files are
evaluation inputs. They must remain outside the new Cell package.

Reuse the pinned Qwen loader, tensor integrity checks and relevant
lifecycle tests. The new state schema, memory initialization and training
pipeline require independent implementation. Stage 6C and Stage 7
evidence remain in place.
