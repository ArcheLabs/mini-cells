# Pinned MoM source audit

MoM revision `125038087b71b635c171cec857bd0f21d10955b6` and the existing 340M model
are unchanged. The application validates 24 decoder layers, four routed banks,
a fifth shared bank, Top-2 routing, four heads and width 256. All weights freeze
after loading. The decoder runs attention normalization, MoM, and the original
residual/MLP path.

In the pinned `mom/layers/mom_gated_deltanet.py`, the router runs on the layer
input. FP32 softmax precedes Top-2 and renormalization. `transform` packs selected
tokens, then projections, ShortConvolution and the native Delta Rule produce
expert outputs. `reconstruct` restores token order and weights the routed
outputs. `shared_o` reads the shared bank independently. The two contributions
are added before `o_norm(o, g_proj(h))` and `o_proj`. The legacy product
baseline averages after these transforms, so it is a separate control from
pre-normalization equal fusion and its factorized equivalent.

The existing portable ZIP stores only recurrent tensors and signed-by-content
metadata (SHA-256). Convolution caches, token counts used by the native cache
and other transient fields belong to the query/write stream. The separate
resumable write stream file carries convolution and offset information. The
experiment preserves both formats and tests the entire persistent bundle.
Queries start fresh convolution and position fields, update isolated temporary
caches and discard them.

The new adapter is scoped to explicit experimental reads on the pinned model.
It reuses the upstream transform, ShortConvolution, L2 normalization and
fused recurrent kernel and retains the native output transform. Shared
projections and routing run once per layer. The adapter supports batch-one
unpadded segments of up to 64 tokens on T4. All-4 packs every token into all
four banks for temporary reads; writes remain native Top-2. The shared bank
is outside the routed pool. Learned variants use the same Cell-level rule
to weight the shared branch. For All-4, Cell features use the native Top-2
weights over that mode's actual read outputs.

Training uses paired simultaneous perturbation gradients on full teacher-forced
NLL with native inference kernels. Each direction replays the complete model
and reads independent cache copies. This preserves native FP16 recurrence on
T4, including its inference-only kernel behavior, and updates only the gate.
It is a small development budget and a different optimizer from backpropagation.
