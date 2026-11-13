# Stage 4 engineering validation

Run stage4_t4.ipynb on T4. The fixture expands the online E2 retrofit to
all configured full-attention layers. Each layer has independent head
states. DeltaNet is unchanged. Explicit absolute RoPE positions carry
across calls when full-attention caches are empty. The smoke checks
first-chunk parity, zero retained attention KV and finite continuation.

The fixture initializes the layer-local memories independently. Trained
multi-layer weights, context-removal quality and long-context cost
measurements remain pending.
