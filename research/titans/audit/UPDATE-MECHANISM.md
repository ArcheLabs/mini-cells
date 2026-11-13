# Update mechanism

Source: EPCLM-004 M3 memory.py, NeuralMemory.learn and transition.

Each Cell carries a two-layer MLP with default widths 32 → 64 → 32,
GELU, a key residual and affine LayerNorm after the residual. Its fast
weights start as independent copies of the static initialization. Momentum
starts at zero. For each stream the inner loss sums squared feature errors
and averages over associations in the current chunk.

Autograd computes one gradient per chunk against the incoming fast weights.
A linear map of the mean normalized Key emits three sigmoid scalars shared
across tensors and layers. Momentum uses eta, step size uses theta scaled
by 0.05, and forgetting uses alpha scaled by 0.01. The functional transition
is S' = eta S − theta grad and M' = (1 − alpha) M + S'. Width one implements
sequential updates. Wider chunks aggregate the loss before a single
transition. Record one gradient evaluation per learn call.

Online adaptation detaches previous fast weights, momentum, controls and
representations from outer optimization. Meta-training retains the
higher-order graph. The portable Cell codec stores fast tensors, momentum,
update step and compatibility metadata, with integrity and shape checks.
