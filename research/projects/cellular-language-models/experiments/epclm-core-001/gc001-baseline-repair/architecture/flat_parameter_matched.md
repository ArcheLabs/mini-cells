# Active-parameter matched flat recurrence

Let N=2048, D=128, V=2048. Shared embedding/readout plus flat RMSNorm cost
`2*V*D + D + N*D + N = 788608` parameters. Bias-free down/up add `2*N*r = 4096*r`.
The nearest integer rank to target966784 is43 (tie with44 resolved toward43), giving
964736, error2048/966784 ≈ .2118%. All allocated tensors train and participate; there
are no leftover left/center/right/update modules from old F0.

After identical broadcast event injection, each of four rounds computes
`h += .1 * up(SiLU(down(RMSNorm(h))))` over the whole flattened vector. Readout is the
same 2048→128 pool/RMSNorm/2048 head as C1. Tests check gradients in every up-projection
output row and cross-block coupling; a single FP32 finite difference need not exceed
rounding in every coordinate. Coordinate activity is not independent information capacity.

Estimated linear FLOPs per event are `K*(4*N*r) + 2*N*D + 2*D*V`, excluding norms,
activations, memory and kernel overhead. Equal parameter counts do not equal compute.
Report measured GPU times rather than inferring throughput from this estimate.
