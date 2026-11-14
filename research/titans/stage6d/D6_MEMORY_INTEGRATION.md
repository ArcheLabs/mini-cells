# D6 · Fusion

Convex interpolation bounds mixing weights but attenuates local output
even when memory content is poor. Raw additive residual preserves local
output and exposes scale mismatch. A normalized gated residual can bound
relative magnitude while preserving the local path:

O = L + gate × RMS(L) × R / max(RMS(R), epsilon).

Normalize per head and token in FP32. Empty memory bypasses the branch.
Apply the native Qwen output gate and projection consistently, once.
Compare against frozen 6C interpolation and a true bypass. Measure
local and recall RMS, effective residual RMS, gate distribution, NLL
and donor sensitivity. Normalization can amplify weak reads near zero.
Keep fusion disabled for the Stage 7 direct retrieval primary endpoint.
Treat normalized residual as the integration candidate, pending measured
correct-history gain and native language quality checks.
