# Representation pipeline

The 0.8 runtime uses Qwen3.5 last-layer hidden states. Its online write path
encodes Key and Value segments in separate fresh sessions. Generation uses
the active session's last hidden state as Query. M3 applies independent
linear maps followed by SiLU, with L2 normalization for Key and Query.
The runtime averages active Cell readouts and applies a gated projected
residual before the frozen LM head.

The development pilot instead independently encodes complete Key, Value
and Query packets in fresh sessions and pools the last token. It optimizes
the memory projections, static initialization, controls, output projection
and gate with retrieval CE plus answer-token CE. Qwen is frozen. The
evaluation uses cosine candidate ranking and full-vocabulary greedy
prediction. Neither path implements open-set rejection.

The published pilot scored 64/512 correct, 74/512 empty and 67/512 permuted
at memory level. Generation was 0/512 in all conditions. These results
leave representation alignment and update efficiency unresolved.
