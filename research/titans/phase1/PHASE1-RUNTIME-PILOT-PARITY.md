# Runtime and development pilot parity

Both paths use frozen Qwen last-layer hidden states and the same M3
projection family. Runtime writes project separately encoded fresh Key
and Value inputs at every token. Runtime generation selects the last
hidden position of the active session, including carried context, and
projects that representation as Query. The development pilot independently
encodes Key, Value and Query packets and selects the final token of each.
It trains K/V/Q, the initial fast weights, controls, output projection and
gate using cosine candidate CE and full-vocabulary answer-token CE.
The runtime has no outer loss.

A pilot adapter can be scored through Runtime only with the runtime's
actual session hidden and update path. Passing separately precomputed
canonical queries would measure pilot behavior. The existing pilot
published metrics, while its trained adapter stayed in Kaggle Output.
Checkpoint readback and the four loss/baseline ablations remain pending.
