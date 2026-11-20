# D5 · State

Static model state includes checkpoint, tokenizer, addressing maps and
fusion weights. Session state includes local KV, DeltaNet convolution and
recurrent tensors, absolute position and pending segment writes.
Persistent Cell state includes fast weights, momentum, step, revision and
compatibility metadata. Include any explicitly adopted address basis in
that schema. Runtime snapshots and Cell artifacts remain independent.

Create → snapshot → read → stage write → validate → commit → export.
Import verifies compatibility and integrity before attaching. Fork copies
mutable tensors and records parent revision. Incomplete segments require
an explicit flush or discard. Fresh sessions begin with empty KV and fresh
DeltaNet state when testing Cell-only transfer.

E1 bypasses memory. E2 reads correct A and continues target B writes.
E3 swaps only A for same-layer/head donor state at matched age. E4 resets A
at the same boundary and permits B writes. Keep target tokens, positions,
and session state equal at the intervention. Later states may diverge as
a causal result of the intervention.
