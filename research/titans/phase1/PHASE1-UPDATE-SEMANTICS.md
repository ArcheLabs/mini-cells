# Update semantics

Sequential native M3 recomputes the gradient from the latest state for
every association. Native chunk and batch arms average the chunk loss and
apply one transition. At load 8, chunk 8 and full batch are identical.

The isolated serial reference evaluates each gradient at chunk-start
weights and applies tokenwise momentum and forgetting. Width-one
equivalence with native M3 passed on CPU. This checks the boundary case.
Independent multi-token verification against the paper equations is
pending. The reference retains M3's memory network and control
parameterization. Its results cannot establish full-paper fidelity.

Each recorded native update includes one loss forward, gradient evaluation,
control evaluation and memory transition. There are zero outer optimizer
steps. Runtime in seconds measures CPU update wall time. Equal exposure
does not imply equal computation.
