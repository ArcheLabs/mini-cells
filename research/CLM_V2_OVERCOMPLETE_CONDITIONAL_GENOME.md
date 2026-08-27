# CLM v2 — Overcomplete Conditional Genome

CLM v2 treats programs as alternative capacity, not fragments of a mandatory computation.

Each TextNCA stage retains its original 512-hidden FFN only as a frozen developmental scaffold and
adds an independent overcomplete bank: one always-on 128-hidden shared MLP, twelve independent
64-hidden conditional MLPs, and a strictly pointwise receptor. Total FFN-equivalent genome capacity
is 896 hidden channels, or 1.75× the dense FFN. At K=6, active sparse-branch capacity is 512, equal
to the original FFN; K=5, 4, and 3 use 87.5%, 75%, and 62.5% respectively.

The developmental path is

`dense scaffold → off-path local imitation → α homotopy → scaffold-free consolidation → K reduction`.

At α=1 the main recurrent path is exactly TextNCA. The conditional branch may run only to optimize
normalized local imitation against a stop-gradient scaffold target. During handoff,
`Fα = α FD + (1-α) FC` with α fixed successively at 0.75, 0.5, 0.25, and 0. At α=0 inference the
dense scaffold is not called. It is a temporary developmental scaffold, not part of final CLM
inference.

All routing is local to `norm_ffn(candidate_state)`. Cell activation is always one. The model does
not contain phenotype, topology adaptation, lifecycle operations, task/capability labels, semantic
experts, or a global router.
