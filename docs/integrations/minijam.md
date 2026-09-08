# MiniCells ↔ MiniJAM integration

MiniCells owns the HybridCLM API, Cell artifact format, provenance checks, and
model-evolution research. MiniJAM owns JAM execution semantics, RPC, network
deployment, and ecosystem infrastructure.

The Stage-1 integration uses the application-neutral Work, state, Service
lifecycle, and public PVM executor boundaries documented in
[`docs/minijam/`](../minijam/). MiniCells must not duplicate MiniJAM protocol
specification; consult the canonical [MiniJAM documentation](https://docs.minijam.xyz/)
for network and RPC details.

The Stage-1 hierarchical training compatibility work is engineering evidence
and remains separate from formal HybridCLM claims. No release workflow starts a
formal seed.
