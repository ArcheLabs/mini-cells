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

## CI layers

Routine MiniCells CI is source-free with respect to MiniJAM: it runs the
application-neutral boundary check and validates the public contract fixture at
[`configs/minijam-stage1-contract-v1.json`](../../configs/minijam-stage1-contract-v1.json).
The canonical dependency identity is recorded in
[`configs/dependencies.json`](../../configs/dependencies.json); it contains no
MiniJAM or Jambda source commit.

Real node/deployment checks belong to the manual
`.github/workflows/minijam-integration.yml` workflow and will consume a
released, digest-pinned MiniJAM OCI image when one is published. The
source-level `tools/bootstrap_deps.sh` and `tools/bootstrap_minijam.sh` helpers
are legacy development tools only.
