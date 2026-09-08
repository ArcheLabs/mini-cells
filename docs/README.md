# MiniCells Documentation

> Status: Current
> Scope: Engineering

Scientific experiment history is maintained under [`research/`](../research/README.md), not `docs/`.

## Getting Started

- [Rust/PVM build](getting-started/rust-pvm-build.md)

## Architecture

- [Architecture overview](architecture/overview.md)
- [Repository layout](architecture/repository-layout.md)
- [Model state](architecture/model-state.md)

## MiniJAM Integration

- [MiniCells ↔ MiniJAM boundary](integrations/minijam.md)
- [MiniJAM overview](minijam/overview.md)
- [Direct execution](minijam/direct-execution.md)
- [Keeper](minijam/keeper.md)

MiniCells owns the integration contract and Cell artifacts. Canonical MiniJAM
protocol, RPC, deployment, and ecosystem documentation remains at
[`docs.minijam.xyz`](https://docs.minijam.xyz/).

## HybridCLM

- [Overview](hybrid-clm/overview.md)
- [Quickstart](hybrid-clm/quickstart.md)
- [Model support](hybrid-clm/model-support.md)
- [Cell placement](hybrid-clm/cell-placement.md)
- [Mutation format](hybrid-clm/mutation-format.md)
- [Alpha control](hybrid-clm/alpha-control.md)
- [Provenance and safety](hybrid-clm/provenance-and-safety.md)
- [Research status](hybrid-clm/research-status.md)

## Deployment

- [Local deployment](deployment/local.md)
- [Smoke test](deployment/smoke-test.md)

## Reference

- [Protocol](reference/protocol.md)
- [Model format](reference/model-format.md)
- [Training fidelity](reference/training-fidelity.md)

## Historical Releases

- [CLM-0.1](releases/clm-0.1.md)
- [CLM-0.3](releases/clm-0.3.md)

## Research

See the [research landing page](../research/README.md), [final mechanism report](../research/reports/clm-core-mechanism-0.4.md), and immutable [canonical artifacts](../artifacts/experiments/).
