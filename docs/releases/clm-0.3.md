# CLM-0.3 Progressive Growth implementation

> Status: Historical Release
>
> This document describes a historical implementation and is not the current research status.

This implementation adds a research-only growth layer on top of the exact
CLM-0.1 model. The four CLM-0.1 root routes and the GRU recurrent substrate
are copied unchanged. A birth adds one stable-ID expert and a local binary
split below one existing leaf; it never changes the root from four to five
routes.

The implementation provides:

- explicit recursive `RouteLeaf`/`RouteSplit` trees;
- deterministic cosine 2-means split initialization with a 512-sample gate;
- pressure `U * (1 + G)` and deterministic random-parent selection;
- masked-dense training and sparse-dispatch evaluation;
- function-preserving birth parity, recurrent-state capture, and merge-back diagnostics;
- optimizer moment inheritance and global-step LR scheduling;
- versioned dynamic checkpoints with RNG/data position metadata;
- JSONL/live progress telemetry and resumable worker arguments.

The CLM-0.1 checkpoint is verified against SHA-256
`87d36c408ae3873ffd567ebf17050661b42ddae2c8d5d1bab84b2c27c3c7e7a0` before
CLM-0.3 construction.

The formal 3-replicate GPU experiment is not run by the implementation or
preflight tests. No progressive-growth signal is claimed here.
