# Cellular Language Models

This project studies language models whose computation is organized around
Cells, including native CLMs, HybridCLM, routing, and language-quality
validation. It brings together the former language-validation, native-CLM, and
HybridCLM lines so each experiment can be read in its research context.

## Contents

- [`literature/`](literature/README.md) indexes theory documents in Chinese and
  English and the existing experiment-specific paper collections.
- [`context/`](context/) preserves the original stage documents and conclusions.
- [`experiments/`](experiments/) contains experiment units, including the
  routing series, compositional Cell Graph, and Path-First CLM.
- [`examples/`](examples/) contains project-specific examples.

## New research line: Mamba-3 Cellular

- [Mamba-3 Cellular (M3C)](experiments/mamba3-cellular-series/README.md): Mamba-3 hierarchical backbone with optional stateless, flat Cells and bounded shared Workspace Attention. Proposal only; no training authorized.

## Running an experiment

- [EPCLM-003/004 — Unified Memory Architecture, 10M](experiments/epclm-004-unified-memory-10m/README.md): M0 repository/environment audit and staged implementation plan. E0–E5 implementation and training remain pending.

Start from that experiment's README or notebook and use its frozen configuration
and source commit. Do not infer scientific approval from the presence of a
launcher or a completed development run.
