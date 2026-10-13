# Experiment 001 — Echo

Status: implementation specification. Scope: Kaggle research validation only.

Echo asks whether a deliberately small, PVM-friendly one-dimensional neural
cellular model can reliably reproduce short symbolic sequences. It is a
conventional character copy task, not a claim of life, consciousness, memory,
or sentience.

The field has 64 cells. Each synchronous update applies one shared residual MLP
to a fixed local neighborhood and the position's input embedding. The default
uses radius 2, four iterations, 16 hidden values per cell, ReLU, and clamping to
[-1, 1]. There is no attention, normalization, dropout, or circular boundary.

Synthetic data mixes random supported symbols (70%) with deterministic
pseudo-text (30%). Validation is fixed from seed 10001. Loss and metrics cover
only source positions.

The gate requires all seeds 1, 2, and 3 to finish at 99% or better token
accuracy, mean exact-sequence accuracy of at least 95%, fewer than 100,000
parameters, deterministic checkpoint evaluation, and only the permitted simple
operations. A single-run report is not the three-seed scientific conclusion.

The commands and result layout are documented in the repository README. No JAM,
PVM, fixed-point, frontend, wallet, NFT, or economic work is part of this stage.
