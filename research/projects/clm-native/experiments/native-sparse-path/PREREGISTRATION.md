# Screening preregistration v1

Recorded before M2. The exact machine-readable configuration is [configs/default.json](configs/default.json). The full supplied protocol is retained byte-for-byte in [PROTOCOL_v1.0.md](PROTOCOL_v1.0.md). New results cannot rewrite historical research.

## Hypotheses and independent decisions

H0: true messages, synchronous sparse receiver updates, no oracle input and no all-pairs scoring. Engineering gate only.
H1: Task B C0 vs state-addressing B0/B3, quality vs complete measured forward latency and cost. B1/B2 alone cannot establish H1.
H2: legal candidate edge swaps preserve source outdegree, target multiset, message/update counts and all candidate read/score budgets; original replay and restoration must agree. One noncommutative perturbation or a path drawing is insufficient.
H3: input/node permutation, depth 6 vs training depth 4, and N=128 vs training N=64 with fixed c/k. Only frozen-checkpoint evaluation.
H4: encoding, score, lookup, message, aggregation, update, readout and scheduler are measured together. Resident-input CPU/GPU latency is distinguished from dataset/transfer pipeline time. No estimate is called total hardware FLOPs.

## Task and data lock

Task A: shuffled layered nonlocal addresses; state-dependent one-hop choice and continuous gated transformation; no route labels. Task B: cycles, multiple roots, state-dependent top-2 branching and synchronous summation/merging, bounded frontier. Task C: static-priority nonlocal computation with the same affine/tanh rules; dynamic choices unnecessary. Labels come only from an independent NumPy executor. A second literal Torch test executor cross-checks it.

Candidate graph generation precedes solving and receives no answer. Every candidate list has eight distinct addresses. Correct selection is defined among those candidates (conditioned bounded-discovery problem), not arbitrary content discovery. Train/dev/sealed/OOD instance seeds occupy separate 100,000-sized ranges; template ranges are disjoint between splits. Index order and physical numbering are randomized independently of values/answers. No semantic labels, route oracle, global state or labels are fields of the model input.

The same generated examples, same initialization seed, optimizer, 200 steps, batch eight, 512 train instances and 64 dev instances are used for B0/B1/B2/C0 on A/B/C. Loss is MSE, fixed entropy weight 0.002, stability weight 1e-5; hard operation limits replace tunable cost penalties (cost coefficients zero). No tuning sweep. Final checkpoint only; no dev checkpoint selection. Truncated straight-through softmax weights are a biased differentiable training approximation, not an unbiased policy gradient. Inference is discrete.

State width 32; N=64; c=8; k<=2; sources/updated recipients<=8; maximum T=16. Instance depth=4 provides a common task-defined stopping point before T. Pruned targets use fixed visible address-invariant priorities, never learned score or label. No message until recipient budget is determined. No growth/lifecycle systems.

## Fixed screening gate

Seed zero; results always SCREENING/scientific_decision=false. Task B C0 must improve MSE over the strongest B0/B1/B2 by >=5%, paired per-example bootstrap 95% upper difference <0, latency ratio <=1.10. A trained global mean/root-pooling diagnostic must not solve Task B (accuracy <=0.50 at max-coordinate absolute error <=0.1). Zero-prediction accuracy is also reported. If shortcuts solve B, stop and version the generator before training more arms. These are entry criteria, not scientific success.

Only a passed Task B gate permits B3 and seeds [0,1,2] with the same budgets. No favourable signal only on A is sufficient. Paired intervals from a single seed are conditional on that seed. Three independent training seeds and the strongest state-addressing B3 are mandatory for a positive conclusion. Longer training or model-size changes require a separately versioned amendment, not an implicit rerun.

## Sealed gate

No sealed instances or answers are generated during development, tests, or screening. Only a final checkpoint evaluation with a separately reviewed APPROVED registration lock is permitted. Lock must pin config hash, source, checkpoint hash and three distinct seed identities. Initial lock is NOT_REGISTERED. A successful M1/M2 run does not approve formal confirmation.

## Evidence identity

Every run records source commit, source-tree hash, configuration hash, deterministic data manifest/hash, seed/arm/task, precision, training steps, optimizer/RNG checkpoint, elapsed time, parameters and checkpoint hash. Raw per-example predictions, targets, edge traces, cost counters and learning curves remain available. Do not report NO_SUPPORT from an undertrained single-seed screen; report INCONCLUSIVE and the stopped milestone instead.
