# Experiment 006 audit repair 1

Scientific status: NOT REGISTERED. No formal training or confirmation corpus access authorized by this repair.

## Repairs and evidence

- Confirmation training maps only training assets. Periodic quality evaluation is disabled. The final checkpoint is frozen before the deferred confirmation loader opens tokens. A failed final evaluation resumes that same checkpoint without additional training.
- State-path (R1/R0), connection (R2/R1), and system competitiveness (development-selected R1 or R2 versus M0) have independent evidence verdicts. Formal claims still require human review. Select the system candidate before Gate 4, never from confirmation results.
- Context diagnostics include never-read fraction, first-read step histogram, separate ContextRead gradient/update statistics, and intermediate-head NLL before/after a read. The latter is a diagnostic proxy, not a causal contribution estimate or training objective. Same-path context no-op is a separate intervention.
- R1 and R2 both receive identical multiset-preserving order and context interventions. R2 adds connection removal, accurately named predecessor shift, and three seeded permutations of prior rows. Order noncommutativity alone does not establish connection value; the report includes the paired R2-minus-R1 order effect.
- Actual parameter inventories replace hand-written capacity estimates. Unique touched parameters and repeated executed parameter work are distinct. Linear, normalization and embedding call records cross-check the selected execution trace. GEMM FLOPs exclude uncounted non-GEMM operations; parameter work is not bytes moved. Cache bytes describe logical K/V tensors, not allocator residency.
- A tiny real Cell–Router exact-enumeration test covers Cell parameters influencing later policy probabilities, with a numerical derivative cross-check.
- GPU micro-preflight uses only synthetic integer tokens, one forward/backward per arm, no optimizer, decode parity and a three-repeat timing fixture. It is engineering evidence only and cannot establish learned paths, T4 performance, or the 30B/500M target.

## Unfrozen items

Real dataset/asset hashes, selected scale, scientific thresholds (including no-context fraction), development and confirmation seeds, training budget and GPU authorization remain unfrozen. REINFORCE and the shared Cell Pool architecture are retained; no Path Search redesign is introduced. Historical experiments and original Gate 0 evidence are preserved.

## Verification

Local Python 3.8 / Torch 2.4: 71 tests passed, including reused Experiment 005 tests. CPU Gate 0 runs both proposed scales. Exact results are recorded alongside this report. GitHub Actions validates the delivered commit separately.
