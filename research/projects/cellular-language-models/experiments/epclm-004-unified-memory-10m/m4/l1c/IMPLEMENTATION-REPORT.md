# H4 L1-C implementation and CPU evidence

Status: CPU preflight passed. Official Tesla T4 budget probe, training and research metrics are pending. This host exposes Quadro K620. H4 L1-C has zero official training updates.

## Historical baseline

Audited main 99aff2e83a9b08d76f860d1d322dff8fd68a7f5f. The preserved L1 source is 345604818da5287dfc026fe4f900d3d60a0211a3 and the result is cc46f0911ea00a5f42d0013b3a321014b397fdfc. The historical ledgers show 32 updates, 65,536 inputs and 1,024 targets per arm. Each arm made 1,536 identical Token 432 predictions on each task. The prior screen is inconclusive. Historical checkpoint digests were read from the published metrics. Historical checkpoint binaries were not available locally.

The historical adapter passed raw embeddings to the M3 Memory. L1-C passes the existing SWA hidden output to the same projections. It separates A acquisition from B supervision without adding parameters. Each canonical arm has 10,065,158 resident parameters and identical static initialization. The M2 trainer retains the original AdamW, sampler, checkpoint and rollback implementation. A multi-session entry validates the fixed boundary and the masked A labels before invoking it.

## CPU evidence

- Fixed-token Key perturbations changed the Value position context and K/V projections, and the Answer position context and Q projection. Observed L2 differences were 0.303 / 0.0772 / 0.0236 for fact context/K/V, and 0.0463 / 0.0143 for query context/Q.
- Hooks verified that A writes SWA features and B reads SWA features. The A Cell has 17 updates. B preserves its Fast Weights, Momentum and update counter. The B-only loss has a nonzero gradient through A's updated weights and the static initializer, Key, Value and Query projections.
- Causal suffix perturbation, fixed-shape stream isolation, masked branch execution and frozen parameter hashes passed. Four tiny CPU fixtures committed one optimizer update each. Their ledgers are separate from official accounting.
- Fresh interpreters restored A's Cell and B's Mamba prefix state and reproduced query logits exactly in all four fixtures. B-local SWA KV was reconstructed from the B prefix. No A cache was carried into B.
- Two focused tests passed. Training and held-out pairs have zero overlap. Each episode has 272 A tokens, 240 B tokens, four A targets and four B targets. The closest B fact is 187 tokens from the first Answer position. Five SWA layers have a 156-token receptive field.
- Early implementation exposed a Python import-name collision with the frozen trainer's model module. Renaming the contextual adapter resolved it. The successful preflight uses the final model code.

## Official execution

Four F00/F10/F01/F11 arms share the exact training order, 32 batch-four updates, 65,536 inputs and 1,024 targets each. Three evaluation banks supply 1,536 queries per arm. The launcher uses a no-update T4 probe and stops if the original session budget is insufficient. It records source-specific EM, NLL and gate weights, Token 432 collapse diagnostics, paired bootstrap contrasts and F11 ablations. Initial and final profiles are implemented. Extra profiling at updates 8, 16 and 24 is omitted to protect the execution budget.

The current code and CPU evidence establish the contextual interface and state lifecycle. Whether each branch learns its intended information, whether F11 reaches the 3 pp threshold, and the quality/resource trade-off require official GPU results.
