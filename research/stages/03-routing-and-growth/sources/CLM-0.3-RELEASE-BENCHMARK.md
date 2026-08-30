# CLM-0.3 — Public Release Benchmark

## Purpose

This benchmark is the public-release evidence layer for CLM-0.3. It does not introduce a new growth mechanism and does not compare against MoE.

The release claim is deliberately split into two independent questions:

1. **Language-model quality:** does adding the fixed CLM routing/lineage machinery preserve competitive language-model quality and acceptable reference-runtime cost?
2. **Developmental capability:** does the same model family gain a reproducible ability to create, evaluate, reject, or promote new persistent capacity after training?

The first question is measured here. The second is bound to the already-completed formal CLM-0.3d probationary-mitosis experiment.

## Evidence chain

### Foundation A — TextNCA vs matched Transformer

No new Transformer training is required for this release benchmark. Two earlier controlled scaling experiments are immutable public foundation evidence:

- Experiment 006: ~1.17M-parameter TextNCA vs parameter-matched Transformer through 10M TinyStories tokens.
- Experiment 007: ~30M-parameter TextNCA vs parameter-matched Transformer through 100M TinyStories tokens.

These establish that the underlying TextNCA language substrate can remain close to a conventional dense Transformer across two parameter/token scales.

The release benchmark must read the authoritative checked-in decision files and must not rewrite those historical measurements.

### Foundation B — TextNCA to CLM machinery bridge

The new GPU experiment starts from the exact released Experiment-006 `minicells-v2-10m.pt` TextNCA checkpoint and forks two arms:

- `textnca_continuation`: unchanged TextNCA.
- `clm_fixed4`: the exact same trained TextNCA converted to `ProgressiveGrowthCLM` with four root experts per stage and **no births**.

The bridge isolates the incremental cost of becoming a CLM from the already-established TextNCA language-model baseline.

### Foundation C — probationary developmental capability

The release benchmark binds to the immutable result branch:

`kaggle/clm-0.3d-probationary-mitosis-results`

The required formal status is `CLM_PROBATIONARY_MITOSIS_SIGNAL`, with 72/72 shadow births equivalent, stationary rejection in at least 2/3 replicates, and capability-shift promotion in at least 2/3 replicates.

The release benchmark does not rerun or retune CLM-0.3d.

## Machinery bridge protocol

### Source checkpoint

`artifacts/experiments/006-consumer-language-scaling/minicells-v2-10m.pt`

The benchmark verifies its SHA-256 before execution. Both arms are reconstructed from the same source state.

### Data

The original Experiment-006 materialized training prefix is 15M TinyStories tokens. The release benchmark materializes a longer deterministic stream and trains only from the suffix beginning at token 15,000,000.

Formal continuation budget per arm: **1,000,000 training tokens**.

This avoids intentionally reusing the Experiment-006 training prefix for the release bridge.

Both arms receive:

- identical training examples in identical order;
- identical validation batches;
- identical optimizer hyperparameters;
- identical learning-rate schedule;
- identical frozen teacher;
- identical continuation budget.

### Objective

Both arms use the same continuation objective:

`CE + 0.5 * KL(student || frozen source TextNCA)`

The common teacher is the immutable source checkpoint. The root-router balance loss is zero.

Using the same objective in both arms isolates architecture/routing effects rather than objective effects.

### Optimizer

Both branches intentionally start with a fresh AdamW optimizer from the common model state:

- learning rate: `3e-4`;
- betas: `(0.9, 0.95)`;
- weight decay: `0.1`;
- gradient clipping: `1.0`;
- warmup: `100` steps;
- cosine decay over the 1M-token bridge.

The Experiment-006 optimizer state is not reused because copying dense optimizer moments into the upcycled expert bank would introduce a separate optimizer-inheritance intervention. Both benchmark arms therefore reset optimizer state symmetrically.

### Evaluation ages

Age-zero evaluation occurs before the first optimizer step.

Formal continuation ages:

- 0
- 100K
- 250K
- 500K
- 1M

Validation uses one deterministic common TinyStories validation schedule.

### Age-zero equivalence

Before training, `clm_fixed4` must reproduce the source TextNCA on a common validation batch.

Required:

- PPL ratio within `1e-5` of 1;
- maximum absolute logits difference `<= 2e-5`.

Failure aborts the release bridge.

## Runtime characterization

Runtime is characterization evidence, not a claim of production optimization.

After the 1M continuation, each worker records:

- training tokens/second;
- training peak allocated VRAM;
- inference tokens/second;
- inference peak allocated VRAM;
- total parameters.

`clm_fixed4` inference is benchmarked using the current `sparse_dispatch` reference backend. TextNCA uses its ordinary dense forward.

The benchmark also reports a CLM **active-parameter proxy**:

`shared parameters + one active expert per stage + router parameters`

This is a structural parameter-activation proxy only. It is **not** reported as measured FLOPs.

## Language-quality decision

Let

`R = PPL_clm_fixed4(1M) / PPL_textnca_continuation(1M)`.

Preregistered interpretation:

- `R <= 1.03`: `CLM_RELEASE_LM_QUALITY_COMPETITIVE`
- `1.03 < R <= 1.05`: `CLM_RELEASE_LM_QUALITY_MODEST_OVERHEAD`
- `R > 1.05`: `CLM_RELEASE_LM_QUALITY_HOLD`

This gate measures only the incremental TextNCA-to-CLM machinery tax. Historical TextNCA-to-Transformer evidence remains reported separately rather than multiplying ratios across different experiments.

## Reference-runtime interpretation

Let inference throughput ratio be:

`T = tok/s_clm / tok/s_textnca`

and inference VRAM ratio be:

`M = VRAM_clm / VRAM_textnca`.

The reference runtime is summarized as:

- `CLM_RELEASE_REFERENCE_RUNTIME_ACCEPTABLE` when `T >= 0.50` and `M <= 2.50`;
- otherwise `CLM_RELEASE_REFERENCE_RUNTIME_OPTIMIZATION_REQUIRED`.

This threshold is intentionally loose because `sparse_dispatch` is a correctness/reference implementation, not a fused production kernel.

## Public release decision

The release benchmark emits three independent statuses:

1. `language_quality`
2. `reference_runtime`
3. `developmental_capability`

Overall recommendation:

- `CLM_0_3_PUBLIC_RELEASE_READY` when language quality is competitive, reference runtime is acceptable, and CLM-0.3d capability evidence is valid.
- `CLM_0_3_PUBLIC_RESEARCH_RELEASE_READY` when language quality is competitive or modest-overhead and CLM-0.3d capability evidence is valid, but reference runtime still needs optimization.
- `CLM_0_3_PUBLIC_RELEASE_HOLD` when the machinery bridge exceeds the 5% PPL-overhead boundary or capability evidence is invalid.

Runtime alone cannot invalidate the scientific mechanism; it can restrict the public claim to a research release.

## Public visualizations

The formal run generates publication-ready PNG and SVG figures from machine-readable results.

### Figure 1 — Language-model foundation

Relative PPL bars with `1.0 = matched baseline`:

- Experiment 006 TextNCA / Transformer;
- Experiment 007 TextNCA / Transformer;
- release bridge CLM fixed / TextNCA.

The references are explicitly labeled; ratios from different experiments are never multiplied.

### Figure 2 — CLM machinery bridge

Validation PPL vs continuation age for:

- TextNCA continuation;
- CLM fixed4.

### Figure 3 — Developmental selectivity

Promotion rate from formal CLM-0.3d:

- stationary continuation;
- controlled capability shift.

The figure also reports independently confirmed PPL gains for promoted replicates.

### Figure 4 — Reference runtime and structural cost

CLM/TextNCA normalized ratios with `1.0 = TextNCA` for:

- final PPL;
- active-parameter proxy;
- stored parameters;
- inference time per token;
- training time per token;
- inference VRAM.

Lower is better for all displayed ratios except that the active-parameter proxy is descriptive rather than a quality metric.

## Public summary

The result publisher creates `PUBLIC-RELEASE-SUMMARY.md` containing:

- the three-layer evidence chain;
- the four figures;
- a compact release table;
- supported claims;
- explicitly unsupported claims.

Supported claims may include only claims directly backed by the frozen evidence.

Not claimed by CLM-0.3:

- production-optimized sparse inference;
- repeated lifelong mitosis;
- generality to arbitrary capability shifts;
- large-scale 30M CLM growth;
- measured FLOP savings from the active-parameter proxy.

Repeated endogenous development belongs to CLM-0.4.

## Provenance

Formal release-bridge evidence is bound to:

- exact Git commit and tree SHA;
- clean tracked tree;
- source checkpoint SHA-256;
- corpus/tokenizer hashes;
- suffix offset and continuation schedule hash;
- validation schedule hash;
- optimizer/objective constants;
- runtime device metadata.

Resume under a different code commit is rejected. A code change requires an explicit restart of bridge evidence.
