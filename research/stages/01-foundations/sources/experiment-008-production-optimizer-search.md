# Experiment 008 — Production Optimizer Search

**Status:** implementation ready; results intentionally absent until Kaggle run  
**Baseline:** MINI Cells `93a2e42fc842a2ffa3123ca6faf60fd84ea08f66`  
**Target:** Q8.8 Echo V1 production-compatible training

## Question

Experiment 008 asks one narrow question:

> Under deterministic Q8.8 execution and the existing BASE/PLUS/MINUS Refine model, which optimizer semantics produce real Echo learning rather than merely lower margin loss?

The 512-generation local gate after the guarded SIGN-SPSA v2 repair reduced the fixed-probe loss from `607901` to `573303` (5.69%) with only 20 accepted updates, but fixed-probe token accuracy fell from `28/2168` (1.29%) to `0/2168`. The engineering guard repair therefore prevented blind candidate acceptance, but it did not establish useful trainability.

This experiment does not change MiniJAM, the service ABI, the Rust runtime, or the tracked `service.blob`. It is research-only. A winner must later be ported to Rust and pass Native/PVM gates before fresh-chain E2E.

## Additional semantic issue discovered

Production v2 currently evaluates PLUS/MINUS at `perturbation_q = 4`, chooses a direction from those results, then writes only `update_step_q = 1`.

Therefore the model that was proven better is not the model that is actually retained:

```text
evaluated: parent ± 4 * delta
retained:  parent ± 1 * delta
```

The current retained metrics are also copied from the ±4 candidate even though the retained model is ±1. Experiment 008 treats this as a first-class variable rather than hiding it with parameter tuning.

The search compares three apply modes:

- `legacy-step`: exact current production dynamics; evaluate ±q, then write ±step without rechecking. Used only for baseline reproduction and diagnosis. It can never be recommended.
- `evaluated-candidate`: retain the exact candidate that was evaluated. This requires `step_q == perturbation_q`.
- `step-recheck`: use ±q to select the direction, construct the actual ±step proposal, evaluate that proposal on the same batch, and accept it only if the actual retained proposal improves the BASE objective.

## Exact mirror and baseline gate

`src/minicells/optimizer_search.py` reuses the integer Q8.8 forward path from `continual_learning.py` and mirrors Rust `canonical_batch` plus the fixed local probe.

Before any search, the full Kaggle profile must reproduce the committed local run exactly:

| Metric | Required |
|---|---:|
| fixed-probe generation 0 loss | 607901 |
| fixed-probe generation 0 correct | 28 |
| fixed-probe tokens | 2168 |
| legacy generation 512 loss | 573303 |
| legacy accepted updates | 20 |

If this baseline does not reproduce, the script writes `BASELINE_REPRODUCTION_FAIL` and stops. Do not interpret any search result until this gate passes.

## Search axes

### A. Perturbation / step scale

Global guarded SPSA explores representative `(q, step)` pairs from small to large perturbations. This tests whether the 3.9% baseline acceptance rate is primarily a scale mismatch.

### B. Block SPSA

The experiment reuses the deterministic contiguous cyclic block mechanism already used by Experiment 004. It tests blocks of 64, 128, 256 and 512 parameters in addition to the 4476-parameter global direction.

This keeps two-sided SPSA evaluation while reducing the dimensionality of each random direction.

### C. Objective

Two deterministic objectives are compared:

- `loss`: current hinge-margin loss only.
- `accuracy-lex`: maximize correct Echo tokens first; use margin loss only as a tie-breaker.

This explicitly tests the observed mismatch where margin loss improved while token accuracy collapsed.

### D. Effective batch

Rust V1 has a canonical microbatch size of four. Experiment 008 preserves that exact group as group 0, then optionally aggregates 2 or 4 deterministic microbatches before deciding.

Additional groups use the domain:

```text
mini-cells:batch-extra:v1
```

with `(parent_hash, generation, group_id)`. This is an experimental, productionizable definition; it does not pretend that Rust currently supports a larger single `MAX_BATCH`.

## Staged search

The `full` profile uses:

1. Baseline reproduction: 512 generations.
2. Stage 1 structural screening: 128 generations over scale/apply/block variants using loss and one canonical microbatch.
3. Stage 2: the top three structures are expanded over `loss` vs `accuracy-lex` and 1/2/4 microbatch groups, then trained to 512 generations.
4. Finalists: the top three Stage-2 configurations continue deterministically to 2048 generations.
5. Each finalist runs a 128-generation regression from the solved Q8.8 model.

The CLI accepts generation overrides. A longer 4096-generation finalist validation can be run with:

```bash
python scripts/run_optimizer_search.py \
  --profile full \
  --final-generations 4096
```

The `smoke` profile exists only to validate the experiment harness quickly. It must not be used for a production decision.

## Ranking and hard gates

Ranking is deliberately not loss-only. It orders configurations by:

1. final fixed-probe token accuracy,
2. final fixed-probe loss improvement,
3. final/best loss stability,
4. production integration cost,
5. acceptance rate.

A configuration becomes a production candidate only when all gates pass:

- final fixed-probe loss improves by at least 10%,
- final fixed-probe token accuracy improves by at least 2 percentage points,
- final loss is within 15% of the best observed fixed-probe loss,
- at least four updates were accepted,
- solved-model regression passes,
- apply mode is not `legacy-step`.

These are search decision gates, not claims of consumer-grade capability.

## Production integration cost

The output labels each configuration:

- `drop-in-v2`: existing result/state shape can express it directly.
- `runtime-semantic-change-no-result-growth`: block selection, accuracy-aware decision, or deterministic microbatch aggregation requires Rust semantic/config work but no larger Training result.
- `requires-extra-proposal-metrics`: `q != step` with `step-recheck` requires the actual step proposal to be evaluated and represented before Accumulate can make a correct decision.
- `invalid-retained-guard`: current `legacy-step`; never recommend.

A high-performing expensive candidate remains visible, but the decision file does not silently mutate production.

## Outputs

The default output directory is:

```text
results/production-optimizer-search-v1/
```

Key files:

- `baseline.json`
- `stage1.csv`
- `stage2.csv`
- `finalists.csv`
- `solved-regression.csv`
- `probe-curves.csv`
- `decision.json`
- `recommended-runtime.json` (only when all gates pass)
- `finalist-probe-loss.png`
- `finalist-probe-accuracy.png`
- `loss-accuracy-frontier.png`
- `runs/<config-id>/model.bin`
- `runs/<config-id>/metrics.jsonl`
- `runs/<config-id>/probes.jsonl`
- `runs/<config-id>/state.json`

Each run is resumable. Re-running the script continues an existing compatible configuration from its saved Q8.8 model and generation.

## Local smoke

```bash
python -m pip install -e .
pytest tests/test_optimizer_search.py -q
python scripts/run_optimizer_search.py --profile smoke
```

The smoke profile validates parity and orchestration only.

## Kaggle

Open:

```text
research/kaggle/experiment-008-production-optimizer-search.ipynb
```

Run all cells. CUDA is not required: the experiment deliberately uses the exact integer CPU mirror so the search measures production Q8.8 semantics rather than a floating-point proxy.

Save the Kaggle output directory before changing any thresholds or code.

## What happens after Kaggle

If `decision.json` says `PASS`:

1. inspect the top finalist and `recommended-runtime.json`;
2. port that exact semantic configuration to Rust production code;
3. add Rust unit/integration tests for the new optimizer;
4. rerun the fixed Native gate;
5. rebuild clean `service.blob`;
6. run 128-generation Native/PVM exact parity;
7. only then start fresh MiniJAM E2E.

If the decision is `NO_PRODUCTION_CANDIDATE`, do not start PVM or E2E. The result means the current model/objective/search family needs another research iteration.
