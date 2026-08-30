# MiniCells CLM-0.1 Research Preview

CLM-0.1 is the first public MiniCells cellular-language-model checkpoint. Its purpose is to freeze
and expose the conditional-computation primitive that has already produced a positive result before
research moves directly toward progressive growth in CLM-0.3.

## Release claim

CLM-0.1 is a recurrent cellular language model with function-preserving expert upcycling and
strictly local conditional routing. The release is intended to support the following bounded claim:

> Existing TextNCA computation can be inherited into a larger latent expert genome without changing
> the initial function, after which local recurrent-state routing causes copied experts to
> differentiate and provides causally useful conditional computation.

CLM-0.1 is **not** presented as a general-purpose LLM, a speed-optimized MoE, or a self-growing model.

## Locked architecture

- base checkpoint: Experiment 006 `minicells-v2-10m.pt`;
- TextNCA dimension: 128;
- FFN hidden dimension: 512;
- recurrent stages: 3;
- recurrent iterations: `(4, 4, 4)`;
- windows: `(8, 32, 128)`;
- inherited full-width experts per stage: 4;
- active experts per local state: 1;
- local router: cosine prototype router;
- release initialization: geometry prototypes from unlabeled local perceptions;
- release candidate: Geometry replicate 2;
- continuation after the 10M-token base: 1M TinyStories tokens;
- final inference backend: `sparse_dispatch`.

The release candidate is selected from the already preregistered Upcycling Study 001 arms because
Geometry replicate 2 had the lowest final PPL among the three Geometry replicates. This selection is
recorded before the release rebuild; the release pipeline may not switch to another checkpoint based
on Validation 002.

## Release gates

The release script aborts unless all of the following are true:

1. all three Geometry replicas reproduce the published Upcycling Study 001 final PPL within absolute
   tolerance 0.05;
2. all three matched dense-continuation replicas also reproduce their published Study 001 PPL within
   absolute tolerance 0.05, protecting against training/environment drift;
3. before continuation, every copied-expert Geometry model passes the existing logits/PPL/recurrent-
   state `CLM_UPCYCLING_EQUIVALENCE` gate;
4. Conditionality Validation 002 returns `CLM_LOCAL_CONDITIONALITY_SIGNAL` in at least 2/3 replicas;
5. the selected r2 checkpoint loads strictly into the public CLM model;
6. `masked_dense` and `sparse_dispatch` produce matching release-evaluation PPL within `1e-4`;
7. the exact generated bundle loads through `CLM.from_pretrained()` and completes deterministic
   generation smoke tests;
8. the public release bundle is created with model, tokenizer, config, model card, conditionality
   evidence/decision, generation samples, and isolated inference benchmark telemetry.

## Public bundle

The generated directory is:

`results/clm-0.1-release/bundle/minicells-clm-0.1/`

and contains:

- `model.pt` — model weights and release provenance;
- `config.json` — locked architecture/routing configuration;
- `tokenizer.json` — exact tokenizer inherited from Experiment 005/006;
- `MODEL_CARD.md` — generated model card containing the actual Validation 002 result;
- `benchmark.json` — dense, masked-dense, and sparse-dispatch telemetry;
- `conditionality-002-decision.json` — authoritative conditional-routing release gate;
- `conditionality-002-evidence.csv` — per-replicate release evidence;
- `generation-samples.json` — public API smoke-test generations.

The same files are copied to the flat release-results directory for publication by
`scripts/publish_clm_0_1_release.py`.

## Python API

After installing the package:

```python
from minicells import CLM

model = CLM.from_pretrained("artifacts/releases/clm-0.1")
text = model.generate(
    "Once upon a time",
    max_new_tokens=48,
    temperature=0.8,
    top_k=40,
    seed=7,
)
print(text)
```

Routing telemetry can be returned during generation:

```python
result = model.generate(
    "The little girl opened the door",
    max_new_tokens=16,
    seed=7,
    return_routing=True,
)
print(result.text)
print(result.routing_usage)
```

`routing_usage` is per-generation-step aggregate expert utilization over the recurrent cellular
forward pass; it is diagnostic telemetry, not a semantic expert label.

## Benchmark boundary

CLM-0.1 reports isolated inference-only telemetry after optimizer/scaler/teacher allocations are
removed and after the non-benchmarked model is moved off the GPU:

- validation PPL;
- total model parameters;
- total expert parameters;
- active routed expert parameters;
- router parameters;
- tokens/second;
- peak allocated VRAM;
- masked-dense vs sparse-dispatch parity/telemetry.

No wall-clock speedup is claimed. The current Python sparse dispatcher prioritizes correctness over
kernel efficiency. CLM-0.1 also does not claim active FFN FLOPs below the original dense TextNCA:
one complete 512-hidden expert remains active per local update.

## Explicit non-goals for 0.1

CLM-0.1 intentionally excludes:

- phenotype;
- cell birth/death;
- progressive or automatic growth;
- online self-learning;
- 2D or growable topology;
- multimodality;
- semantic expert labels;
- 30M/100M scaling;
- sub-dense active FFN compute.

Those features must not be slipped into the 0.1 release branch. After 0.1 is frozen, the intended
next public research target is CLM-0.3 Progressive Growth.

## Reproduce and publish

On Kaggle with Internet enabled and two T4 GPUs when available, use:

`research/kaggle/clm-0.1-release.ipynb`

or run:

```bash
python scripts/run_clm_0_1_release.py --fresh
```

After reviewing `decision.json`, `conditionality-002-decision.json`, the generated model card,
generation samples, and benchmark, publish with:

```bash
python scripts/publish_clm_0_1_release.py --push
```

The publication branch is `release/clm-0.1-artifacts`.
