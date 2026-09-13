# Native CLM Architecture Lab

`research/native-clm/` is the long-lived architecture laboratory for **native cellular language models**.

The purpose of this research is deliberately narrower than the continual-learning program:

> Find the best next-token-prediction architecture for a native CLM before adding growth, mitosis, online writes, rollback, replay-free learning, JAM integration, or on-chain constraints.

The canonical working surface is a single living notebook:

- `native_clm_lab.ipynb`

The notebook may evolve as new architecture ideas are added. Completed run records must not be rewritten. Each promoted run is identified as `NCLM-XXX` and its immutable metrics/configuration should be exported under `results/` before a later notebook change supersedes the working candidate.

## Research question

Primary question:

> Can a parameter-matched native CLM outperform a modern small Transformer on next-token prediction?

The initial scale is approximately 2M parameters because it is intended for fast architecture search. Candidates that survive this stage should be re-tested at approximately 8M, 30M, and then larger scales.

The comparison must distinguish three claims:

1. **quality / parameter** — same train tokens and approximately the same parameter count;
2. **quality / training compute** — approximately the same total training FLOPs;
3. **quality / inference compute** — account for recurrent CLM steps rather than treating recurrence as free depth.

A win in (1) is useful but must not be reported as a win in (2) or (3).

## Baselines

The notebook keeps baselines and CLM candidates in one comparison table.

Required initial baselines:

- `T0`: historical/small Transformer reference when useful;
- `T1`: modern ~2M decoder-only Transformer using a current small-LLM recipe: pre-norm RMSNorm, RoPE, SwiGLU, grouped/multi-query compatible attention, tied token/LM-head weights where appropriate;
- `C0`: historical native CLM baseline reproduced at the same tokenizer/data/context budget.

Architecture changes are added one at a time where possible:

- `C1`: gated update candidate;
- `C2`: step/phase-conditioned recurrence;
- `C3`: small phase-specific modulation while retaining a shared Cell body;
- `C4`: receptive-field schedule search;
- later candidates: dual state, adaptive recurrence, compute/communication separation, functional Cell specialization.

Names are labels, not conclusions. A candidate becomes a validated optimization only after a controlled comparison demonstrates improvement.

## Historical evidence boundary

Existing MiniCells work contains strong evidence about **continual-learning mechanisms**, but much less direct evidence about static native-CLM architecture quality.

Examples of results that must not be misreported as PPL architecture wins:

- growth-restored plasticity;
- replay-free/subspace-certified mitosis;
- routing/growth engineering results;
- functional-boundary and dependency-partition experiments;
- CLM-0.4-mini token-level continual-learning validation.

They constrain the architecture search, but they do not by themselves show that a Cell update, routing rule, or recurrent layout improves next-token perplexity.

In particular, negative evidence from real-representation work warns against assuming that address-based specialization is equivalent to functional specialization.

## Experimental discipline

For a direct architecture comparison, freeze or report at minimum:

- dataset revision and split hashes;
- tokenizer and vocabulary;
- sequence/context length;
- train token budget;
- optimizer and schedule;
- seed set;
- parameter count (total and trainable);
- validation NLL/PPL;
- estimated training FLOPs;
- estimated inference FLOPs/token;
- wall-clock throughput/latency as an engineering metric;
- peak accelerator memory.

Do not select a candidate on the formal evaluation seeds. Use development runs for architecture iteration, then freeze the candidate and run untouched confirmation seeds.

## Living notebook, immutable evidence

`native_clm_lab.ipynb` is intentionally mutable. It is the current laboratory, not an immutable scientific artifact.

Promoted runs are immutable records:

```text
research/native-clm/results/
  NCLM-001/
    config.json
    metrics.json
    environment.json
    README.md
```

A later experiment may supersede the interpretation of an older run, but must not overwrite its recorded configuration or metrics.

## Initial optimization order

The first search should stay small and causal:

1. reproduce `T1` and `C0` under one data/tokenizer protocol;
2. test gated update dynamics;
3. test phase/step conditioning;
4. test low-rank or small phase-specific modulation while retaining shared Cell parameters;
5. optimize receptive-field schedule;
6. only then introduce adaptive depth/routing/specialization.

This order is intended to answer *why* a CLM improves rather than merely produce a complicated model that happens to score better.

## Non-goals for the first phase

The initial Native CLM lab does **not** attempt to validate:

- continual learning;
- Cell growth/mitosis;
- transaction rollback;
- replay-free safety;
- MoE-to-CLM conversion;
- JAM execution or consensus constraints.

Those mechanisms can be reintroduced after a strong static native CLM has been established.
