# Experiment 004 — Learned Cell Graph

Status: **REGISTERED / UNRUN**. Development seeds **42, 43, 44**; scientific confirmation is **false**. No result is claimed before all five arms finish for every seed.

## Architecture and reused components

Reuse the pinned TinyStories corpus, tokenizer, seeded windows, 61M parameter causal Transformer, eight FFN Cells per layer, four-hop limit, optimizer, staged temperature schedule, evaluator, observed matmul counter and Pipeline v2. Fixed entry Cell 0 and absorbing STOP are retained. The document's 100M example is not a new capacity target. A full attention block inside every Cell would change the 003 substrate and is outside this comparison.

003 already has contextual **internal** transition heads. 004 adds a persistent learned `8 x 9` prior per layer:

`p(next | state, Cell i) = softmax((head_i(updated_state) + T[i, :]) / temperature)`.

The prior contains eight destinations plus STOP. It is not an external global router. Softmax produces dense positive probabilities; empirical edge support and thresholded prior support are reported separately, not claimed as exact structural sparsity.

## Training

- First 25%: soft conditional-mean relaxation, temperature 2.
- Next 25%: geometric anneal 2 → 0.1, soft forward.
- Last 50%: selected hard forward with renormalized top-2 backup-gradient mixture. The chosen Cell and an alternative are evaluated at the same incoming state; STOP is identity. Top-2 probabilities are floored at 0.05 before renormalization so the backup compute receives a positive gradient even at low temperature. Backward uses the mixture; forward equals the chosen branch. Cells outside top-2 have no counterfactual gradient. This is a biased surrogate.

Hard inference disables backups. Training Cell visits and throughput are recorded separately; inference FLOP savings must never imply equal training savings. Comparison with 003 is a joint prior + gradient change; it does not isolate topology's causal effect.

## Session plan: choose 004 in Kaggle Experiment

Use the original **bingjiangli/minicells** notebook through Pipeline v2. Choose **004**, seed group **1/2/3**, session **reference/graph**, and enable **execute**. Group 1 uses seed 42, group 2 uses 43, group 3 uses 44. The default is 004/group 1/reference. No source or Dataset hash needs to be entered.

Actions automatically remembers the first successfully uploaded Dataset and source commit in a code-keyed cache, reusing both across the matrix. Result PR merges do not change the code key. A change to the experiment code/protocol creates a new input cache and source identity; it never joins different source versions into one matrix. Cache misses prepare a new hash-checked snapshot; incompatible identities remain separate. Results accumulate on a source-specific branch and one result PR. Runtime reads mounted assets and embedded source; GitHub and HF credentials stay outside the GPU runtime.

For **each seed 42, 43, 44**, run these two sessions from the **same source commit** and immutable Dataset:

| Session input | GPU queue A | GPU queue B | Maximum worker budgets |
|---|---|---|---|
| reference | soft-cell, then fixed-cell | global-router | 1850 + 850 / 850 seconds |
| graph | graph-cell | hard-cell (003 STE) | 3000 / 1850 seconds |

Physical GPU placement alternates by seed. There is no DDP. Each arm gets the full **10M token** target; baseline time caps rise from 600 to 850 seconds so the old 420-second training cutoff is not reused. Each arm reserves 180 seconds for checkpoint/evaluation. Application cap stays **3300 seconds**, server cap **3600 seconds**. Budget feasibility is a cap calculation, not a guarantee that graph will finish on T4. A slow arm remains incomplete; the token budget and cap are not relaxed after observing it.

Six sessions complete the development matrix. Worst-case application wall time is 5.5 hours total; conservative dual-GPU accounting is 11 GPU-hours if Kaggle bills each GPU separately. Actual quota accounting follows Kaggle. No GPU smoke is part of this implementation.

## Frozen development criteria

All five full-budget arms, identical validation population, capacity within 1%, source/config/protocol/input identity must exist before any gate is evaluated. Each seed must independently satisfy:

1. Mean per-layer full-path entropy `graph < hard`.
2. Graph validation PPL `<= 13`.
3. Observed graph forward GEMM FLOPs/token `< 50%` of **fixed-cell** (not the more expensive soft-cell baseline).
4. At least two distinct hard paths and two utilized non-entry Cells in every graph layer.

Per-layer full validation populations, learned STOP vs forced truncation, empirical edge counts, transition entropy, path diversity and plug-in token/Cell MI are preserved. Fixed entry is excluded from MI. MI is descriptive and biased at finite sample size; it is not proof of semantic specialization. Additional prior, path and efficiency plots are derived views.

## Analyze after both sessions are archived

```
PYTHONPATH=src:. python -m minicells.research.graph_report \
  --archive research/artifacts/004-cell-graph/<source-commit> \
  --output research/artifacts/004-cell-graph/<source-commit>/matrix-analysis
```

Collection updates this derived matrix automatically. Missing sessions or failed checks produce `DEVELOPMENT_INCOMPLETE`. All three supported paired development seeds yield `DEVELOPMENT_SUPPORTED`, never a formal confirmation. Historical 001–003 protocols and decisions retain their authority; the new hard baseline is a registered reproduction under 004, not a rewrite of 003.
