# Native CLM v0 — JAM Learning Demo

Status: **DEMO_COMPLETE**

> This is an engineering demonstration, not a new formal continual-learning decision.

## Model identity

- Base SHA-256: `91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f`
- JAM SHA-256: `e0a43a314a3f6cd3d0ab404bb67608eb432c2b324dc6f51a916eb5445df5313e`
- Selected step: **1200**
- Training rows: **409**
- Learner-invisible reasoning rows: **50**

## Before / after

| Benchmark | Before | After |
|---|---:|---:|
| JAM validation answer NLL | 2.5743 | 0.1426 |
| JAM factual token accuracy | 0.3580 | 0.9723 |
| JAM relational token accuracy | 0.3553 | 0.7015 |
| JAM misconception token accuracy | 0.3504 | 0.8613 |
| JAM reasoning answer NLL | 2.4368 | 1.6049 |
| JAM reasoning token accuracy | 0.3917 | 0.6346 |
| TinyStories validation perplexity | 2.2206 | 2.4016 |

## Claim boundary

The run supports the narrow demo claim that an already-trained Native CLM can acquire
bounded JAM knowledge through post-training. It does not claim replay-free continual
learning, general JAM reasoning, or superiority over ordinary fine-tuning.

See `QA_LOG.md` for deterministic before/after generations.
