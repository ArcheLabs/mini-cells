# Titans learning, forgetting, and Memory Policy

This research proposal records the discussion following P4-M1.1. Implementation and experiment authorization remain subject to the existing P4 stage gates.

## Mechanism and evidence

Titans learns online by minimizing the squared error between the Memory output for a Key and its target Value. The current loss gradient is the surprise signal. In simplified notation, the update is

```text
loss = ||M_prev(key) - value||²
S = eta * S_prev - theta * grad(loss)
M = (1 - alpha) * M_prev + S
```

Eta retains past surprise, theta scales the current gradient, and alpha gates the previous weights. These controls depend on the input. The paper describes channel-wise forgetting. Context-sensitive behavior is a motivation for adaptive controls and should be measured directly in MiniCells.

The official Titans paper's Table 4 reports the following LMM ablations. Lower perplexity and higher accuracy are better.

| Configuration | Language modeling PPL | Reasoning accuracy | Long-context accuracy |
|---|---:|---:|---:|
| LMM | 27.01 | 47.83 | 92.68 |
| Without momentum | 28.98 | 45.49 | 87.12 |
| Without weight decay | 29.04 | 45.11 | 85.60 |
| Without persistent memory | 27.63 | 46.35 | 92.49 |

The authors leave theoretical guarantees about capacity and memory management open. These ablations motivate local controls for P4. [NeurIPS 2025 paper, §§2, 4.5 and Appendix](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)

## Frozen M3 and P4 observations

Frozen M3 computes three input-dependent scalar controls per stream from the mean Key of each chunk. Sigmoid outputs scale eta directly, theta by the default `max_lr=0.05`, and alpha by `max_forget=0.01`. The transition broadcasts each control over the Fast Weight tensors. The learned channel-wise forget gate described in the paper is a candidate variant. A future implementation needs an explicit channel axis and broadcast rule for each tensor. [Frozen M3 Memory](../../epclm-004-unified-memory-10m/m3/freeze/original-source/memory.py)

At constant maximum alpha, the direct coefficient on the initial weights is `0.99^n`. At 128, 1,024, 2,048 and 8,192 updates, the fractions are approximately 0.276, 3.39e-5, 1.15e-9 and 1.75e-36. This calculation excludes additive updates and does not measure recall. P4-M1.1 observed zero final first-group recall in nine order runs. The smoke does not attribute that loss to decay, gradient interference or capacity. Its acquisition harness discards per-update control diagnostics. [CPU smoke](../reports/M1.1-RESULTS.md)

## Proposed experiments

Start by recording realized alpha, eta and theta from the existing `learn()` diagnostics. Report distributions by write phase, chunk and update count, cumulative direct decay, gradient and momentum norms, recall of new associations and retention of earlier associations. Preserve the scalar baseline and the same seeds, maps, initial state and evaluation bank.

| Arm | Forgetting | Learning | Momentum |
|---|---|---|---|
| A | Off | Adaptive | Adaptive |
| B | Fixed | Adaptive | Adaptive |
| C | Adaptive scalar | Adaptive | Adaptive |
| D | Adaptive scalar | Fixed | Adaptive |
| E | Adaptive scalar | Adaptive | Off |
| F | Adaptive channel-wise | Adaptive | Adaptive |

Freeze fixed-control values and bounds before comparisons. Separate a granularity comparison at matched per-association exposure from one at matched update calls. Report both counters and cost, along with known recall and random and hard-negative acceptance. Sweep learning strength, forgetting caps and chunk sizes only in separately recorded variants. Compare the scalar and channel-wise gates at matched bounds and account for any extra control parameters.

## Proposed Memory Policy

Use Titans adaptive updates as the candidate default engine, with user preferences applied as bounded policy constraints. The engine retains responsibility for computing per-update controls. The Cell Runtime owns isolation, fork, checkpoint, restore and archive.

| Preset | Intended behavior |
|---|---|
| Balanced | Adaptive controls with replay disabled by default |
| Retention First | Lower forgetting and learning bounds, with optional bounded replay or dedicated Cells |
| Learning First | More responsive learning, with calibrated forgetting bounds and smaller chunks |
| Context Isolated | Independent Cells and policies by domain |
| Read Only | Recall with Fast Weights and Momentum fixed |

Custom settings could expose learning and forgetting strength, adaptive/low/high surprise persistence, chunk widths 1/4/8/16, and replay strategy. Apply scaling within validated bounds, for example `alpha_effective = clip(scale * alpha, 0, alpha_max)`. Surprise persistence constrains eta and should be evaluated separately from factual retention. Serialize the effective per-Cell policy and policy version alongside checkpoint compatibility metadata.

## Framework extensions and sequencing

Replay is a separate input scheduler. Compare none, partial, interleaved and full replay at declared presentation and update budgets. A bounded reservoir is a candidate sampling strategy. Track buffer capacity, provenance, privacy and deletion rules. Keep stored replay examples outside Fast Weights.

Protected Memory is a separately versioned store for explicitly protected facts. Neural Memory learns associations in Fast Weights. Replay Memory retains a bounded set of examples for consolidation. Titans Persistent Memory consists of input-independent learned task parameters, whereas this proposal's Protected Memory has explicit storage and modification semantics.

Phase A instruments the existing engine and runs controlled ablations, including channel-wise forgetting. Phase B calibrates presets and checkpointed policies. Phase C adds the replay scheduler, Protected Memory, archive and policy API. UI and autonomous decisions about learning, forgetting, replay and Cell creation follow mechanism validation and operate under user-defined limits.

## References

- [Titans: Learning to Memorize at Test Time, NeurIPS 2025 proceedings entry](https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)
- [Google Research, Titans and MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) (companion overview)
