# EPCLM GC-001 Baseline Repair
## Experimental Design & Implementation Specification

**Project:** MiniCells / EPCLM\
**Experiment:** GC-001 Baseline Repair\
**Stage:** Development Benchmark\
**Priority:** P1 → P2 → P3\
**Status:** Proposed — Not Yet Implemented

---

# 1. Objective

GC-001 已经完成 Cellular 自回归生成引擎，并获得了第一组语言生成、性能和基线数据。

但实验暴露出两个问题：

1. 当前 NO-COMM 基线只有入口 Cell 保持非零状态，不能代表完整的 2,048 维独立 Cellular 状态系统。
2. 当前 Flat Recurrent 参数量明显高于 C1，缺少严格的参数量匹配对照。

此外，Transformer Reference 尚未实际运行，生成质量也尚未经过完整诊断。

本次 Baseline Repair 的目标是修复这些问题，建立一套更可靠的 EPCLM 语言建模基准。

本次实验分为三个优先级：

- **P1 — Baseline Fairness Repair：** 修复 Cellular 对照，增加 Independent Cellular 和 Parameter-matched Flat Recurrent。
- **P2 — Transformer Reference：** 运行现有 Native Transformer，建立 Attention-based 参考基线。
- **P3 — Generation Quality Diagnosis：** 系统诊断语言生成质量、训练状态和自回归退化。

三个优先级完成后，形成 GC-001 的完整补充实验报告。

# 2. Existing GC-001 Baseline

必须保留原始 GC-001 结果，不能覆盖、修改或重新解释为新实验。

## 2.1 固定实验参考

- Original result commit: `49c6fef619d51cac9ce5079d698e2876ff2d144b`
- Implementation source: `fc999f7677200ea875b8aacf2fe56ee896bd43ec`
- Original training source: `398a191dbb32e383c86c89b9c981777c7cd6c77f`
- Original data identity: `1eac9053e15ab4666a4f77635dd785728801a2cbea44a22472c42e8a920092cb`
- Original training seed: `173018021`
- Generation seed: `713`
- Precision: FP32
- Vocabulary: 2,048 BPE tokens
- C1 state: 16 × 128 = 2,048 dimensions
- C1 evolution steps: K=4
- C1 residual coefficient: α=0.1

Original training protocol:

| Parameter | Value |
|---|---|
| Training tokens | 250,000 |
| Batch size | 4 |
| TBPTT | 64 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Betas | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1 |
| Precision | FP32 |
| Document reset | Every document |

The original C1 and F0 checkpoints must be reused as frozen historical references.

Do not retrain them in this experiment.

## 2.2 Existing results

| Model | Parameters | Validation PPL | Development Holdout PPL |
|---|---:|---:|---:|
| C1 | 966,784 | 335.17 | 352.05 |
| B1 Frozen No-Comm | 966,784 | 376.58 | 394.42 |
| NO-COMM Trained | 966,784 | 340.13 | 356.48 |
| F0 Flat Recurrent | 1,493,120 | 340.37 | 361.03 |
| T1 Transformer | Not run | Not available | Not available |

The current NO-COMM implementation is a center-only transition. Its other cells remain zero because only Cell 0 receives token events.

It must remain available as a historical diagnostic baseline, but must not be used as evidence of a full-capacity independent Cellular architecture.

# 3. Priority 1 — Baseline Fairness Repair

## 3.1 Objectives

Establish three comparable architectures:

1. Cellular with inter-cell communication.
2. Independent Cellular without inter-cell communication.
3. Flat Recurrent without Cellular organization.

All three must use:

- The same tokenizer.
- The same training dataset.
- The same training token budget.
- The same state dimension.
- The same event input representation where applicable.
- The same output and evaluation protocol.
- The same training seed and optimizer configuration.

The primary question is:

**Does inter-cell communication provide a measurable advantage over independent cellular state evolution when event access and model capacity are controlled?**

## 3.2 New architecture A: Independent Cellular

Name: `IC-ALL-SELF`

This is the most important new baseline.

### A. State architecture

Preserve the existing Cellular dimensions:

- 16 Cells.
- 128 dimensions per Cell.
- Total State Dimension: 2,048.
- Shared local transition parameters.
- No inter-cell communication.

Each Cell maintains its own persistent state.

All 16 Cells must participate in every input event.

### B. Event distribution

The original C1 injects the event into Cell 0 only.

That behavior must not be reused for this baseline.

If the same embedding is broadcast identically to all Cells, their initial states and subsequent trajectories can remain identical under shared transitions.

To avoid this degeneracy, introduce a fixed, deterministic, cell-specific coordinate permutation.

For token embedding e, define:

`P_i(e) = roll(e, 8 * i)`

where i is the Cell index, from 0 to 15, and roll is a cyclic coordinate permutation over 128 dimensions.

The event injection is:

`state[:, i, :] += P_i(e)`

Properties:

- Every Cell receives every token event.
- Each Cell receives a distinct representation.
- All transformations are parameter-free.
- No additional trainable embedding parameters are introduced.
- The event encoding is deterministic and reproducible.

The exact permutation rule must be versioned in the experiment protocol.

### C. Independent transition

The Independent Cellular model must preserve the existing C1 transition parameter structure.

For each Cell:

`normalized_i = norm(state_i)`

`message_i = center(normalized_i) + left(normalized_i) + right(normalized_i)`

`delta_i = update(concat(normalized_i, message_i))`

`state_i = state_i + alpha * delta_i`

This corresponds to the existing neighbors-replaced-by-self transition semantics.

All three projections remain active and trainable.

Important:

- Do not freeze the left/right projections.
- Do not replace the transition with center-only.
- Do not use neighboring Cell states.
- Do not aggregate Cell states during recurrent evolution.
- Global aggregation is permitted only in the final readout.

Each Cell evolves independently, using the same shared local transition parameters.

This preserves the full parameter structure while eliminating communication between Cells.

### D. Required assertions

Add tests verifying:

1. Every Cell receives every token event.
2. The 16 event representations are deterministic.
3. Cell states are not artificially identical.
4. All 16 Cells have nonzero state after normal prefill.
5. No Cell reads another Cell's state during local evolution.
6. Perturbing one Cell cannot change another Cell's next state.
7. All intended transition parameters receive gradients during training.
8. Snapshot and Restore preserve all Cell states.

The model must report:

- Nonzero Cell count.
- Per-Cell state norm.
- Pairwise Cell-state similarity.
- Trainable parameter count.
- State memory.

## 3.3 New architecture B: Cellular Communication Control

Name: `C1-BROADCAST`

This is a required companion control.

It uses exactly the same event distribution as `IC-ALL-SELF`.

The only difference is the source of neighboring messages.

For each Cell:

`message_i = center(normalized_i) + left(normalized_(i-1)) + right(normalized_(i+1))`

All other components must remain unchanged:

- Same event permutations.
- Same embedding.
- Same 16 × 128 state.
- Same update network.
- Same K=4.
- Same α=0.1.
- Same readout.
- Same training protocol.

This gives us a controlled comparison:

| Property | C1-BROADCAST | IC-ALL-SELF |
|---|---|---|
| 16 Cells | Yes | Yes |
| Every Cell receives events | Yes | Yes |
| Cell-specific event representation | Yes | Yes |
| Persistent Cell states | Yes | Yes |
| Shared local transition | Yes | Yes |
| Inter-cell communication | Yes | No |
| Trainable parameter structure | Same | Same |

The difference between these two models is the communication topology.

The original C1 remains a separate historical reference because it uses entry-only event injection.

Do not merge the original C1 and C1-BROADCAST identities.

## 3.4 New architecture C: Parameter-matched Flat Recurrent

Name: `F0-PM-BROADCAST`

The existing F0 has 1,493,120 parameters, substantially more than C1.

A new Flat Recurrent implementation is required.

### A. State

Preserve a total persistent state dimension of 2,048.

The implementation may retain the [16,128] storage shape, but the recurrent transition must operate on the flattened 2,048-dimensional state.

### B. Event input

Use exactly the same 16-cell event distribution as C1-BROADCAST and IC-ALL-SELF.

Flatten the 16 transformed event representations into one 2,048-dimensional input.

### C. Flat recurrent transition

Use a low-rank dense recurrent transformation:

`h = flatten(state)`

`delta = W_up(SiLU(W_down(RMSNorm(h))))`

`h = h + alpha * delta`

Repeat the transition K=4 times.

Use bias-free linear projections unless the implementation requires otherwise.

The transition must operate across the entire flattened state, not independently on each Cell.

### D. Parameter matching

The target is the C1 trainable parameter count:

`target_parameters = 966784`

Implement a rank-selection routine that searches for the integer bottleneck rank r producing the closest trainable parameter count.

Acceptance target:

`abs(actual_parameters / target_parameters - 1) <= 0.01`

If no rank satisfies the tolerance, report the nearest architecture and explain the discrepancy before proceeding.

Parameter matching must use active trainable parameters, not merely allocated tensors.

Do not retain unused Cellular projections or other dead parameters to artificially inflate the count.

All trainable parameters must participate in the forward pass.

### E. Required reporting

Report:

- Bottleneck rank.
- Total allocated parameters.
- Active trainable parameters.
- State dimensions.
- Weight memory.
- Estimated linear FLOPs.
- Measured training time.
- Measured generation throughput.

This is a parameter-matched comparison, not a compute-matched comparison.

## 3.5 Priority 1 training protocol

All three new models must use the same protocol:

- C1-BROADCAST
- IC-ALL-SELF
- F0-PM-BROADCAST

Training:

| Setting | Value |
|---|---|
| Tokens | 250,000 |
| Seed | 173018021 |
| Batch | 4 |
| TBPTT | 64 |
| Optimizer | AdamW |
| LR | 3e-4 |
| Betas | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1 |
| Precision | FP32 |
| Scheduler | Existing warmup/cosine |
| Document reset | Every document |

All runs must start with fresh model and optimizer states.

Do not reuse incompatible optimizer checkpoints.

The original C1 and F0 remain frozen references.

Training data order, document resets, target alignment and loss calculation must remain identical.

## 3.6 Priority 1 acceptance criteria

Engineering acceptance:

- All three models complete their prescribed training.
- No numerical failures.
- Parameter matching passes for F0-PM.
- All intended parameters are active.
- Independent Cellular has 16 active state blocks.
- Independent Cellular has no inter-cell state dependency.
- C1-BROADCAST and IC-ALL-SELF use identical event injection.
- All evaluation and generation outputs are reproducible.

Scientific reporting:

Compare:

1. Original C1.
2. C1-BROADCAST.
3. IC-ALL-SELF.
4. Existing NO-COMM.
5. Existing F0.
6. F0-PM-BROADCAST.

The primary communication comparison is C1-BROADCAST versus IC-ALL-SELF.

Do not infer communication superiority from the original C1 versus the existing center-only NO-COMM.

# 4. Priority 2 — Transformer Reference

## 4.1 Objective

Run the existing Native Transformer implementation as a reference architecture.

This is not an attempt to build a new Transformer.

Reuse the existing implementation:

`ModernTransformerLM`

Existing configuration:

| Component | Configuration |
|---|---|
| Hidden dimension | 192 |
| Layers | 4 |
| Query heads | 6 |
| KV heads | 2 |
| FFN | 480 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Embeddings | Tied |
| Context | 128 |

## 4.2 Runtime

Reuse the existing `TransformerReference` adapter.

Important constraints:

- Maximum retained context: 128 tokens.
- KV Cache: Disabled.
- No incremental KV implementation.
- Context is recomputed through the existing reference adapter.
- Padding must not be treated as actual history.
- Snapshot must preserve bounded context and identity.

This is a reference-only execution mechanism.

Do not apply C1's no-history-replay requirement to the Transformer architecture.

## 4.3 Training

Use:

- Same tokenizer.
- Same training data.
- Same 250K-token target.
- Same seed.
- Same FP32 precision.
- Existing AdamW configuration.
- Existing training objective.
- Document boundaries and data alignment preserved.

The existing T1 training adapter uses bounded context and 64-step loss accumulation rather than recurrent activation carry.

Report this difference explicitly.

Do not claim that the Transformer and recurrent models have identical history access or training computation.

## 4.4 Evaluation

Evaluate:

- Validation NLL/PPL.
- Development Holdout NLL/PPL.
- Training convergence.
- Generation quality.
- Greedy generation.
- Sampling generation.
- Prefill latency.
- Decode latency.
- Memory usage.
- Context replay cost.

The Transformer must use the same fixed generation prompts and evaluation seeds.

## 4.5 Acceptance criteria

- T1 completes training.
- Model identity and checkpoint hashes are verified.
- Context limit is enforced.
- KV Cache remains disabled.
- Snapshot/Restore works.
- Generation outputs are preserved.
- Performance costs are explicitly reported.

If the execution budget is exhausted, mark T1 as INCOMPLETE or NOT_RUN.

Do not include incomplete results in architecture comparisons.

# 5. Priority 3 — Generation Quality Diagnosis

## 5.1 Objective

Determine why current C1 generation produces repetitive or incoherent text.

This priority must not automatically trigger additional large-scale training.

The first step is diagnosis using existing checkpoints and the newly trained baselines.

## 5.2 Models to evaluate

Primary comparison:

- Original C1.
- C1-BROADCAST.
- IC-ALL-SELF.
- F0-PM-BROADCAST.
- T1, if completed.

Historical diagnostic references:

- B1 Frozen No-Comm.
- Existing NO-COMM.
- Existing F0.

Keep historical and newly trained models clearly separated.

## 5.3 Fixed generation protocol

Use the existing 16 prompts.

For every model:

**Greedy**

- Temperature: 0
- Maximum tokens: 128

**Sampling**

- Temperature: 0.8
- Top-K: 40
- Top-P: 0.95
- Maximum tokens: 128

Use generation seed 713 and the existing per-prompt seed derivation.

Do not modify prompts or select prompts based on preliminary results.

Preserve all generated samples, including poor outputs.

## 5.4 Automatic metrics

Report the following for every model and decoding mode:

| Metric | Purpose |
|---|---|
| Validation NLL/PPL | Next-token prediction |
| Holdout NLL/PPL | Development generalization |
| Distinct-1 | Unigram diversity |
| Distinct-2 | Bigram diversity |
| Repeated 3-gram rate | Repetition |
| Mean token entropy | Readout uncertainty |
| Mean sampling entropy | Sampling distribution |
| Average generated length | Generation behavior |
| EOS rate | Natural termination |
| Tokens/s | Generation throughput |
| Prefill latency | Prompt processing |
| Decode latency | Token generation |

Do not treat low repetition as proof of coherent generation.

Do not treat low PPL as proof of good long-form generation.

## 5.5 Training diagnostics

Inspect and report:

- Initial training loss.
- Final training loss.
- First 50 update mean loss.
- Last 50 update mean loss.
- Loss progression.
- Gradient norm.
- Numerical stability.
- Validation progression.
- Development Holdout progression.
- EOS target prediction behavior.

Determine whether the models exhibit:

1. Insufficient training.
2. Training instability.
3. Validation degradation.
4. Excessive next-token confidence.
5. Repetitive attractor states.
6. Poor EOS calibration.

These are diagnostic hypotheses, not predetermined conclusions.

## 5.6 State diagnostics

For Cellular models, add:

- Per-Cell state norm.
- Nonzero Cell count.
- Pairwise Cell-state similarity.
- State norm progression during generation.
- State changes per generated event.
- State saturation or numerical-boundary proximity.

The purpose is to determine whether generation collapse correlates with:

- State homogenization.
- One or more dominant Cells.
- Loss of state diversity.
- Excessive state magnitude.
- Insufficient state evolution.

Do not modify the model transition during this diagnostic stage.

## 5.7 Human coherence evaluation

Automated metrics cannot fully assess language quality.

Prepare a blind review of the complete generated samples.

Reviewers should not see model identities during scoring.

Use a 1–5 scale for:

- Grammaticality.
- Local coherence.
- Semantic consistency.
- Prompt relevance.
- Story progression.
- Repetition and degeneration.

Preserve reviewer comments and model-independent sample IDs.

Where multiple reviewers are unavailable, explicitly report the limitation.

Do not select only favorable samples.

## 5.8 Conditional training extension

Do not retrain C1 automatically.

After completing diagnostics, determine whether the observed generation failure is consistent with undertraining.

A training extension may only be proposed if:

- Training loss is still meaningfully improving.
- Validation has not shown material deterioration.
- There is evidence that the model has not converged.
- The same extension budget can be applied to the selected comparison arms.
- A separate protocol and training identity are created.

If an extension is justified, create a separate experiment rather than modifying GC-001 or silently continuing its checkpoints.

No automatic 1M/10M-token expansion is permitted.

# 6. Data and Evaluation Integrity

Preserve the original GC-001 data audit.

Original data scope:

- Training documents: 1,827.
- Training prediction tokens: 416,027.
- Validation: development[:16].
- Development Holdout: development[16:].
- Formal independent Test: NOT_AVAILABLE.
- Tokenizer training overlap with Development: possible.

Rules:

1. Verify all original data asset hashes.
2. Preserve the original tokenizer.
3. Preserve the original validation and holdout boundaries.
4. Check document and token-sequence duplicates.
5. Reset state at every document boundary.
6. Never access sealed confirmation data.
7. Never claim independent-Test performance.
8. Never use the Development Holdout for hyperparameter selection while presenting it as an untouched test set.

All results must distinguish Validation from Development Holdout.

# 7. Execution and Resource Constraints

## 7.1 Execution batches

Recommended execution organization:

**Batch A — P1**

Train:

- C1-BROADCAST.
- IC-ALL-SELF.
- F0-PM-BROADCAST.

Reuse:

- Original C1 checkpoint.
- Original F0 checkpoint.
- Original data assets.

Then evaluate and publish the complete primary comparison.

**Batch B — P2**

Train and evaluate:

- Native T1 Transformer.

Reuse the published Batch A results.

Do not repeat completed C1 generation or baseline training.

**Batch C — P3**

Run the complete generation-quality analysis.

Reuse all compatible published model checkpoints and samples.

Do not repeat completed experiments.

## 7.2 Budget

Preserve the original GC execution limits:

- Execution budget: 5,400 seconds.
- Recovery reserve: 600 seconds.
- Publication reserve: 600 seconds.
- Safety reserve: 1,200 seconds.
- Notebook hard limit: 7,200 seconds.

Each batch must respect its own execution deadline.

No CPU fallback training.

No hidden additional training.

No automatic hyperparameter search.

No automatic expansion to larger models or token budgets.

## 7.3 Recovery

Preserve the existing authenticated checkpoint recovery mechanism.

Every new checkpoint must be bound to:

- Source identity.
- Protocol identity.
- Dataset identity.
- Architecture identity.
- Precision.
- Training seed.
- Target token count.

A checkpoint from a different architecture or protocol must never be restored as if it were compatible.

Incomplete runs must remain INCOMPLETE.

# 8. Repository Organization

Create a dedicated implementation branch:

`codex/epclm-gc001-baseline-repair`

Keep the original GC-001 implementation and results immutable.

Suggested implementation organization:

```text
research/projects/cellular-language-models/
  experiments/
    epclm-core-001/
      implementation/
        generative_baselines.py
        generation_evaluation.py
        tests/
      gc001-baseline-repair/
        protocol.json
        README.md
        architecture/
          independent_cellular.md
          flat_parameter_matched.md
        reports/
```

Suggested result namespaces:

```text
artifacts/experiments/epclm-core-001/
  gc001-baseline-repair/
    summary.json
    decision.json
    baseline-comparison.json
    generation-metrics.json
    generation-samples.json
    performance.json
    data-audit.json
    state-diagnostics.json
    human-evaluation.json
    training-curves.json
```

Transformer results should be identified separately but linked to the same primary experiment identity.

Use the existing metadata-only publication mechanism.

Do not publish:

- Raw training documents.
- Raw development documents.
- Model weights.
- Optimizer state.
- Runtime snapshots.
- Private credentials.

# 9. Required Tests

Before GPU execution, add CPU synthetic contract tests.

### Independent Cellular

- All 16 Cells receive events.
- Cell-specific event permutations are deterministic.
- No inter-cell state dependency.
- All 16 Cells can evolve.
- Left/center/right projections are active.
- Shared parameter structure is preserved.
- State and snapshot identity are correct.

### C1-BROADCAST

- Identical event input to IC-ALL-SELF.
- Neighbor communication is active.
- Cell ordering and ring boundary are correct.
- Transition remains synchronous.

### Flat Parameter-matched

- State dimension equals 2,048.
- All recurrent dimensions are active.
- No dead trainable parameters.
- Rank selection is deterministic.
- Parameter matching is within tolerance.
- Event input and output alignment are correct.

### Cross-model

- Identical tokenizer identity.
- Identical evaluation data.
- Identical prompt set.
- Identical generation configuration.
- Correct EOS behavior.
- Snapshot/Restore integrity.
- Checkpoint recovery identity validation.

All existing GC-001 tests must continue to pass.

# 10. Final Report Requirements

The final report must contain the following sections.

## A. Engineering Results

- Runtime compatibility.
- Model implementation.
- Training completion.
- Checkpoint recovery.
- Snapshot/Restore.
- Reproducibility.
- Performance.

## B. Baseline Comparison

Compare:

- Original C1.
- C1-BROADCAST.
- IC-ALL-SELF.
- Existing NO-COMM.
- Existing F0.
- F0-PM-BROADCAST.
- T1.

Report:

- Parameters.
- Active parameters.
- State dimensions.
- Training tokens.
- Validation NLL/PPL.
- Holdout NLL/PPL.
- Training time.
- Inference time.
- Estimated FLOPs.
- Memory.
- Generation metrics.

## C. Communication Analysis

The primary comparison is:

`C1-BROADCAST vs IC-ALL-SELF`

Explain whether communication provides measurable improvements under matched event input, state capacity and parameter structure.

Do not confuse this with the original C1 versus center-only NO-COMM comparison.

## D. State Architecture Analysis

Compare:

- Cellular distributed state.
- Independent Cellular state.
- Flat Recurrent state.
- Transformer bounded context.

Discuss state utilization, state persistence and history-access mechanisms separately.

## E. Generation Quality

Include all fixed prompts, generated samples, automated metrics and human review.

Explicitly identify failure cases.

## F. Scientific Conclusions

Separate findings into:

1. Engineering verified.
2. Empirically observed.
3. Hypotheses supported by preliminary evidence.
4. Not yet established.

A single training seed and development-only evaluation cannot establish architectural superiority.

Do not claim Cellular is superior to Transformer without repeated seeds, independent evaluation data and appropriately controlled comparisons.

# 11. Definition of Done

GC-001 Baseline Repair is complete when:

- [ ] P1 Independent Cellular is implemented and trained.
- [ ] P1 C1-BROADCAST is implemented and trained.
- [ ] P1 Parameter-matched Flat Recurrent is implemented and trained.
- [ ] Parameter matching and active-parameter checks pass.
- [ ] Original C1 and F0 remain immutable.
- [ ] P2 Native T1 completes or is explicitly marked incomplete.
- [ ] P3 automated generation diagnostics are complete.
- [ ] Human generation review is recorded or explicitly marked pending.
- [ ] All fixed evaluation samples are preserved.
- [ ] Data integrity checks pass.
- [ ] All artifacts are published with verified identities.
- [ ] Final report distinguishes engineering success from scientific evidence.

# 12. Final Research Direction

The purpose of this experiment is not to demonstrate that Cellular is better than Transformer.

It is to answer three more fundamental questions:

**1. Does Cellular Organization provide an advantage over a parameter-matched Flat Recurrent architecture?**

**2. Does communication between Cells provide an advantage over independent Cells when all Cells have access to the same events?**

**3. Can persistent Cellular State support coherent autoregressive language generation without retaining the full token history?**

These questions form the next experimental foundation for EPCLM.

GC-001 remains the original generative baseline.

GC-001 Baseline Repair establishes a more reliable comparison framework.

The subsequent research should build on these results rather than repeatedly rebuilding the same generation infrastructure.
