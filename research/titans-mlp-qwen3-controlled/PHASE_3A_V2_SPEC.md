# Phase 3A v2 — Learned Memory Addressing

**Implementation Specification**

**Project:** MiniCells / Titans-MLP-Qwen3-Controlled

**Status:** Ready for implementation

**Objective:** Replace direct frozen-hidden-state similarity with trainable, role-specific Query, Key and Value representations, and verify cross-template semantic addressing using a frozen Qwen3 backbone.

---

# 1. Executive Summary

## 1.1 Why Phase 3A v1 Failed

Phase 3A v1 attempted to use the hidden states of a frozen Qwen3-0.6B model directly as memory addresses.

The implementation compared:

- Full factual statement representations (Write).
- Natural-language question representations (Query).

Both were extracted from the final token or mean-pooled hidden states of Transformer block 23.

The formal five-seed experiment produced:

| Condition | Raw Cosine | Linear Probe |
|---|---:|---:|
| Seen-template | 12.97% | 28.91% |
| Held Query | 3.40% | 4.14% |
| Held Both | 3.79% | 3.44% |

The 32-way random Recall@1 baseline is 3.125%.

The principal finding is that a linear probe improves matching for familiar templates but does not generalize reliably to unseen query templates.

This does not prove that Qwen hidden states lack semantic information. It demonstrates that the current extraction and addressing mechanism does not establish a sufficiently compatible Query-Key space.

There are also experimental limitations:

1. Train, Validation and Test entity/value prefixes differ.
2. Write and Query use semantically asymmetric representations.
3. Held-out templates have limited linguistic diversity.
4. Value representation has not been independently validated.
5. No actual Titans Fast Weights retrieval was performed.

Reference:

- Formal results: `kaggle/phase3a-jara65tg`
- Result commit: `b2799943b1afc74da08acaedd68a7f529e8ef8bf`
- Executed source commit: `0351b200e29878d44e3ec830d68332c13698b552`

Preserve all v1 artifacts unchanged.

## 1.2 New Architectural Hypothesis

The new hypothesis is:

**A frozen pretrained language model can provide useful contextual representations, while a separately trained memory-addressing system learns how to transform those representations into compatible Query, Key and Value spaces.**

Instead of:

```text
Qwen Hidden State
       |
       +----------------+
       |                |
       v                v
    Write             Query
       |                |
       +------ Cosine --+
                |
                v
          Memory Address
```

Implement:

```text
                    Frozen Qwen
                         |
                  Contextual States
                         |
              Role-aware Extraction
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
           Write       Query       Value
             |           |           |
             v           v           v
            W_K         W_Q         W_V
             |           |           |
             v           v           v
           Key         Query       Value
             |           |           |
             +---- Address Match ---+
                         |
                         v
                Learned Address Space
                         |
                         v
                  Candidate Retrieval
```

The three projections have distinct responsibilities:

- `W_K`: transform write representations into memory keys.
- `W_Q`: transform read representations into compatible query addresses.
- `W_V`: transform stored information into a stable value representation.

The Qwen backbone remains frozen.

Only the small addressing adapters are trained.

## 1.3 Relationship to Titans-Llama

This experiment adopts an important architectural principle from Titans-style model integration:

**Memory representations should be learned as part of the model's computational architecture rather than assumed to be directly available from arbitrary pretrained hidden states.**

However, Phase 3A v2 is not a reproduction of the complete Titans-Llama architecture.

The following remain out of scope:

- Full language-model training objective.
- Titans recurrent Fast Weights.
- Online memory updates.
- Memory read/write integration into Transformer attention.
- Learned memory fusion and output gating.
- Full Outer Training.

Phase 3A v2 is specifically a controlled experiment for learned memory representations.

---

# 2. Experimental Scope and Invariants

## 2.1 Model

Use exactly the same model as Phase 3A v1.

| Property | Required value |
|---|---|
| Model | Qwen/Qwen3-0.6B-Base |
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Hidden size | 1024 |
| Decoder layers | 28 |
| Extraction layer | Index 22 |
| Extraction position | Post-block |
| Precision | FP32 |
| Backbone | Frozen |
| Initial device | CPU |

Model and configuration SHA256 verification must remain identical to the existing `audit_qwen.py` requirements.

The extraction process must use the same tokenizer revision and configuration.

## 2.2 Frozen Backbone

The following are prohibited:

- Updating Qwen weights.
- Enabling gradients through Qwen.
- Changing model revision.
- Changing tokenizer revision.
- Replacing the model with another checkpoint.
- Modifying the Transformer architecture.

Use:

```python
model.eval()
model.requires_grad_(False)

with torch.inference_mode():
    ...
```

The extracted feature cache must be generated once and reused during adapter training.

Do not repeatedly run Qwen during every adapter training step.

## 2.3 Experimental Variables

The following are allowed to change:

- Representation extraction.
- Pooling.
- Query/Key/Value projection heads.
- Addressing loss.
- Adapter initialization.
- Adapter optimizer and learning rate.
- Candidate retrieval evaluation.

All changes must be isolated from Phase 3A v1.

---

# 3. New Representation Architecture

## 3.1 Role-specific Inputs

The previous experiment used one generic hidden representation for three different semantic roles.

This must change.

We now explicitly extract:

- Entity representation.
- Relation representation.
- Value representation.

For a fact:

```text
E088881 works on V047873.
```

The intended semantic structure is:

```text
Entity:   E088881
Relation: works on
Value:    V047873
```

The Write Key must contain Entity and Relation information without leaking the Value.

The Query Key must contain the corresponding Entity and Relation information.

The Value must be represented independently.

## 3.2 Token-span Extraction

Use tokenizer offset mappings to locate:

- Entity span.
- Relation span.
- Value span.

Templates must provide exact character spans.

For Qwen's causal decoder, use the following primary extraction:

```python
h_entity = mean_pool(hidden[entity_span])

h_relation = hidden[last_token_of_relation_span]

h_address = concat(h_entity, h_relation)

h_value = mean_pool(hidden[value_span])
```

The resulting dimensions are:

| Representation | Dimension |
|---|---:|
| Entity | 1024 |
| Relation anchor | 1024 |
| Address packet | 2048 |
| Value | 1024 |

Write and Query must use the same extraction contract.

For Write:

- Entity must precede Relation.
- Relation must precede Value.

For Query:

- Entity must precede the Relation phrase.

This ensures the relation-anchor hidden state can attend to the Entity and Relation information.

The Value must never be included in the Key packet.

If a template violates these ordering constraints, reject it during dataset validation rather than silently changing pooling behavior.

## 3.3 Extraction Ablations

Implement these extraction variants:

| Variant | Description |
|---|---|
| V1 Last-token | Original Phase 3A v1 method |
| V1 Mean | Original full-text mean pooling |
| Relation-anchor | Last hidden state of Relation span |
| Role-packet | Entity mean + Relation anchor |
| Entity-Relation mean | Mean pooling over both annotated spans |

The primary experiment uses:

`Role-packet`

The other methods are diagnostic baselines.

Do not select the best extraction method using Test results.

## 3.4 Value Extraction

Extract Value independently.

For Write statements, use the hidden states corresponding to the Value token span.

Also generate independent Value contexts, for example:

```text
Recorded value: V047873.

The stored identifier is V047873.

Value assigned to this record: V047873.
```

These provide different contextual views of the same Value.

The Value encoder must learn to produce compatible vectors for the same Value across these contexts.

---

# 4. Learned Addressing Adapter

## 4.1 Primary Architecture

Implement three independent projections.

```python
class AddressingAdapter(nn.Module):
    def __init__(self, input_dim=2048, value_dim=1024,
                 projection_dim=256):
        super().__init__()

        self.query = nn.Linear(
            input_dim, projection_dim, bias=False
        )

        self.key = nn.Linear(
            input_dim, projection_dim, bias=False
        )

        self.value = nn.Linear(
            value_dim, projection_dim, bias=False
        )

        self.query_norm = nn.LayerNorm(projection_dim)
        self.key_norm = nn.LayerNorm(projection_dim)
        self.value_norm = nn.LayerNorm(projection_dim)

    def encode_query(self, x):
        return F.normalize(
            self.query_norm(self.query(x)), dim=-1
        )

    def encode_key(self, x):
        return F.normalize(
            self.key_norm(self.key(x)), dim=-1
        )

    def encode_value(self, x):
        return F.normalize(
            self.value_norm(self.value(x)), dim=-1
        )
```

This is the primary linear architecture.

Projection dimension:

`256`

All output representations must be finite and L2-normalized.

## 4.2 Why Separate WQ and WK?

A shared projection assumes that Write and Query representations already have compatible statistical structures.

That assumption was not established by Phase 3A v1.

Separate projections allow the system to learn:

\[
Q = \operatorname{normalize}(W_Q h_Q)
\]

\[
K = \operatorname{normalize}(W_K h_K)
\]

The objective is:

\[
\operatorname{sim}(Q_i,K_i)
>
\operatorname{sim}(Q_i,K_j),\quad i\ne j
\]

This does not require the original hidden states to have similar geometry.

The adapter learns the compatibility between the two roles.

## 4.3 Shared Projection Baseline

Also implement:

```text
W_shared(h_query)
W_shared(h_write)
```

This baseline tests whether separate Query and Key projections provide a measurable advantage over a shared encoder.

## 4.4 MLP Projection Ablation

Implement a small MLP alternative:

```text
Linear(input_dim, 512)
       |
      GELU
       |
    Dropout(0.1)
       |
Linear(512, 256)
       |
    LayerNorm
       |
 L2 Normalize
```

Use independent MLPs for Query and Key.

This is an ablation, not the primary architecture.

Do not increase model capacity arbitrarily.

## 4.5 Value Projection

Use:

\[
V = \operatorname{normalize}(W_V h_V)
\]

The Value projection must be independently trainable.

Its objective is to produce a stable representation of the same Value across different contextual forms.

It is not a language decoder.

Do not claim that a successful Value embedding experiment proves text reconstruction.

---

# 5. Dataset v2

## 5.1 Dataset Scale

Preserve the previous formal dataset scale.

| Split | Facts | Entities |
|---|---:|---:|
| Train | 1,024 | 256 |
| Validation | 256 | 64 |
| Test | 512 | 128 |

Each Entity has exactly four associated Relations selected from eight possible Relation types.

This provides:

- Same-Entity / different-Relation hard negatives.
- Same-Relation / different-Entity hard negatives.
- Unknown Entity queries.
- Unknown Entity-Relation combinations.

The Relation distribution must be balanced across each split.

Each of the eight Relations should occur equally often within each split.

Use a deterministic randomized assignment algorithm.

Do not associate a particular Entity ID range with a particular Relation.

## 5.2 Entity and Value Identifiers

Use a consistent format across all splits.

Examples:

```text
E088881
E126423
E195948

V047873
V161123
V188401
```

Do not encode Train, Validation or Test in identifiers.

The following are prohibited:

```text
E0xxxxx
E1xxxxx
E2xxxxx
```

Entity and Value identifiers must come from independent random pools.

Requirements:

- No duplicate Entities.
- No duplicate Values.
- No Entity/Value overlap.
- No cross-split Entity overlap.
- No cross-split Value overlap.
- No deterministic relationship between Entity and Value numeric suffixes.
- Verify tokenizer-level identifier uniqueness.

## 5.3 Relations

Retain the original eight semantic relations:

1. works on
2. belongs to
3. is assigned to
4. operates
5. reports to
6. uses
7. manages
8. is located in

Do not change the semantic meaning of these relations.

## 5.4 Template Families

The previous held-out templates were insufficiently diverse.

Create three independent template families:

| Family | Purpose | Usage |
|---|---|---|
| A | Training templates | Train |
| B | Development templates | Validation |
| C | Confirmatory templates | Test |

For every Relation and every role (Write and Query), provide at least three distinct fluent templates per family.

This means:

- At least 72 Write templates.
- At least 72 Query templates.

Templates must be authored and reviewed for grammatical correctness.

Do not mechanically substitute relation names into generic templates if the resulting sentence is unnatural.

Each template must include:

```python
{
    "template_id": "...",
    "family_id": "A",
    "relation_id": "...",
    "role": "write",
    "text": "...",
    "entity_span": [start, end],
    "relation_span": [start, end],
    "value_span": [start, end]
}
```

For Query templates, Value spans are absent.

Template families must differ in actual linguistic construction, not merely punctuation or introductory phrases.

Do not use the same template with reordered words and call it an independent family.

## 5.5 Template Split Protocol

Train:

- Write family A.
- Query family A.

Validation:

- A/A: Seen-template validation.
- A/B: Held Query validation.
- B/A: Held Write validation.
- B/B: Held Both validation.

Test:

- A/A: Seen-template.
- A/C: Held Query.
- C/A: Held Write.
- C/C: Held Both.

Family C must not be used during training or hyperparameter selection.

The primary generalization conditions are:

- A/C.
- C/C.

Do not combine them into a single metric.

## 5.6 Multiple Views Per Fact

Each fact must have multiple Write and Query expressions.

During each training epoch:

- Randomly sample one Write expression.
- Randomly sample one Query expression.
- Sample them independently.

This prevents the model from learning a fixed pairing between two particular sentences.

For evaluation, use multiple combinations of Write and Query templates.

Aggregate first by fact, then by Entity, then by Seed.

Do not count multiple paraphrases of the same fact as independent facts.

## 5.7 Value Context Dataset

For each fact, create at least three Value contexts.

Example:

```text
Recorded value: V047873.

The stored identifier is V047873.

Value assigned to this record: V047873.
```

Training values must be disjoint from Validation and Test values.

Use family A for training, B for validation and C for testing.

---

# 6. Training Objectives

## 6.1 Query-Key Contrastive Learning

The primary objective is symmetric InfoNCE.

For a batch of N paired facts:

\[
S_{ij}=\frac{Q_i K_j^\top}{\tau}
\]

Use:

\[
L_{Q\rightarrow K}
=
CE(S,\operatorname{diag})
\]

\[
L_{K\rightarrow Q}
=
CE(S^\top,\operatorname{diag})
\]

Then:

\[
L_{QK}
=
\frac{1}{2}
(L_{Q\rightarrow K}+L_{K\rightarrow Q})
\]

Initial temperature:

\[
\tau=0.07
\]

The positive pair is the Query and Write Key belonging to the same Entity-Relation fact.

All other facts in the batch act as negatives.

## 6.2 Hard Negative Sampling

Training batches must explicitly contain hard negatives.

Use a balanced batch sampler.

Recommended batch:

- 32 Entities.
- 4 facts per Entity.
- 128 total facts.

This naturally produces:

1. Same Entity, different Relation.
2. Different Entity, same Relation.
3. Different Entity, different Relation.

These are more informative than purely random negatives.

Do not use only random unrelated negatives.

## 6.3 Value Contrastive Learning

Train the Value encoder independently.

For two contextual views of the same Value:

\[
V_i^{(1)}, V_i^{(2)}
\]

Use symmetric InfoNCE:

\[
L_V =
\frac{1}{2}
\left[
CE(V^{(1)}V^{(2)\top}/\tau)
+
CE(V^{(2)}V^{(1)\top}/\tau)
\right]
\]

Use the same initial temperature of 0.07.

The complete objective is:

\[
L=L_{QK}+L_V
\]

The Query/Key and Value heads have independent parameters.

Record the two losses separately.

Do not allow Value supervision to use Test identifiers or Test contexts.

## 6.4 Optimizer

Primary configuration:

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 128 |
| Maximum epochs | 100 |
| Early stopping patience | 10 |
| Gradient clipping | 1.0 |
| Temperature | 0.07 |
| Projection dimension | 256 |

A small learning-rate comparison is allowed:

- 0.0003
- 0.001

Select using Validation only.

Do not tune using Test.

## 6.5 Training Seeds

Use:

```text
42
43
44
45
46
```

Each seed must independently control:

- Adapter initialization.
- Batch sampling.
- Template selection.
- Training order.

All baselines within a seed must use the same generated dataset and candidate sets.

## 6.6 Early Stopping

Primary early-stopping metric:

Validation Held Query MRR.

Secondary tie-breaker:

Validation Held Both Recall@1.

Value encoder uses its own Validation Value Recall@1.

Save:

- Best checkpoint.
- Final checkpoint.
- Optimizer state.
- Training history.
- Validation history.
- Random number generator state.

Test results must not influence checkpoint selection.

---

# 7. Baselines and Ablations

All methods must use identical facts, candidate pools and test conditions.

Implement:

| Method | Description |
|---|---|
| Random | Random candidate ranking |
| Raw Cosine | Original Qwen representations |
| V1 Ridge | Original Phase 3A linear probe |
| Shared Linear | One shared learned projection |
| Separate Linear | Independent WQ and WK |
| Separate MLP | Independent MLP Query/Key encoders |
| Canonical Oracle | Exact Entity-Relation matching |

The Canonical Oracle is only an upper-bound and dataset correctness check.

It is not a learned model.

The primary comparison is:

**V1 Ridge vs Shared Linear vs Separate Linear.**

The MLP is a secondary capacity ablation.

## 7.1 Required Ablations

1. Shared versus separate Query/Key projections.
2. Linear versus MLP projections.
3. Last-token versus Role-packet extraction.
4. Relation-anchor versus Role-packet extraction.
5. With and without hard-negative batch construction.
6. Symmetric versus one-directional contrastive loss.
7. Query/Key versus Value encoder performance.

Do not run arbitrary Transformer-layer searches during the confirmatory experiment.

The primary layer remains block index 22.

Any alternative layer investigation must be separately reported as exploratory research.

---

# 8. Evaluation Protocol

## 8.1 Candidate Pool Sizes

Evaluate:

- 32-way.
- 128-way.
- 512-way.

For the 512 Test facts:

- 32-way: 8 Entities × 4 facts.
- 128-way: 32 Entities × 4 facts.
- 512-way: all 128 Test Entities × 4 facts.

Every candidate pool must include the correct fact and preserve the intended hard-negative composition.

Candidate ordering must be deterministically randomized.

Do not place the correct candidate at a fixed index.

## 8.2 Retrieval Conditions

Evaluate separately:

| Condition | Write | Query |
|---|---|---|
| Seen | A | A |
| Held Query | A | C |
| Held Write | C | A |
| Held Both | C | C |

The primary conditions are Held Query and Held Both.

## 8.3 Address Retrieval Metrics

Report:

- Recall@1.
- Recall@5.
- MRR.
- Mean Rank.
- Median Rank.
- Positive Similarity.
- Hardest Negative Similarity.
- Positive-Negative Margin.

The margin is:

\[
Margin = Score(Q,K^+)-\max Score(Q,K^-)
\]

A positive average margin is desirable.

Do not interpret high cosine similarity alone as successful retrieval.

## 8.4 Unknown Query Evaluation

Construct two types of unknown queries:

1. Unknown Entity with a known Relation.
2. Known Entity with a Relation that is not associated with that Entity.

Both must have no correct candidate in the memory candidate set.

Report:

- False Positive Rate.
- AUROC.
- Precision.
- Recall.
- Maximum similarity distribution.
- Rejection accuracy.

Choose the rejection threshold using Validation only.

Do not adjust it after inspecting Test results.

## 8.5 Value Retrieval

Evaluate the Value encoder independently.

Given a Value context, rank candidate Value vectors.

Report:

- Value Recall@1.
- Value Recall@5.
- Value MRR.
- Value positive-negative margin.
- Value cosine similarity.

Evaluate at 32, 128 and 512 candidates.

This only validates Value representation consistency.

It does not validate text generation or Titans memory reconstruction.

---

# 9. Statistical Analysis

## 9.1 Per-Seed Results

Every formal metric must be reported independently for Seeds 42–46.

Report:

- Mean.
- Standard Deviation.
- 95% Confidence Interval.

Use seed-level results as independent experimental runs.

## 9.2 Paired Comparison

Compare each learned method against:

- Raw Cosine.
- V1 Ridge.

Use paired evaluations on identical test facts and candidate pools.

Report the improvement in percentage points.

Use 10,000 paired bootstrap resamples.

Since each Entity has multiple Relations, bootstrap by Entity rather than treating all facts as fully independent.

The same sampled Entity groups must be used for both methods in each paired comparison.

## 9.3 Statistical Baselines

For N candidates:

\[
RandomRecall@1=\frac{1}{N}
\]

\[
RandomMRR=\frac{H_N}{N}
\]

where:

\[
H_N=\sum_{r=1}^{N}\frac{1}{r}
\]

Report the correct random baseline for every candidate size.

Do not reuse the 32-way baseline for 128-way or 512-way experiments.

---

# 10. Acceptance Gates

Freeze these gates before formal Test execution.

## 10.1 Addressing Gate

For Separate Linear on the primary Role-packet representation:

| Metric | Target |
|---|---:|
| Seen 32-way Recall@1 | ≥ 90% |
| Held Query 32-way Recall@1 | ≥ 80% |
| Held Both 32-way Recall@1 | ≥ 80% |
| Held Query MRR | Above random baseline |
| Held Both MRR | Above random baseline |
| Unknown Query FPR | ≤ 5% |

Additionally:

- Held Query Recall@1 must improve over Raw Cosine by at least 10 percentage points.
- Held Both Recall@1 must improve over Raw Cosine by at least 10 percentage points.
- The paired 95% confidence interval for both improvements must have a lower bound above zero.
- Results must be consistent across the five seeds.
- All confidence intervals and raw per-seed metrics must be published.

The mean Recall targets are engineering acceptance thresholds. The confidence intervals are reported separately; do not confuse a high mean with statistically established improvement.

## 10.2 Scaling Gate

Scaling is a secondary acceptance dimension.

Targets:

| Candidates | Held Query Recall@1 |
|---|---:|
| 32 | ≥ 80% |
| 128 | ≥ 70% |
| 512 | ≥ 50% |

If the 32-way gate passes but larger candidate pools fail, report:

`ADDRESSING_PASS_WITH_CAPACITY_LIMITATION`

Do not claim large-scale addressing capability.

## 10.3 Value Gate

Separate Value encoder targets:

| Candidates | Value Recall@1 |
|---|---:|
| 32 | ≥ 90% |
| 128 | ≥ 80% |
| 512 | ≥ 60% |

The Value encoder must also show positive margins and no numerical instability.

## 10.4 Final Decisions

Use explicit machine-readable decisions:

- `PASS`
- `PASS_WITH_CAPACITY_LIMITATION`
- `ADDRESSING_PASS_VALUE_FAIL`
- `FAIL`
- `INCOMPLETE`
- `NOT_TESTED`

Overall `PHASE_3A_V2_PASS` requires both Addressing and Value gates.

If Addressing passes but Value fails, retain the separate result. Do not open Phase 3B automatically.

If all gates fail, do not modify thresholds to force a PASS.

---

# 11. Representation Collapse and Numerical Diagnostics

Monitor:

- Query vector norms.
- Key vector norms.
- Value vector norms.
- Mean pairwise cosine similarity.
- Positive versus negative similarity distributions.
- Per-dimension variance.
- Covariance effective rank.
- Gradient norms.
- Projection weight norms.
- Non-finite values.

The model must not satisfy contrastive training by collapsing all vectors toward a common direction.

L2 normalization alone does not prevent directional collapse.

If collapse occurs:

- Save the failing checkpoint.
- Record the relevant diagnostics.
- Do not silently introduce new losses.
- Treat a new loss or architecture as a separate experiment.

---

# 12. Implementation Structure

Create a new isolated implementation:

```text
research/titans-mlp-qwen3-controlled/
│
├── PHASE_1.md
├── PHASE_2.md
├── PHASE_3A.md
├── PHASE_3A_V2.md
│
├── phase3a.py
├── summarize_phase3a.py
│
└── phase3a_v2/
    │
    ├── README.md
    ├── config.json
    │
    ├── dataset/
    │   ├── generate.py
    │   ├── templates.json
    │   └── validate.py
    │
    ├── representations/
    │   ├── extract.py
    │   ├── spans.py
    │   └── pooling.py
    │
    ├── models/
    │   ├── addressing.py
    │   └── losses.py
    │
    ├── training/
    │   ├── sampler.py
    │   └── train.py
    │
    ├── evaluation/
    │   ├── candidates.py
    │   ├── retrieval.py
    │   ├── unknown.py
    │   └── statistics.py
    │
    ├── tests/
    │   ├── test_dataset.py
    │   ├── test_spans.py
    │   ├── test_addressing.py
    │   ├── test_losses.py
    │   ├── test_training.py
    │   └── test_evaluation.py
    │
    └── results/
```

Reuse existing utilities where appropriate.

Do not duplicate or modify Phase 1 memory implementation unnecessarily.

---

# 13. Tests

The following tests are mandatory.

## Dataset

- Entity disjointness.
- Value disjointness.
- Tokenized identifier uniqueness.
- Relation balance.
- Relation assignment balance per Entity.
- Template family separation.
- Template grammatical and span validation.
- No duplicate rendered facts.
- No accidental split-specific prefixes.
- Candidate pool correctness.

## Representation Extraction

- Correct model revision.
- Correct layer index.
- Correct hidden dimension.
- Correct span alignment.
- No empty spans.
- No overlapping Entity/Relation/Value spans.
- Write Value is excluded from Key.
- Query and Write output dimensions match.
- All vectors finite.

## Addressing Adapter

- WQ/WK/WV output dimensions.
- Independent parameter ownership.
- Correct L2 normalization.
- Shared projection baseline.
- Separate projection baseline.
- Gradient propagation through adapters.
- No gradient propagation into Qwen.
- Deterministic initialization under fixed seeds.

## Loss

- Positive pairs receive correct labels.
- In-batch negatives are correctly constructed.
- Symmetric loss is finite.
- Gradient is finite.
- No train/test leakage.

## Evaluation

- Correct candidate indexing.
- Correct Recall@K.
- Correct MRR.
- Correct random baseline.
- Correct paired bootstrap.
- Unknown query evaluation.
- No Test-based threshold selection.

---

# 14. Resource Plan

## 14.1 CPU First

The first complete implementation must run on CPU.

Requirements:

- FP32.
- Frozen Qwen.
- Cached representations.
- Small adapter only.
- Deterministic execution.
- Fixed PyTorch thread count.

Feature extraction must not be repeated during adapter epochs.

## 14.2 GPU Optional

After CPU Smoke passes, T4 execution may be used for faster formal runs.

GPU is an optional acceleration mechanism.

It must not change:

- Model revision.
- Model weights.
- Tokenizer.
- Precision.
- Dataset.
- Training objective.
- Acceptance thresholds.

Record CPU/GPU numerical differences.

Do not make GPU availability a prerequisite for implementation correctness.

## 14.3 Resource Recording

Record:

- Model loading time.
- Feature extraction time.
- Cache size.
- Adapter training time.
- Peak RAM.
- Peak GPU memory if applicable.
- Checkpoint size.
- Evaluation time.

---

# 15. Reproducibility and Artifact Archiving

Every formal run must preserve the complete experiment.

Suggested output:

```text
artifacts/experiments/
  titans-mlp-qwen3-controlled/
    phase3a-v2-<run-id>/
      manifest.json
      config.json
      templates.json
      dataset/
        train.json
        validation.json
        test.json
        hashes.json
      features/
        manifest.json
      checkpoints/
        seed-42.safetensors
        seed-43.safetensors
        seed-44.safetensors
        seed-45.safetensors
        seed-46.safetensors
      metrics/
        per-seed.json
        per-relation.json
        per-template.json
        per-capacity.json
      logs/
      summary.json
      decision.json
      RESULTS.md
```

The exact artifact directory may follow the existing repository conventions, but the information above must be retained.

## 15.1 Manifest

Record:

```json
{
  "experiment": "phase3a-v2",
  "model": "Qwen/Qwen3-0.6B-Base",
  "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
  "hidden_layer": 22,
  "hidden_size": 1024,
  "projection_size": 256,
  "backbone_frozen": true,
  "backbone_training_updates": 0,
  "fast_weight_updates": 0,
  "outer_training_updates": 0,
  "seeds": [42, 43, 44, 45, 46],
  "device": "cpu",
  "dtype": "float32",
  "dataset_sha256": "...",
  "template_sha256": "...",
  "model_sha256": "...",
  "tokenizer_sha256": "...",
  "source_commit": "...",
  "config_sha256": "..."
}
```

Also record:

- Python version.
- PyTorch version.
- Transformers version.
- Tokenizers version.
- Actual executed script hashes.
- Training configuration.
- Per-seed checkpoint hashes.

## 15.2 Kaggle Result Publication

The experiment runner must produce a durable output directory.

Do not rely on a temporary cloned repository as the only location of results.

The runner must:

1. Complete the experiment.
2. Validate all expected files.
3. Calculate SHA256 for every artifact.
4. Generate a complete archive.
5. Copy the archive into the durable Kaggle output directory.
6. Produce a publication manifest.
7. Verify the published artifact by reading it back.

GitHub publication may be a separate operation.

If GitHub publication fails, the Kaggle output must remain recoverable.

Never report experiment completion as artifact publication success.

---

# 16. Execution Order

Codex must execute the work in the following order.

## Step 1 — Repository Audit

Inspect:

- `research/titans-mlp-qwen3-controlled/phase3a.py`
- `summarize_phase3a.py`
- `audit_qwen.py`
- `research/titans/stage6d/D1_QK_REPRESENTATION.md`
- `research/titans/stage6d/D2_MEMORY_ADDRESSING.md`
- `research/titans/stage7/run_stage7.py`

The existing Stage 6D/Stage 7 design already contains separate addressing and value projections.

Reuse relevant concepts and utilities.

However, do not copy model-specific assumptions blindly.

In particular, the existing Stage 7 Qwen3.5 implementation uses a different native attention geometry from the pinned Qwen3-0.6B experiment.

The v2 implementation must respect the actual Qwen3-0.6B hidden-state interface.

Before implementation, produce a short audit describing:

- Existing reusable modules.
- Differences between the two model architectures.
- New modules required.
- Potential compatibility issues.

## Step 2 — Dataset v2

Implement and validate the dataset generator.

Generate a complete manifest.

Run all dataset leakage tests.

Do not proceed if any split or template isolation check fails.

## Step 3 — Feature Extraction

Extract and cache:

- Write Key packets.
- Query Key packets.
- Value representations.
- Original v1 representations for baseline comparison.

Verify model and tokenizer hashes.

Ensure no Value leakage into Key features.

## Step 4 — Adapter Implementation

Implement:

- Raw baseline.
- V1 Ridge baseline.
- Shared Linear projection.
- Separate Linear WQ/WK.
- Separate MLP WQ/WK.
- W_V.

Implement all unit tests before training.

## Step 5 — CPU Smoke

Use one seed and a reduced dataset.

Verify:

- Training runs.
- Gradients are finite.
- Backbone remains frozen.
- Query/Key compatibility improves on training data.
- Validation metrics are generated.
- Checkpoints can be saved and restored.
- Test data remains sealed.

Smoke results must not be used to modify formal acceptance thresholds.

## Step 6 — Formal Five-seed Training

Execute Seeds 42–46.

Run all pre-registered methods.

Use Validation for:

- Learning-rate selection.
- Early stopping.
- Rejection threshold selection.

Do not use Test for any training or model-selection decision.

## Step 7 — Formal Evaluation

Evaluate all four template conditions.

Evaluate 32, 128 and 512 candidates.

Evaluate Value representations separately.

Run unknown-query rejection tests.

## Step 8 — Independent Result Validation

The summarizer must independently verify:

- Five completed seeds.
- Correct source hashes.
- Correct model hashes.
- Correct dataset hashes.
- All required conditions.
- All required candidate sizes.
- All required metrics.
- No missing or duplicate result records.

Recompute summary metrics from raw per-query results.

Do not trust a precomputed summary without independent verification.

## Step 9 — Archive and Report

Generate:

- `RESULTS.md`
- `summary.json`
- `decision.json`
- Per-seed metrics.
- Per-relation metrics.
- Per-template metrics.
- Artifact manifest.
- Checkpoint hashes.

Verify the archived result by reading it back.

---

# 17. Explicit Exclusions

Phase 3A v2 must NOT:

1. Train Qwen weights.
2. Modify the Qwen architecture.
3. Modify Titans MLP recurrence.
4. Execute Titans Fast Weights retrieval.
5. Perform online memory writes.
6. Start Titans Outer Training.
7. Integrate memory into the Transformer attention path.
8. Train a full language model.
9. Introduce GPU as a mandatory dependency.
10. Modify Phase 3A v1 historical artifacts.
11. Use Test results for hyperparameter selection.
12. Change acceptance thresholds after formal execution.

This is a learned addressing experiment, not yet a persistent memory experiment.

---

# 18. Final Research Questions

The final report must answer these questions explicitly.

### Q1. Does a learned shared projection outperform raw hidden-state similarity?

This tests whether a common address transformation is sufficient.

### Q2. Do independent WQ and WK provide a meaningful advantage?

This tests whether Write and Query require different transformations.

### Q3. Does the learned address space generalize to unseen natural-language templates?

This is the primary research question.

### Q4. Does role-aware extraction outperform the original full-text last-token representation?

This tests whether explicitly separating Entity, Relation and Value resolves the v1 representation asymmetry.

### Q5. Can W_V produce stable Value representations for unseen identifiers?

This establishes whether the Value interface is ready for subsequent memory experiments.

### Q6. Does performance remain usable as candidate capacity increases?

This tests whether the addressing space can scale beyond a small retrieval task.

### Q7. What is the remaining gap between a learned addressing adapter and a complete Titans memory architecture?

The report must explicitly distinguish:

- Learned address alignment.
- Value representation.
- Fast Weights retrieval.
- Persistent memory.
- Language-model integration.

A successful Phase 3A v2 does not prove the latter three.

---

# 19. Go/No-Go Decision

The final decision must be:

| Result | Next action |
|---|---|
| Addressing and Value both pass | Open Phase 3B design review |
| Addressing passes, Value fails | Improve Value representation before Phase 3B |
| Seen passes, Held Query fails | Investigate semantic generalization |
| Shared passes, Separate fails | Re-evaluate role-specific architecture |
| Separate passes, Shared fails | Retain independent WQ/WK |
| All learned methods fail | Revisit representation extraction and training objective |
| Candidate scaling fails | Investigate capacity and address collisions |

Do not automatically start Phase 3B.

Phase 3B must receive a separately reviewed implementation specification.

---

# 20. Codex Deliverables

The implementation is complete only when the following exist:

1. Phase 3A v2 implementation.
2. Dataset v2 and leakage tests.
3. Role-aware representation extraction.
4. Trainable WQ/WK/WV adapters.
5. Contrastive training implementation.
6. Raw, Ridge and learned-projection baselines.
7. Five-seed experiment runner.
8. Candidate retrieval evaluator.
9. Value retrieval evaluator.
10. Unknown-query evaluator.
11. Statistical analysis.
12. CPU Smoke results.
13. Formal five-seed results.
14. Complete artifact archive.
15. `RESULTS.md`.
16. Machine-readable final decision.

Preserve Phase 3A v1 completely.

Do not modify Phase 1 or Phase 2.

Do not start Phase 3B, Phase 3C or Outer Training.

**The goal is to establish whether a small, trainable addressing system can transform frozen Qwen representations into a generalizable memory address space.**

Only after this is demonstrated should Titans Fast Weights be introduced into the experiment.