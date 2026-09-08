# Native CLM v0 — JAM Learning Demo

Status: **IMPLEMENTED / GPU RUN PENDING**

This demo starts from the canonical Native CLM v0 M1 checkpoint and performs bounded JAM post-training using `research/datasets/jam-knowledge-v0.1`.

## Goal

The demo is deliberately narrower than the Stage-06 continual-learning program. It asks only:

> Can the already-trained Native CLM acquire new, human-readable JAM knowledge after its initial TinyStories training?

A successful run supports the public demo statement:

> **A Native CLM can learn new JAM knowledge after its initial training.**

It does **not** reopen or rewrite the registered M2/M3 continual-learning decisions.

## Frozen base identity

The run refuses to train unless the starting checkpoint has SHA-256:

```text
91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

Default source:

```text
archelabs-org/native-clm-v0/final-model.pt
```

The root Hugging Face checkpoint is never overwritten. A successful JAM artifact is published under:

```text
archelabs-org/native-clm-v0/jam-v0.1/
```

## Data

The runner validates and deterministically materializes `jam-knowledge-v0.1` before training.

Learner-visible:

- 409 generated training examples;
- 180 generated validation examples for checkpoint selection.

Final evaluation:

- 180 factual examples;
- 66 relational examples;
- 49 misconception examples;
- 50 canonical learner-invisible cross-concept reasoning examples.

The 50 reasoning rows are never used for training or checkpoint selection.

## Training

This is ordinary full-model post-training, not the failed Granite narrow-expert mutation path and not a replay-free continual-learning protocol.

The default run uses:

- 1,200 optimizer steps;
- separate learning rates for shared/router/Cell parameter groups;
- answer-token-only JAM loss;
- 20% TinyStories rehearsal loss;
- generated JAM validation answer NLL for checkpoint selection;
- no reasoning-holdout feedback during training.

Because full-model post-training is allowed to update parameters protected by historical Stage-06 certificates, the JAM artifact must not be interpreted as carrying a valid replay-free certificate guarantee from M1/M2. The certificates remain checkpoint state only; this demo makes no certificate-safety claim.

## Benchmarks

The same base and selected JAM model are evaluated on:

1. JAM generated validation answer NLL / answer-token accuracy;
2. JAM factual answer metrics;
3. JAM relational answer metrics;
4. JAM misconception answer metrics;
5. learner-invisible JAM reasoning answer metrics;
6. TinyStories validation loss/perplexity;
7. six deterministic before/after greedy Q&A generations.

Teacher-forced metrics are the primary quantitative output. Free-generation Q&A is included for human-readable logs and screenshots.

## Hosted run

Kaggle prerequisites:

- GPU enabled;
- Internet enabled;
- `HF_TOKEN` Secret with write access to `archelabs-org/native-clm-v0`;
- `GITHUB_TOKEN` Secret with write access to `ArcheLabs/mini-cells`.

Notebook:

```text
research/notebooks/06-native-clm/native-clm-jam-demo-kaggle.ipynb
```

The notebook performs the complete workflow:

```text
canonical M1 checkpoint
  -> verify SHA-256
  -> build + validate JAM dataset
  -> BEFORE benchmarks / Q&A
  -> JAM + TinyStories rehearsal post-training
  -> select on JAM validation only
  -> AFTER benchmarks / Q&A
  -> fresh checkpoint reload
  -> upload jam-v0.1 to Hugging Face
  -> push lightweight results to GitHub
```

## Durable outputs

GitHub:

```text
artifacts/demos/native-clm-jam-v0.1/
  RESULTS.md
  benchmarks.json
  provenance.json
  QA_LOG.md
  training.csv
  HF_README.md
```

Hugging Face:

```text
jam-v0.1/
  final-model.pt
  README.md
  benchmarks.json
  provenance.json
  QA_LOG.md
```

Binary model weights are intentionally not committed to Git.

## Claim boundary

A successful run may be described as a learning demonstration of Native CLM v0. It must not be described as proof of replay-free continual learning, autonomous Cell growth, general JAM reasoning, or superiority over Transformer/MoE/LoRA fine-tuning.
