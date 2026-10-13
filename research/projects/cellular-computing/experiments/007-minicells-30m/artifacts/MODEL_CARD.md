# MiniCells-30M v0

MiniCells-30M v0 is the first retained ~30M-parameter MiniCells language-model artifact.

## Architecture

- Parameters: 29,602,800
- Hidden dimension: 720
- Heads: 8
- FFN dimension: 2880
- Hierarchical causal windows: [8, 32, 128]
- Recurrent iterations: [4, 4, 4]
- GRU carry bias: 2.0
- Context length: 128
- Tokenizer vocabulary: 2,048

## Training

- Dataset: TinyStories
- Consumed training tokens: 100,000,000
- Optimizer: AdamW
- Base learning rate: 0.0003
- Warmup steps: 1,000
- Weight decay: 0.1
- Tokenizer SHA-256: `becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2`

## Result

- Status: `GREEN`
- Diagnosis: `MINICELLS_30M_PARAMETER_SCALING_COMPETITIVE`
- MiniCells PPL @100M: 5.3532
- Transformer PPL @100M: 5.3290
- PPL ratio @100M: 1.0045x
- Learning-slope ratio: 0.9940

## Artifact

- File: `minicells-30m-v0-fp16.pt`
- Bytes: 62,171,013
- SHA-256: `95c33bca902e571675994257368d5f791b3d7c8dd0832c3f5cf2a67c406f14ed`

The retained artifact stores FP16 weights plus the architecture configuration. It is intended
for inference and future model work; optimizer/resume checkpoints are intentionally not
published to Git.

## Scope

This model was trained only on TinyStories. It can generate simple story-like English text,
but it is not an instruction-following assistant and should not be described as a general
knowledge, reasoning, coding, or chat model.
