# Portable Cell v0.1

## Source audit

Upstream is [sarahpannn/titans-pytorch at c60b4ef](https://github.com/sarahpannn/titans-pytorch/tree/c60b4efaa3a3c6df4deebd57cbccf75947b87cad).
`TitanLayerState.attention_kv` and its attention position are Session state.
The cross-layer value residual is forward-local. `NeuralMemState.weights`,
`states`, `updates`, and `seq_index` belong to the Cell. Its pending segment
and mask must be empty before commit. Outer NMM/Gate weights belong to the
immutable model bundle. The caller supplies the bundle SHA256 and a trusted
layout reference, including dtype and tensor shapes, to restore.

The upstream decoder layer sends a single position to both attention and
Neural Memory. Apply `upstream-position.patch` in the pinned upstream checkout
to expose a separate memory position for direct calls to `forward_core`.
The ordinary forward defaults to its existing position semantics. The bridge
stages one forward over Titan decoder layers. The end-to-end Hugging Face
generation route has not been validated with the bridge.

## Verification

| Status | Evidence |
| --- | --- |
| PORTABLE_CELL_STORAGE_PASS | CPU fixture, atomic Safetensors and corrupt payload rejection |
| PORTABLE_CELL_RESTORE_PASS | CPU fixture restored in a fresh interpreter against a separately constructed trusted layout |
| PORTABLE_CELL_FORK_PASS | CPU fixture, independent child tensor storage and lineage |
| PORTABLE_CELL_CONTINUE_PASS | Not established. The fixture verifies revision and position advancement only |
| PORTABLE_CELL_RUNTIME_PASS | Not established. The patch applies and compiles. No trained-bundle runtime smoke was run |
| PORTABLE_CELL_SEMANTIC_VALIDATION_BLOCKED | No compatible trained Llama 3.2 1B Instruct Titan-Llama NMM/Gate bundle was identified in the inspected checkout and upstream README |

Semantic acceptance still requires the trained bundle plus a fresh-process
comparison of recall and continued learning, including empty and donor
controls. No new training was launched.

Run the engineering fixtures with PyTorch, Safetensors and pytest installed.

```sh
python -m pytest -q research/portable-cell-v01
```

## Single-session GPU follow-up

Run [kaggle_runtime.py](kaggle_runtime.py) in a Kaggle GPU notebook with Internet
and an `HF_TOKEN` secret granted access to Llama 3.2 1B Instruct. The launcher
resolves the current model revision once and passes its immutable SHA to every
worker. The experiment uses one Titan decoder layer (8), FP32, seed 42, and
16-token chunks. It enables upstream history blending. The default training
fallback performs four outer AdamW steps on seeded synthetic token IDs and
checks that both the shared memory initialization and history gate changed.
The backbone is frozen. This is a reproducible engineering checkpoint.

Both comparisons run an in-process arm, an export worker, and a fresh import
worker. Both arms start a fresh Session before Chunk 2. The restore worker
derives the expected tensor layout from an unrelated input, discards that
state, and restores the saved Chunk 1 Cell. The report compares every state
tensor, the committed cursor, and the raw NMM retrieval vector using
`atol=1e-6` and `rtol=1e-5`. Results, checkpoint, and phase timings stay in
the saved Kaggle output. The first comparison provides a measured timing
reference for the rest of the workflow. The launcher uses a small fixed
training budget and has no run-time countdown.

The local checkout has no supported GPU or cached gated model weights. Before
running this entry, the engineering fixtures, patch application, and Python
compilation can be verified locally. The real GPU forward, short training,
and parity results are pending. A parity pass measures migration fidelity.
Semantic recall remains a separate experiment.
