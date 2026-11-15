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
