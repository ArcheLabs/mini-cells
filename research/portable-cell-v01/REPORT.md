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

## v0.2 GPU entry

For a Kaggle Python script or code cell, paste the complete contents of
[`kaggle_runtime.py`](kaggle_runtime.py). The optional .ipynb file is for
Notebook import. Enable GPU and Internet, configure `GITHUB_TOKEN`, and
select Save & Run All. The launcher streams the worker log to Kaggle and saves it as raw.log. The model snapshot is
resolved once and hashed, and all subprocesses use that same local snapshot.
The runtime follows the pinned upstream seg512 architecture on layers 0–3.
FP32 uses standard attention kernels.

Bootstrap uses the upstream trainer and LongAlign/FineWeb route. The default
plan is 12 updates at 1024 tokens, batch size 1 and the upstream warmup.
The trainer writes its native checkpoint after every update. Wall time is
measured and reported without a countdown, per the user's earlier instruction.
The simulator-free GPU route still needs to run before throughput or time
estimates for this configuration can be established.

The launcher discovers compatible manifests in mounted /kaggle/input assets.
Explicit checkpoint input is also available through the runner CLI. Search of
unmounted Kaggle versions, historical branches, and upstream model releases
is not automated in this entry. The launcher records artifact hashes and
publishes curated files using the repository publisher. Raw files exceeding
GitHub's file limit and the full optimizer checkpoint remain in Kaggle Output.

Both parity arms reset Session state before Chunk 2. The comparison covers
serialized states, each layer's memory output, and full logits with fixed
FP32 tolerances. Exact normalized snapshot hashes are also reported. The
semantic worker compares Correct, Empty and Donor on 16 fixed synthetic
facts and reports a separate verdict.

Local verification covers CPU fixtures, syntax and applying the upstream
patch. No GPU results or semantic success have been established locally.
