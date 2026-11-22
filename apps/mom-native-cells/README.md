# MiniCells MoM-Native Cells

Local product MVP. One frozen MoM model serves every Cell. The persistent
state is the FP32 native recurrent matrices for the 24 layers and five banks.
A bank that never received tokens is represented by its native absent state.
A new document or query starts a stream at offset zero with empty short
convolution caches. Within a document the native cache persists across
chunks. This version restarts a failed document from its last committed Cell.

## Run

Use Python 3.11+, a CUDA-enabled PyTorch and Triton build, Transformers 4.51.3,
FLA 0.2.2, Safetensors, FastAPI with Pydantic 2 and Uvicorn. Checkout OpenSparseLLMs/MoM
at `125038087b71b635c171cec857bd0f21d10955b6`. Download the snapshots named by the
immutable revisions in backend.py with Hugging Face Hub. Include model config,
Safetensors weights, and the matching tokenizer assets. The local loader checks
revision directories, exact architecture, weight loading and vocabulary.
The manifest records file hashes and runtime dependency versions.

After installing CUDA PyTorch and Triton, install the application dependencies
with `python -m pip install -r requirements.txt`. From this directory run:

```sh
python server.py /path/to/mom /path/to/model-snapshot /path/to/tokenizer-snapshot /path/to/cells
```

Open http://127.0.0.1:8000. BF16 is the default. Use `--dtype float16` explicitly
on a T4 for inputs of up to 64 tokens per forward. The pinned Chunk kernel
casts to BF16, so long chunks require a GPU with native BF16. Keep one server process per Cell storage directory. API schema is
available under /docs while the server is running.

## Execution and Cell format

The upstream layer normalizes the shared input, invokes native MoM attention,
then applies the shared decoder residual and MLP. The multi-Cell adapter
intercepts only the attention call. Each Cell has its own transient cache,
and the adapter weights native projected attention outputs in FP32 before
returning the fused contribution to the unmodified decoder. No Cell and
single Cell inference use the native forward directly.

Each exported ZIP stores manifest.json and recurrent_states.safetensors.
The manifest pins the model and tokenizer, records the Cell ID and revision
lineage, and checksums the tensor payload. Import validates compatibility,
shape, dtype, and finite values. Writes stage a new state in memory and
publish the completed bundle atomically. Queries read disposable copies.
The progress endpoint supports polling and cancellation between chunks.

## Acceptance

```sh
python -m unittest -v test_runtime
python acceptance.py /path/to/mom /path/to/model-snapshot /path/to/tokenizer-snapshot /path/to/new-evidence
```

The real-model acceptance saves answers for no Cell, A, B and A+B,
tests native versus one-branch fusion, and imports then appends knowledge
in another process. Review the saved answer text for actual knowledge
recall. GPU acceptance and browser walkthrough remain pending. The local
unit tests exercise fusion placement on a synthetic branch, lineage and
the portable bundle envelope. Mid-document resume, full metrics, and
retention across knowledge replacement still require additional work.
