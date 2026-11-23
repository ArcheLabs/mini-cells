# MiniCells MoM-Native Cells

Local product MVP. One frozen MoM model serves every Cell. The persistent
state is the FP32 native recurrent matrices for the 24 layers and five banks.
A bank that never received tokens is represented by its native absent state.
A new document or query starts a stream at offset zero with empty short
convolution caches. Within a document the native cache persists across
chunks. The Web writer checkpoints complete stream state after each chunk. Resubmit
the same text and chunk size against the same revision to resume.

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
python -m pip install httpx
python -m unittest -v test_runtime
python acceptance.py /path/to/mom /path/to/model-snapshot /path/to/tokenizer-snapshot /path/to/new-evidence
```

The real-model acceptance saves answers for no Cell, A, B and A+B,
tests native versus one-branch fusion, and imports then appends knowledge
in another process. Review the saved answer text for actual knowledge
recall. GPU acceptance and browser walkthrough remain pending. The local
unit tests exercise fusion placement on a synthetic branch, lineage and
the portable bundle envelope. Mid-document CUDA resume, full metrics, and
retention across knowledge replacement still require additional work.

## Recall Diagnostic v1

The initial T4 engineering run produced incorrect free-form answers.
See [the original run](https://github.com/ArcheLabs/mini-cells/tree/kaggle/mom-native-cells-fd68aeab620d/artifacts/experiments/mom-native-cells-fd68aeab620d).

Run `kaggle_recall.py` as a Python Script on one T4 with Internet and
GITHUB_TOKEN. The runner scores the original two facts in Cloze and QA
formats, with separate founder, prize-year and joint questions. Each cell
is written once per run. Full-cache and recurrent-only states come from
that exact write. The joint full-cache arm fuses the two live streams
layer by layer. A/B-only controls use the recurrent-only path.
Candidate scores use separately cloned or freshly initialized caches.
Reports include total and mean token log probability, ranks, ties and
margins. Public facts may also be present in the pretrained model.

For a follow-up, pass `--ablation` to recall_diagnostic.py to isolate
convolution reset and position reset. The default remote run covers the
three core arms and fresh-process import. It does not rerun free generation.

## MoM Compositional Recall v2

Run `kaggle_compositional_recall.py` as a Kaggle **Python Script** with a T4 or
other CUDA GPU, Internet access, and the existing `GITHUB_TOKEN` secret. One
launch runs the CPU checks, verifies the pinned dependencies, executes the
paired diagnostic and returns results to a verified GitHub artifact branch.

The development fixture contains 16 disjoint two-hop groups across four seeds,
with both projects queried per group. A stores project-to-relay mappings. B
stores relay-to-code mappings. Both swaps change values while preserving fact
order and the candidate set. All four combinations are scored before and
after a fresh process imports the individually exported recurrent states.
No Cell and each single-Cell variant are scored once per query and evaluated
against the same counterfactual gold assignments. Each answer starts with
new query caches and dual-Cell inference uses the existing equal-weight fusion.

The primary metric is total teacher-forced answer log probability. Both
candidates have exactly the same token count within each query. Ties are
counted separately and do not earn Top-1 credit. A complete factorial result
for a query requires a positive gold margin in all four combinations. A
complete group requires both queries to pass this check. The report includes
individual margins, per-token scores, both process results, state checksums
and per-seed counts. Bundles are temporary experimental inputs.

This run is development-only. Diagnostic completion indicates that all arms
were scored. The factorial and single-Cell controls determine the strength of
the compositional recall evidence.

## Python SDK

Install the `mom` extra and use `from minicells import MiniCells`. The Web Demo
shares this runtime. The query endpoint defaults to automatic selection.
The `/api/query` legacy weighted request remains available for manual
comparisons. See [installation](../../docs/python-sdk/installation.md) and
[the Python API](../../docs/python-sdk/python-api.md).
