# Installation

Use Python 3.11+ on Linux with a CUDA-enabled PyTorch build, a compatible Triton
installation, and Git on PATH. The documented GPU target is a Tesla T4 in FP16.
On an existing compatible CUDA stack, install from the SDK branch.

```sh
python -m pip install "mini-cells[mom] @ git+https://github.com/ArcheLabs/mini-cells.git@codex/minicells-python-sdk-001"
```

For a local checkout use `python -m pip install '.[mom]'`. Add the `web`
extra for the demo. PyPI publication is outside this release.

On first open the SDK fetches pinned MoM 340M weights, pinned tokenizer assets
and the audited upstream MoM commit. Set `cache_dir` to control where these
assets are reused. For an offline deployment, supply the original pinned model
snapshot directory as `model` and pass the pinned MoM Git checkout and tokenizer
snapshot through `source` and `tokenizer`. Keep the snapshot directory names
(their commit IDs). `MiniCells.open` validates the architecture and records file
hashes, tokenizer and runtime identity. The native Cell loader uses exact
identity compatibility, including the recorded execution dtype and versions.
