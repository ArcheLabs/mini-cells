# Native MoM 1.3B model identity audit

Phase 1 is **BLOCKED**. The designated revision is
[`5c8be94196cb1827e820077680ac2ee46ac949bc`](https://huggingface.co/linear-moe-hub/MoM-Gated-Deltanet-1.3B/tree/5c8be94196cb1827e820077680ac2ee46ac949bc).
The pinned snapshot lists one 3,236,933,080-byte Safetensors file and no shard
index. The checkpoint does not ship Python model code. The intended runtime
uses the repository's pinned MoM source at `125038087b71b635c171cec857bd0f21d10955b6`.

| Check | Result | Evidence |
| --- | --- | --- |
| Model identity and revision | PASS | The pinned upstream asset inventory matches the requested checkpoint. |
| Configuration | PASS | All five JSON files match their pinned Git blob IDs. |
| Tensor layout | Header verified | All 843 tensor names and shapes match the audited layer layout. All are BF16. |
| Parameter count | Header verified | 1,618,420,000 stored elements. The tied embedding and output matrices have equal observed SHA-256, leaving 1,552,884,000 distinct model parameters. |
| Checkpoint integrity | BLOCKED | The full file did not arrive. The expected LFS SHA-256 is in the manifest. |

The 24-layer checkpoint uses a hidden size of 2048 with a native Q/K/V width
of 1024 (four heads of 256). Each layer has four routed banks, a shared bank,
short convolution width 4, and a 5632-wide MLP. The tied input/output matrix
has shape 32000 × 2048. The config has no standard attention blocks.

The pinned tokenizer has 32,000 tokens, BOS 1, EOS 2, and PAD 2. With
Transformers 4.51.3, the existing `AutoTokenizer` call tries to load a
SentencePiece model absent from the snapshot. A CPU check against the
serialized `tokenizer.json` confirmed that the fast tokenizer can reproduce
the published backend's token IDs with and without special tokens. The loader
integration is still pending.

The full-file HTTP transfer reached the embedding and LM-head ranges before
timing out. Ranged requests independently recovered the header, but subsequent
transfers stalled or timed out. The current
[model manifest](model_manifest.json) records the scope of the observed data.

From the repository root, regenerate the full receipt when all six pinned
files are available in the revision-named directory. Use Python 3.11 or newer.

```bash
python apps/mom-native-cells/audit_native_mom_13b.py \
  /path/to/snapshots/5c8be94196cb1827e820077680ac2ee46ac949bc \
  --output docs/python-sdk/model_manifest.json
```

Three CPU audit tests passed, covering exact shape and key checks, duplicate
keys, tied-weight equality, and payload length. GPU preflight and capacity
gates A–F are NOT_RUN.

## Kaggle CPU entry

Import [the Notebook](../../apps/mom-native-cells/kaggle_mom_13b_checkpoint_audit.ipynb)
as a notebook, or upload [the Python file](../../apps/mom-native-cells/kaggle_mom_13b_checkpoint_audit.py)
in Script mode. Both contain the same standalone source. Select **Accelerator:
None** and **Internet: On**, then use **Save & Run All**. Model files are cached
under `/tmp`. The job writes `/kaggle/working/model_manifest.json` and prints
`CHECKPOINT_AUDIT=PASS` or `CHECKPOINT_AUDIT=BLOCKED`.
