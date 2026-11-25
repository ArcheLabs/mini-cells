# Native MoM 1.3B runtime compatibility

The pinned Kaggle checkpoint audit passed. The runtime implementation has
passed local structural and software tests. Actual 1.3B weight loading through
the runtime and native forward execution still need a real run.

| Area | Existing runtime | 1.3B implementation and verification |
| --- | --- | --- |
| Model registration | Pinned 340M model and separate tokenizer | Registry resolves the exact 1.3B snapshot and its own tokenizer. A unit test checks both routes and rejects unknown models. |
| Model dimensions | Hidden size 1024, 24 layers | Hidden size 2048 and 24 layers. Native projected width remains 1024. Meta initialization matched all 843 parameter names and shapes. |
| Routed and shared state | Four routed banks and one shared bank | Runtime derives layer count, bank count, and recurrent shape from config. Each working cache gets independent tensor copies. |
| Embedding and LM head | Native MoM output projection | Full manifest checks both checkpoint copies before Transformers binds them. Post-load checks enforce a shared parameter, unique parameter count, dtype and device. |
| Tokenizer | Legacy AutoTokenizer with its own pinned snapshot | The 1.3B path opens the checkpoint's exact tokenizer JSON and verifies vocabulary size. Local token IDs match the serialized tokenizer before and after BOS insertion. |
| Cache and SDK | Independent read streams, Automatic Selection, equal selective fusion and no-memory mode | The SDK continues to use the same paths. The 1.3B checkpoint has recurrent and convolution state. It has no dense attention KV cache. Cell bundles continue to validate full model identity. |
| Precision | FP16 selected by default on CUDA; native recurrent state stored as FP32 | BF16 checkpoint is verified before runtime conversion. The 340M kernel choices and dependency pins are reused. |
| Experimental learned fusion | Learned gates were built for the 340M width | Existing gates compare hidden and value vectors in the same space. Those checkpoints are outside the 1.3B SDK equal-fusion path. |

## Local validation

- 41 tests passed across the Native MoM app, including legacy 340M asset
  resolution, Cell lifetime, query selection, fusion and result publication.
  Legacy 340M GPU inference was not repeated.
- The exact pinned MoM Python classes constructed the 1.3B model on Meta with
  shape-only FLA stand-ins. The test checked all parameter names and shapes,
  the tied output and input, and 1,552,884,000 distinct parameters. It did not
  load weights or execute any native kernel.
- The real tokenizer ran on CPU with the pinned local files.

## Gates

| Check | Status |
| --- | --- |
| Model configuration | PASS |
| Architecture compatibility and tensor shape validation | PASS on Meta |
| Model initialization | PASS on Meta |
| Strict-loading rejection tests | PASS |
| Real checkpoint load through the SDK | NOT_RUN |
| Native CPU forward | N/A under the approved acceptance change |
| 340M software regression | PASS |
| T4 single-GPU preflight and A–F capacity | NOT_RUN |

The approved protocol uses the independent T4 preflight for real checkpoint
loading, native FLA execution, and logits checks.
