# MiniCells 0.8 implementation audit

The runtime reuses the EPCLM-004 Titans transition and EPCLM-006 Cell codec.
The Qwen adapter isolates Transformers 5.3.0 cache internals. In this version
the DeltaNet layer consumes existing state on single-token calls. Resumed
multi-token inputs therefore use sequential calls. Attention positions are
derived from the new KV cache length. The cumulative processed count is
retained separately. Runtime snapshots allow only recurrent and convolution
tensors and validate shapes and dtypes against the running model.

The pinned Qwen revision is 2fc06364715b967f1860aea9cf38778875588b17.
Install transformers==5.3.0 and run tests/test_minicells_runtime.py with pytest.
The test uses small randomly initialized Qwen weights and the actual Titans
implementation. The caller provides the shared projection, gate, and Cell
compatibility identity. Generation accepts token IDs and returns the pending
last token separately so callers can carry it into the next call.
