# GC-001 repository audit (before implementation)

Audit base: f60730b2a0ae603e76a771a39830e9518382df81. Development, no GPU execution.

- `implementation/model.py`: C1 is exactly 16×128 FP32 persistent state, 2048 BPE
  vocabulary, ring radius 1, entry cell 0, alpha .1, synchronous shared residual
  update, K=4, concatenated state pool/readout. No token attention or KV cache.
- `implementation/runtime.py`: functional step/reset/evolve and identity-checked
  state-only snapshots already exist. Its legacy `generate(tokens,count)` is greedy,
  stateless between calls, returns IDs only, and lacks EOS/sampling/RNG continuation.
  Reuse this executor through a generation session subclass/factory; preserve legacy API.
- `trainer.py`: AdamW 3e-4, betas .9/.95, decay .1, warmup/cosine, norm clip 1,
  batch 4/TBPTT 64. State is detached between truncations but retained until document
  reset. Shared v2 checkpoint has atomic payload+checksum and RNG/data cursor/optimizer
  restore; model/token/source/precision identities prevent incompatible reuse.
- `data.py` + Native tokenizer: ByteLevel BPE, special-token *names* `<pad>`, `<unk>`,
  `<bos>`, `<eos>`. IDs must be read from recovered tokenizer, not assumed. Dataset
  builder prepends BOS and appends EOS, trains next-token pairs including EOS targets,
  deduplicates normalized content hashes, and partitions fixed TRAIN-revision stories
  by content hash mod10. Assets are pinned by manifest/file SHA-256.
- There is no independent Test split. Historical tokenizer training may overlap the
  development stories. Do not silently create an independent-test claim. GC protocol
  retains development[:16] as validation and development[16:] as an **exploratory
  development holdout**, audits content/token duplicates and boundaries before training,
  and reports formal Test NLL/PPL as NOT_AVAILABLE. Confirmation remains sealed.
- `frozen_model.py`: C1 immutable byte/weight/config validation and versioned private
  Output recovery already work. Extend optional asset selection (train data/F0) without
  accepting any incompatible optimizer checkpoint. Prefer known saved version 142.
- Existing C1/F0 result branch `kaggle/epclm-core-001-398a191dbb32` was inspected:
  each 250K tokens/977 updates, common source/protocol/data/seed. C1 966784 parameters;
  F0 1493120, 2048-number state, nonlinear dense low-rank recurrence. Reuse both final
  checkpoints; do not retrain an already completed Flat comparison. F0 is state/token
  matched, **not parameter matched**. No strict compute-matched claim is justified.
- Existing C0 is neighbors-replaced-by-self, not center-only message removal. Add only
  the missing trained center-only baseline, keeping input/readout/K/state/parameter
  shapes unchanged and documenting its inactive left/right parameters. Frozen C1
  center-only inference is B1; separately trained baseline is B2.
- Native `native_clm_runtime.ModernTransformerLM` is mature T1: width192, 4 layers,
  GQA 6/2 heads, FFN480, RoPE/RMSNorm/SwiGLU, tied embeddings, context128. Reuse it for
  the optional second batch; it has no incremental KV implementation. A bounded-token
  adapter may replay its **reference-only** context and must report KV=0/disabled and
  its actual context/state costs. This mechanism must never enter C1 generation.
- X4 lives in Native phase C and is a historical local-attention CLM, not Core C1.
  GRUCell modules exist inside historical staged token models; no standalone compatible
  persistent GRU/LSTM/SSM baseline was found in the inspected Core/Native implementation.
  Reusing those staged sequence models as a 2048-state RNN would change the comparison.
- Existing diagnostics/Inspector remain independent. Existing Kaggle secret labels,
  frozen recovery, trainer, deadline, immutable bundle and metadata-only publisher
  are reusable. New GC feature/result branches must preserve all old launchers/results.

Necessary additions: sampling helpers; stateful generation session over Runtime;
identity-bound generation snapshots with private sampling RNG; seven small CPU
contracts; fixed prompts, generation/quality/performance evaluation; center-only
training baseline; optional T1 adapter; audited development split; bounded one-launch
Kaggle orchestration and isolated GC publication. No change to C1 math or weights.
