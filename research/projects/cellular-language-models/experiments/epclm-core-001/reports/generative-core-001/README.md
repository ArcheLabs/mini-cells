# EPCLM GC-001 implementation and execution

Status: IMPLEMENTATION_READY; real GPU pilot, generation, baseline training and metrics NOT_RUN.
This record reports implementation, not successful language generation or scientific confirmation.

## Run the primary batch

Import [generative-core-001.ipynb](../../generative-core-001.ipynb) into the existing
private `bingjiangli/minicells` Notebook. Enable GPU, Internet, KAGGLE_API_TOKEN and
GITHUB_TOKEN, then Save & Run All. The feature branch is
`codex/epclm-generative-core-001`, separate from Core/CP and main.

One launch executes a tiny CPU contract gate, exact frozen C1/data recovery, data audit,
one 5-prompt×32-token greedy GPU pilot with real snapshot continuation, full frozen
C1 generation/quality/performance evaluation, B1 frozen center-only inference ablation,
250K-token trained center-only B2, and the already trained 250K-token F0. It publishes
separate GC results. It never retrains C1 or the completed F0, never runs a large CPU
training/benchmark, and does not automatically enter Transformer or 1M/10M training.
The original [specification](../../GENERATIVE-CORE-001.md) and before-edit
[repository audit](../../generative-core/generative-core-audit.md) are preserved.
The [GC protocol](../../generative-core-protocol.json) independently freezes choices.

## Existing Runtime extended, not replaced

`Runtime.generation(tokenizer, model_identity, tokenizer_sha, seed, eos_trained=True)`
returns `GenerationSession`, a subclass of the existing Runtime. Legacy functional
step/reset/evolve/snapshot/restore and legacy greedy generation are unchanged. The
session uses inherited Runtime.step for exactly one event at a time; it has no C1
attention, KV cache, prompt-token cache, historical-state list or prompt replay.
Only current state carries history into C1's readout. Generated IDs may be retained
for text display, stopping and return values; they never re-enter model computation
as a sequence.

- `prefill(prompt)` resets this session, inserts the audited BOS once (including empty
  prompt), consumes each prompt token once, and returns logits/count. An existing
  leading BOS is not duplicated. PAD/interior BOS/EOS in a prompt are rejected.
- `generate(max_new_tokens=128, temperature=0, top_k=0, top_p=1, stop_sequences=[])`
  uses current readout and private CPU sampling RNG. Greedy does not advance RNG;
  temperature/top-k/top-p operate on readout only. PAD/BOS are masked, real trained
  EOS stops generation. Every emitted token, including terminal EOS, is consumed
  once. EOS/control tokens are hidden from text; UNK remains visible for review.
  Max-length ends only the current call, allowing chunked continuation. EOS/stop
  strings terminally stop the session. Stop strings inspect generated output only;
  the final consumed BPE token may contain characters beyond a stop string, while
  displayed text is trimmed at the string. Return values include this call's IDs
  and the entire generated continuation text, not a guaranteed UTF8 character stream.
- `read()` does not mutate state, position, counters or RNG. `evolve(steps)` does no
  event injection and remains opt-in; normal C1 generation always uses K=4.
- `reset()` clears this session only and requires prefill/restore before generation.
  New sessions do not share state or RNG, although immutable weights are shared.
- Generation snapshots wrap the existing state snapshot and its checksum, with exact
  checkpoint identity, weight/config/tokenizer hashes, generation config, position,
  stop state, output IDs and private sampling RNG. Their own checksum binds metadata
  and RNG. Restore rejects wrong model/checkpoint/tokenizer or corrupt bytes. State
  snapshots contain no weights/optimizer; training checkpoints stay separate.
- C1 model/state/logits are FP32 with autocast disabled. Sampling uses CPU FP64
  probability calculations for robust low-temperature filtering; this does not
  change model precision. Same-environment determinism is checked, not promised
  across different GPU/library versions.

## Data correction and honest scope

Original pinned assets have train and development only, with potential tokenizer
training exposure to development. GC does not fabricate an independent Test or touch
sealed confirmation data. It freezes development[:16] as validation (reproducing
original C1 NLL 5.814626392302277) and development[16:] as an exploratory development
holdout. Before any new baseline training it validates asset hashes, all content IDs,
exact token-sequence duplicate absence, bounds and BOS/EOS/PAD document rules.
Real EOS training targets are verified. IDs are read from the actual tokenizer.

Both subsets are evaluated in full with every document reset and unchanged next-token
alignment. Formal Test NLL/PPL remain NOT_AVAILABLE. The holdout is explicitly named
and reports tokenizer exposure; it is not evidence of independent generalization.
This is the necessary protocol amendment for reusing frozen historical C1 weights.
A strict independent-Test benchmark would require separately authorized clean assets
and tokenizer/contamination review, not relabeling existing data.

## Baseline reuse and fairness

- C1: exact original 250K-token frozen checkpoint and pinned weight/outer-file hashes.
- B1-FROZEN-NO-COMM: same C1 weights, center-only message trajectory. Not an independently
  trained model and not a substitute for B2. Snapshot identity records the transition.
- NO-COMM (B2): the one missing training model; same original cells/dim/event/readout/
  shared update/K=4, but removes left/right messages in every step. It preserves
  parameter shapes; unused left/right parameters are frozen and reported. This is
  not historical C0, whose three projections read self. The serialized Core arm tag
  C0 is supplemented by explicit CENTER_ONLY model/source/protocol identity.
  Importantly, zero-initialized bias-free disconnected non-entry cells stay zero:
  allocated 2048 state numbers are not 2048 used dimensions. Do not hide this
  consequence or change event injection to all cells, which would add another factor.
- F0: reuse exact original source 398a191, same 250K tokens/977 updates/data/seed.
  Its state is a 2048-number flat vector with nonlinear low-rank dense recurrence;
  the [16,128] storage shape does not imply local cellular dynamics. It is a reasonable
  existing state-matched model, not an intentionally weak linear baseline. C1 has
  966784 allocated parameters versus F0 1493120: parameter matching is NOT claimed.
- All new training uses existing Trainer/AdamW/scheduler/clip/TBPTT64/batch4/FP32 with
  the same seed, fixed data order and document resets. State carries across truncation,
  gradients do not. Final 250K endpoint is fixed; test is never used for selection.
  A predeclared first50-versus-last50 mean training-loss improvement >=.01 gate stops
  later work on obvious nonconvergence; no automatic hyperparameter search follows.
- Report token/state matching, allocated/trainable parameters, inactive dimensions,
  analytical linear-FLOP estimates (not measured FLOPs), actual training GPU time,
  end-to-end inference time and memory separately. No compute-matched comparison is
  claimed, no single-seed difference establishes architecture superiority.

## Optional second batch

Only after a compatible primary batch is published, import
[generative-core-001-transformer.ipynb](../../generative-core-001-transformer.ipynb).
It fetches/verifies the same-source/protocol primary result, reuses its C1 pilot and
baseline summaries, then trains/evaluates existing Native T1 for 250K tokens. It does
not repeat the C1 GPU pilot or rerun completed primary generation. Missing/incompatible
primary results stop the second batch. This is a separate result namespace/branch.

The adapter reuses Native ModernTransformerLM (192 dim, four layers, 6 query/2 KV
heads, FFN480, RoPE/RMSNorm/SwiGLU, tied embeddings) rather than building a new attention
model. It retains at most 128 token IDs, replays its bounded context, and has no
incremental KV cache (0 bytes, DISABLED). Padding never becomes fake history: each
lane uses its actual length. Context/parameter/configuration, token buffer bytes,
training objective, prefill/decode costs and context replay are reported explicitly.
Its state-buffer shape is not its hidden width. It is a reference with a different
history access mechanism, exempt from C1's no-history-replay contract. The first batch
reports T1 NOT_RUN. Resource exhaustion leaves incomplete T1 excluded from conclusions.

## Evaluation, recovery and reports

All models use the same 16 fixed prompts, greedy plus temperature .8/top-k40/top-p.95,
seeds and 128-token maximum. EOS may produce fewer tokens; exact emitted count and
stop reason are reported. All generated text is kept, including poor examples. Metrics
include repeated trigram occurrence rate, distinct-1/2, mean raw token entropy, length,
EOS rate, latency and throughput. Metrics do not certify coherence: human review remains
NOT_EVALUATED until actual samples are read. C1's first full sampling prompt also checks
20+snapshot+restore+108 versus continuous generation on GPU; verification cost is separate
from decode time. Early EOS and actual post-restore token count are recorded.

Performance measures 32/128/512/1024 consumed prompt tokens, prefill total/per-token,
one-token decode, fixed latent-state bytes, nonzero state blocks and snapshot size after
prefill. It reports whole 128-max generation times for every sample, not a fabricated
128-token time when EOS stops early. C1 prefill is linear in consumed events despite
fixed state size. Session snapshot UI outputs can grow with generated length; the
history-scaling snapshot is taken immediately after prefill and stores no C1 prompt
history. CPU memory and peak VRAM are process high-water marks, not isolated model-only
allocations. GPU utilization is NOT_MEASURED when no reliable measurement exists.

Recovery uses existing authenticated versioned Kaggle transport, known version 142,
immutable manifests, file hashes and load_checkpoint. Extra original train/F0 assets
are verified. Only GC explicitly opts into `.pt` current-source progress recovery;
read-only diagnostic callers still reject optimizer checkpoints. Restored GC baselines
must match current source/protocol/data/model/precision before use; old-source C1/F0
supply frozen weights only, never their optimizer states. Completed checkpoints and
evaluated model/sample/split results are skipped. Failed publication retains results;
COMPLETE retries publication only. A later manual launch may retry transient asset/primary-result acquisition after archiving the blocked report; it does not retrain finished work. Numerical/integrity failures remain terminal. Incomplete training is never included as COMPLETE.

Budget: one run, <=5400s execution (training/evaluation/checkpointing included), <=600s
asset recovery within that execution budget, 600s publication reserve, 1200s safety,
7200s Notebook hard limit. Per-event deadlines plus outer interruptible timers stop work;
no CPU fallback training, hidden pilots or automatic 1M/10M expansion.

Public output is metadata/aggregate metrics, full generated samples, comparison,
performance, audit and source/config identity only. No raw train/development documents,
weights, optimizer state or runtime snapshots are published. Existing publisher stages
results only on main descendants and verifies remote SHA. Primary branch:
`kaggle/epclm-gc001-<source12>`; reference: `kaggle/epclm-gc001-transformer-<source12>`.
A private publication receipt records verified result commit; runtime engineering pass
is distinct from publication and scientific confirmation.

## Local validation and remaining gates

Seven tiny synthetic generation contracts cover prefill/manual equivalence, exactly-once
consumption, immutable read, greedy and sampling snapshot continuation, session isolation,
EOS/stop, and no history replay. One focused asset regression covers additional train/F0
retrieval, known-version priority and explicit current-source optimizer checkpoint opt-in.
A second focused reference-adapter check verifies native last-token equivalence, bounded context and snapshot identity without training. The existing CPU integration contract is retained. No full CPU language training or CPU
performance benchmark was run. GPU pilot, real generation samples, new baseline training,
latency, memory and quality remain pending the user's Kaggle execution. H1–H5 stay
NOT_EVALUATED, and scientific claims remain limited even if engineering execution completes.
