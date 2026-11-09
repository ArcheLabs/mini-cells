# M3 已正式冻结 — 禁止追加训练

数学、实现与实验身份见 [冻结记录](freeze/FREEZE.zh-CN.md)。
权重保全状态见 [ARCHIVE-STATUS.json](freeze/ARCHIVE-STATUS.json)。
只运行 [archive-only.ipynb](archive-only.ipynb) 进行 CPU 归档；下文训练命令是历史说明，不再放行。

# M3 — Titans MAG-lite (`mag_lite_single_memory`)

One shared dynamic Memory, unchanged M2 E1 SWA, FP32, two real outer updates.
M1/M2 are immutable; M4–M6 and 20M/100M training are excluded. H3 remains
`NOT_EVALUATED`. See [REPORT.zh-CN.md](REPORT.zh-CN.md) for actual evidence/status.

## Mathematics and provenance

Source: [Titans, original paper §§3.1–3.2, 4.2, equations 11–15 and 26–28](https://arxiv.org/html/2501.00663v1),
[NeurIPS 2025 paper](https://papers.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf).
This is a declared MAG-lite adaptation, not the authors' official implementation
or numerical-equivalence claim. It does not use the damaged specification formulas.

For chunk c, projected keys k and values v define

```
L_c(M) = mean_over_tokens sum_over_features (M(k) - v)^2
S_c = eta_c * S_(c-1) - theta_c * grad_M L_c(M_(c-1))
M_c = (1 - alpha_c) * M_(c-1) + S_c
```

Each batch stream has independent M and S. Batch losses sum, so inner updates
are invariant to batch size. `sigmoid(controls(mean(k)))` supplies eta in (0,1),
theta in (0,0.05), alpha in (0,0.01). Chunk means, one update per chunk and
chunk-level controls are explicit choices; this is **not** an exact parallel
reproduction of token-dependent recurrence from the paper. `M` is a two-layer
GELU MLP with residual and LayerNorm (including fast affine weights).
Q/K/V use SiLU, Q/K are L2-normalized; QKV depthwise convolution is omitted.

All reads in a chunk use **preceding-chunk** M; only then may that chunk write.
Current token never reads its own new write. No future chunk contributes to an
output. The last partial chunk also writes after its reads. `stream` calls define
chunk boundaries; arbitrarily splitting a chunk across calls changes write timing.
No full-versus-token-cache equivalence is claimed for E4.

MAG gate is a scalar sigmoid per token. It mixes normalized SWA hidden features
and normalized projected Memory reads. Implementation distributes the tied linear
LM head across this mix, which is algebraically equivalent. A single shared
Memory reads/writes token embeddings in parallel with the complete E1 branch;
there is no per-layer Memory or three-way fusion. This is a simpler gate than
full Titans MAG; no persistent-prefix-token component is added.

## Gradient contract and modes

- `meta_train`: static parameters train; functional Fast Weights and Momentum
  updates use `autograd.grad(create_graph=True)`. All chunks in the episode
  retain their full high-order graph. There is **no first-order approximation**
  and no internal TBPTT/detach boundary in an episode.
- `online_adapt`: static parameters freeze; keys/values/controls and old dynamic
  tensors detach at each update. Only Fast Weights and Momentum update. This is
  explicitly an adaptation mode, not a substitute for meta-training.
- `read_only`: static parameters freeze; neither dynamic tensors nor revision
  changes. No gradients are recorded for recall.

Outer training starts a fresh Cell from the trainable initializer per episode;
there is no cross-episode/optimizer-step meta-gradient. Detached `last_cell` is
an observational runtime snapshot, never the next training episode's state.
Restored Cells are independent leaf tensors: checkpoint/session boundaries do
not preserve a historical autograd graph. Runtime reset discards only this
snapshot. Caller-owned Cells remain persistent. Inherited SWA-only `cached()` is
explicitly disabled to avoid silently omitting Memory.

## Persistence and isolation

Cell consists of Fast Weights, Momentum, update step and metadata. Static model
parameters are stored in the existing outer checkpoint. Memory checkpoint writes
independent hash-enveloped components into a new generation, then atomically
replaces the manifest. Compatibility includes static-model identity and interface;
format/shape/hash mismatches fail closed. Fork clones every tensor and metadata.
A Cell depends on neither Attention KV nor Mamba state. Working snapshots are
counted separately from static parameters and checkpoint bytes.

## Ordered gates and minimal execution

[tests/test_gates.py](tests/test_gates.py) and `run.py` use the **same five gate
functions**, not duplicate tests. Passed results cache under exact implementation,
protocol, Python and PyTorch identity. Failure stops subsequent gates. Gate 1
records centered finite-difference errors/tolerances in FP64 for the initializer
and Q/K/V/control projections, plus an independent three-step recurrence oracle.
Gate 5 is a four-key latent associative positive control, learned through the
actual inner gradient rule, saved, discarded and recalled in a fresh session.
It is not a vocabulary-level MQAR score or evidence for H3.

```bash
OMP_NUM_THREADS=1 python m3/run.py --cpu-only --output /tmp/epclm004-m3-local
```

On Kaggle T4 use [notebook.ipynb](notebook.ipynb), Internet and existing
`GITHUB_TOKEN` secret, then Save & Run All. There is **one launch**, no stage
approvals. Gates 1–5 → Gate 6 canonical 10M E4 → publication. No precision fallback.
Only minimal inherited dependencies are installed, no Mamba/FlashAttention build.
The raw log and all weights remain in Kaggle Output. Curated JSON/log/report files
are published through the existing SHA-verified publisher; M1/M2 are never edited.

Budget: 5,400 seconds execution (including CPU gates), 600 seconds publication
reserve, 7,200 seconds hard notebook bound. Timeout keeps the last durable prefix.
Completed updates resume using the unchanged M2 identity/checkpoint mechanism;
no fresh rerun of completed training. Two committed updates at B4 × 128 input
positions = 1,024 input tokens, eight supervised/query targets. Inner updates
are eight chunk transitions per batch (32 stream transitions), hence 16 batch
transitions / 64 stream transitions over two outer updates. Positive-control
adaptation updates are reported separately and never counted as training tokens.

GPU PASS requires actual Tesla T4, complete training ledger, finite gradients,
changed static/dynamic parameters, restored optimizer and logits, plus positive
control recall on the trained E4 Memory. Resource/timing scope is explicitly
reported; small Smoke timing is not a formal performance claim. Only a real GPU
PASS plus verified publication can produce `M3_ENGINEERING_COMPLETE`.
