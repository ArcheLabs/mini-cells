[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE — ONLINE HISTORICAL ADDRESS-STATE INTEGRATION**

Stage 06 将 Constructive CLM 已正式支持的机制带入真正的 token-predictive model。

## 固定路线图

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🟢 COMPLETE
  M2  fixed-topology replay-free continual language      🔴 NOT SUPPORTED
      certificate protection 减少 forgetting             🟡 PARTIAL EVIDENCE
  M3  global-pool growth-restored continual language     🔴 NOT SUPPORTED
  M3R read-preserving / lineage-isolated growth          🔴 NOT SUPPORTED
      root read ownership 得到保持                       🟡 PARTIAL EVIDENCE
  M3R Address Diagnostic                                 🟢 QUERY GEOMETRY SEPARABLE
  M3L query-sketch lineage gate                           🔴 NOT FEASIBLE
  M3L-1 historical address-state capacity                🟢 LOW_RANK_CAPACITY_SUFFICIENT
      minimum passing low-rank state = 32
  M3L-2 online historical address-state integration      🔵 FROZEN / UNRUN
  M4  Cell ontology / specialization analysis            ⚪ BLOCKED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

在 M3L-2 得到有效 formal decision 之前，不升级到 30M，也不进入 M4。

## Canonical substrate

```text
M1 起始参数                 12,154,368
vocab                       256 UTF-8 bytes
context                     256
shared width                384
shared blocks               6
attention heads             6
FFN width                   1536
Cellular Layers             1
initial Cells               8
active Cells/token          2
Cell operator               384 × 384 linear residual
certificate max rank        64
```

Canonical M1 checkpoint：

```text
Hugging Face  archelabsxyz/native-clm-v0
file          final-model.pt
SHA-256       91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

## M0 — Architecture + execution — 🟢

M0 已完成 sparse routing、Cell-local gradients、certificate projection、dynamic spawn、optimizer enrollment、动态 checkpoint round-trip 与 generation。

## M1 — 真实 next-token training — 🟢

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

详见 [M1_CLOSURE.md](M1_CLOSURE.md)。

## M2 — Fixed-topology continual language — 🔴 NOT SUPPORTED

正式结论：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73211 / 73212 / 73213
```

Certificate protection 有稳定因果收益：

```text
protected mean forgetting     ~0.2115
unsafe mean forgetting        ~0.2790
retention advantage           ~0.0675
protected/unsafe plasticity    ~0.964
```

但 protected A regression 仍约 43.9%，远高于注册的 <=20% 上限。详见 [M2_CLOSURE.md](M2_CLOSURE.md)。

## M3 — Global-pool growth — 🔴 NOT SUPPORTED

正式结论：

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73411 / 73412 / 73413
```

Growth 成功增长到 16 Cells、child reuse 100%、plasticity 保持，但 A regression 从 fixed control 的约 43–44% 恶化到 growth 的约 48–49%。

Post-formal analysis 表明，child 进入 global Top-K 后占据了大约一半的旧域 Cell execution。因此得到新的边界：

```text
safe write growth requires safe read-address growth
```

详见 [M3 formal result](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)。

## M3R — Read-preserving / lineage-isolated growth — 🔴 NOT SUPPORTED

正式结论：

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
protocol = c3e73545899ccf20f54411df701f22dd64b10cb46ff728e862c2d002a94f8627
seeds = 73611 / 73612 / 73613
artifact commit = 986b043a5d2f5ee9140cf35b14f68aacc3b7a942
HF revision = a23b521e137a7e44616809895d44d87cc7d6f87f
```

M3R 将 M1 的原始 8 个 roots 固定为唯一 top-level routing candidates，child 只能在所属 root lineage 内竞争。

三个 seed 全部满足：

- A/B/C/D 的 root-route probe hash 从 initial 到 after-B/C/D 完全不变；
- birth 时 root Top-K ownership 与 root probability 保持；
- growth 到 16 Cells；
- child reuse 100%；
- B/C/D plasticity、zero replay、sparse execution 均保持；
- lineage routing 相比 matched global-growth control 对 A retention 带来约 2.5 个百分点的稳定改善。

但 registered absolute retention target 仍未达到：

| seed | global A regression | lineage A regression | A advantage | lineage forgetting |
|---:|---:|---:|---:|---:|
| 73611 | 0.4967 | 0.4722 | 0.0245 | 0.2106 |
| 73612 | 0.4947 | 0.4691 | 0.0256 | 0.2089 |
| 73613 | 0.4961 | 0.4713 | 0.0248 | 0.2098 |

Child 仍缺少足够 selectivity，最终 child execution share 约为：

```text
A   ~52.3%
B   ~55.1%
C   ~59.5%
D   ~53.6%
```

因此 M3R 解决了 **global root ownership**，但没有解决 **lineage 内 parent/child functional boundary**。其 local decision 仍然只是：

```text
q · k_child > q · k_parent
```

这与 Core 006/007 的早期警告一致：representation/query similarity 不会自动成为安全的 functional mitosis address。

## M3R Address Diagnostic — 🟢 QUERY GEOMETRY SEPARABLE

Checkpoint-only diagnostic 在 24/24 个 lineage edges 上获得有效 coverage。当前 cosine addressing median AUC 约 0.5315，而自由 affine query probe 达到约 0.9623。因此 query geometry 中存在 boundary，但 centroid/cosine decoding 无法恢复它。

## M3L — Replay-Free Query-Sketch Gate — 🔴 NOT FEASIBLE

在更严格的 temporal parent-lifetime ownership 与 sequence-group-heldout split 下，offline affine oracle 仍可分（median AUC 0.9281）。rank-16 Gaussian historical query sketch 达到 0.8968，略低于冻结的 >=0.90 gate，同时其它注册 feasibility metrics 全部通过。因此 M3L 保持有效的 negative mechanism diagnostic。

## M3L-1 — Historical Address-State Capacity — 🟢 LOW_RANK_CAPACITY_SUFFICIENT

M3L-1 固定 M3L 的数据、edge ownership、split、oracle 与 feasibility thresholds，扫描 diagonal/rank-8/16/32/64/128/full-covariance Gaussian historical address state。

已完成 diagnostic 的分类为：

```text
LOW_RANK_CAPACITY_SUFFICIENT
minimum passing rank = 32
publish commit = 5ace6faf344b1b805752a33ffb861aeaf34dad6e
```

这解决了 checkpoint-level capacity selection：冻结的 Gaussian second-order family 不需要 dense covariance，但 rank 16 对注册规则而言容量不足。这个结果本身**不是** online continual-learning success。

## M3L-2 — Online Historical Address-State Integration — 🔵 FROZEN / UNRUN

M3L-2 是新的 registered formal experiment。它将 exact M3R lineage-cosine algorithm 与同样的 protected-write/growth algorithm 对比，唯一新增机制是 persistent rank-32 historical query state 和 affine lineage-local parent/child gate。

注册的 address state：

```text
rank                         32
query width                  384
maximum persistent bytes     52,360 / Cell
current queries / leaf/batch <= 256
bootstrap batches            160
```

由于历史 M1 checkpoint 产生时还没有 address sidecar，M3L-2 明确注册一次 pre-continual bootstrap：从 pinned TinyStories revision 的 `train` split 中读取 10,000 documents，仅构造原始 8 roots 的 sidecar。它不执行 optimizer、parameter、certificate 或 growth update。One-shot bootstrap handle 在 B 开始前释放；A retention 继续使用独立的 TinyStories `validation` split。

Continual start 之后 learner replay 保持为 0。每 50-step growth check，当前 routed-query sufficient statistics 要么合并进尚未分裂 leaf 的 rank-32 state，要么在该 leaf 分裂时成为 child 的初始 state；parent 的 pre-window history 在 split 后冻结。Child operator 精确复制 parent，因此 birth 仍要求 function-preserving。

新的 untouched formal seeds：

```text
74211 / 74212 / 74213
```

Canonical protocol、执行方式和 evidence boundary 详见 [M3L2_REGISTRATION.zh-CN.md](M3L2_REGISTRATION.zh-CN.md)。

## Evidence boundary

M2/M3/M3R formal seeds 均已消耗，禁止再次作为 untouched evidence。M3R Address Diagnostic、M3L 与 M3L-1 都只是 checkpoint-only mechanism diagnostics，不能事后修改任何 M3R formal gate 或 decision。

M3L-2 formal seeds 在 canonical two-GPU Kaggle runner 被明确执行之前保持 untouched。M4 ontology analysis 与 30M scale-up 均继续 blocked，直到 M3L-2 获得有效 formal decision。
